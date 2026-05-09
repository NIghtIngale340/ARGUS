"""Unit tests for ARGUS attack classifier and alert engine."""

import json
import time
from pathlib import Path

import pytest
import torch

from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig
from src.inference.alert_engine import (
    Alert,
    AlertEngine,
    RedisUEBARiskStore,
    ScoredSession,
    UEBARiskStore,
    load_technique_severity,
)


class TestARGUSClassifier:
    """Tests for the classification head on BERT."""

    def test_forward_returns_correct_shape(self) -> None:
        config = ArgusBertConfig(vocab_size=100, max_seq_len=8, hidden_size=64,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=128)
        model = ARGUSClassifier(config=config, num_classes=2, freeze_layers=1)
        ids = torch.randint(0, 100, (4, 8))
        mask = torch.ones(4, 8, dtype=torch.long)
        logits = model(ids, mask)
        assert logits.shape == (4, 2)

    def test_freeze_layers_disables_gradients(self) -> None:
        config = ArgusBertConfig(vocab_size=50, max_seq_len=8, hidden_size=64,
                                  num_hidden_layers=4, num_attention_heads=2,
                                  intermediate_size=128)
        model = ARGUSClassifier(config=config, freeze_layers=2)

        for param in model.bert.embeddings.parameters():
            assert not param.requires_grad

        for param in model.bert.encoder.layer[0].parameters():
            assert not param.requires_grad
        for param in model.bert.encoder.layer[1].parameters():
            assert not param.requires_grad

        for param in model.bert.encoder.layer[2].parameters():
            assert param.requires_grad

        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_count_trainable_params(self) -> None:
        config = ArgusBertConfig(vocab_size=50, max_seq_len=8, hidden_size=64,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=128)
        model = ARGUSClassifier(config=config, freeze_layers=1)
        trainable, total = model.count_trainable_params()
        assert 0 < trainable < total

    def test_multiclass_output(self) -> None:
        config = ArgusBertConfig(vocab_size=50, max_seq_len=8, hidden_size=64,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=128)
        model = ARGUSClassifier(config=config, num_classes=5, freeze_layers=0)
        ids = torch.randint(0, 50, (2, 8))
        logits = model(ids)
        assert logits.shape == (2, 5)

    def test_gradient_flows_through_unfrozen_layers(self) -> None:
        config = ArgusBertConfig(vocab_size=50, max_seq_len=8, hidden_size=64,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=128)
        model = ARGUSClassifier(config=config, freeze_layers=0)
        ids = torch.randint(0, 50, (2, 8))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        assert model.classifier.weight.grad is not None

    def test_load_pretrained_bert_from_mlm_checkpoint(self, tmp_path: Path) -> None:
        """Verify checkpoint loading matches the Phase 2 pretrain.py format."""
        from src.models.argus_bert import ArgusBertForMaskedLM

        config = ArgusBertConfig(vocab_size=50, max_seq_len=8, hidden_size=64,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=128)

        mlm_model = ArgusBertForMaskedLM(config)
        fake_checkpoint = {"model": mlm_model.state_dict(), "global_step": 100}
        ckpt_path = tmp_path / "checkpoint_step_000100.pt"
        torch.save(fake_checkpoint, ckpt_path)

        assert any(k.startswith("model.bert.") for k in fake_checkpoint["model"])
        assert any(k.startswith("model.cls.") for k in fake_checkpoint["model"])

        classifier = ARGUSClassifier(config=config, num_classes=2, freeze_layers=1)
        classifier.load_pretrained_bert(str(ckpt_path))

        ids = torch.randint(0, 50, (2, 8))
        logits = classifier(ids)
        assert logits.shape == (2, 2)

        original_emb = mlm_model.model.bert.embeddings.word_embeddings.weight.data
        loaded_emb = classifier.bert.embeddings.word_embeddings.weight.data
        assert torch.equal(original_emb, loaded_emb)


class TestUEBARiskStore:
    """Tests for the EWMA risk tracking."""

    def test_default_risk_is_zero(self) -> None:
        store = UEBARiskStore()
        assert store.get_risk("user_unknown") == 0.0

    def test_risk_increases_with_high_scores(self) -> None:
        store = UEBARiskStore(decay=0.9)
        risk = store.update_risk("user_a", 10.0)
        assert risk > 0.0

        risk2 = store.update_risk("user_a", 10.0)
        assert risk2 > risk

    def test_risk_decays_with_low_scores(self) -> None:
        store = UEBARiskStore(decay=0.9)
        for _ in range(10):
            store.update_risk("user_b", 12.0)
        high_risk = store.get_risk("user_b")

        for _ in range(10):
            store.update_risk("user_b", 0.1)
        low_risk = store.get_risk("user_b")

        assert low_risk < high_risk

    def test_get_all_risks(self) -> None:
        store = UEBARiskStore()
        store.update_risk("u1", 5.0)
        store.update_risk("u2", 3.0)
        all_risks = store.get_all_risks()
        assert "u1" in all_risks
        assert "u2" in all_risks
        assert len(all_risks) == 2


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    def ping(self) -> bool:
        return True

    def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get(key, {}).get(field)

    def hset(self, key: str, field: str, value: str) -> None:
        self.hashes.setdefault(key, {})[field] = str(value)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.hashes.pop(key, None)


class TestRedisUEBARiskStore:
    def test_risk_persists_across_store_instances(self) -> None:
        client = FakeRedis()
        store_a = RedisUEBARiskStore(client=client, decay=0.9)
        first_risk = store_a.update_risk("user_a", 15.0)

        store_b = RedisUEBARiskStore(client=client, decay=0.9)

        assert first_risk > 0.0
        assert store_b.get_risk("user_a") == pytest.approx(first_risk)

    def test_get_all_risks_reads_redis_hash(self) -> None:
        client = FakeRedis()
        store = RedisUEBARiskStore(client=client, key_prefix="test:ueba")
        store.update_risk("u1", 15.0)
        store.update_risk("u2", 7.5)

        risks = store.get_all_risks()

        assert set(risks) == {"u1", "u2"}
        assert risks["u1"] > risks["u2"]

    def test_clear_deletes_risk_state(self) -> None:
        client = FakeRedis()
        store = RedisUEBARiskStore(client=client)
        store.update_risk("u1", 15.0)

        store.clear()

        assert store.get_all_risks() == {}
        assert store.get_risk("u1") == 0.0


class TestAlertEngine:
    """Tests for the composite severity and alert generation."""

    def _make_attack_session(self, **kwargs) -> ScoredSession:
        defaults = {
            "session_id": "sess_001",
            "user_id": "abc123",
            "host_id": "host456",
            "anomaly_score": 8.0,
            "classification": "attack",
            "classification_confidence": 0.85,
            "technique_id": "T1078",
        }
        defaults.update(kwargs)
        return ScoredSession(**defaults)

    def _make_normal_session(self, **kwargs) -> ScoredSession:
        defaults = {
            "session_id": "sess_002",
            "user_id": "normal_user",
            "host_id": "host789",
            "anomaly_score": 1.0,
            "classification": "normal",
            "classification_confidence": 0.95,
            "technique_id": None,
        }
        defaults.update(kwargs)
        return ScoredSession(**defaults)

    def test_normal_high_confidence_no_alert(self) -> None:
        engine = AlertEngine()
        session = self._make_normal_session()
        alert = engine.process_session(session)
        assert alert is None

    def test_attack_high_confidence_generates_alert(self) -> None:
        engine = AlertEngine()
        session = self._make_attack_session()
        alert = engine.process_session(session)
        assert alert is not None
        assert alert.classification == "attack"
        assert alert.alert_class in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def test_alert_deduplication(self) -> None:
        engine = AlertEngine(dedup_window_secs=10.0)
        session = self._make_attack_session()

        alert1 = engine.process_session(session)
        assert alert1 is not None

        alert2 = engine.process_session(session)
        assert alert2 is None

    def test_alert_dedup_window_expiry(self) -> None:
        engine = AlertEngine(dedup_window_secs=0.1)
        session = self._make_attack_session()

        alert1 = engine.process_session(session)
        assert alert1 is not None

        time.sleep(0.2)

        alert2 = engine.process_session(session)
        assert alert2 is not None

    def test_composite_severity_calculation(self) -> None:
        engine = AlertEngine()
        session = self._make_attack_session(
            anomaly_score=12.0,
            classification_confidence=0.9,
        )
        alert = engine.process_session(session)
        assert 0.0 <= alert.composite_severity <= 1.0

    def test_process_batch(self) -> None:
        engine = AlertEngine()
        sessions = [
            self._make_attack_session(session_id="s1"),
            self._make_normal_session(session_id="s2"),
            self._make_attack_session(session_id="s3"),
        ]
        alerts = engine.process_batch(sessions)
        assert len(alerts) >= 1
