"""Tests for Phase 3 MITRE completion infrastructure."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_mitre_classifier import main as evaluate_mitre_main
from scripts.evaluate_mitre_classifier import summarize_metrics
from src.inference.alert_store import ElasticsearchAlertStore, InMemoryAlertStore
from src.inference.phase3_detection import Phase3DetectionService, resolve_bundle_paths
from src.inference.alert_engine import Alert
from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig
from src.parsing.log_tokenizer import TOKENIZED_CHUNK_FORMAT, TOKENIZED_CHUNK_MANIFEST_FORMAT
from src.training.mitre_dataset import MITREClassificationDataset


VOCAB = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[MASK]": 2,
    "[PAD]": 3,
    "[UNK]": 4,
    "NA_NTLM_Network": 5,
}


def _write_tokenized_manifest(tmp_path: Path) -> Path:
    tokenized = tmp_path / "tokenized"
    chunk_dir = tokenized / "sessions_train_chunks"
    chunk_dir.mkdir(parents=True)
    torch.save(
        {
            "format": TOKENIZED_CHUNK_FORMAT,
            "session_ids": ["s_normal", "s_t1078", "s_t1021"],
            "input_ids": torch.tensor(
                [
                    [0, 5, 1, 3],
                    [0, 5, 5, 1],
                    [0, 5, 5, 1],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                dtype=torch.bool,
            ),
        },
        chunk_dir / "chunk_00000.pt",
    )
    manifest = tokenized / "sessions_train.pt"
    torch.save(
        {
            "format": TOKENIZED_CHUNK_MANIFEST_FORMAT,
            "chunks": ["sessions_train_chunks/chunk_00000.pt"],
            "chunk_count": 1,
            "session_count": 3,
            "max_len": 4,
        },
        manifest,
    )
    return manifest


def _write_labels(tmp_path: Path) -> Path:
    labels = tmp_path / "labels.jsonl"
    rows = [
        {
            "session_id": "s_normal",
            "user_id": "u0",
            "host_id": "h0",
            "technique_id": "normal",
            "split": "test",
        },
        {
            "session_id": "s_t1078",
            "user_id": "u1",
            "host_id": "h1",
            "technique_id": "T1078",
            "split": "test",
        },
        {
            "session_id": "s_t1021",
            "user_id": "u2",
            "host_id": "h2",
            "technique_id": "T1021",
            "split": "test",
        },
    ]
    labels.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return labels


def _write_multiclass_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = ArgusBertConfig(
        vocab_size=len(VOCAB),
        max_seq_len=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = ARGUSClassifier(config=config, num_classes=3, freeze_layers=0)
    torch.save(
        {
            "config": config,
            "num_classes": 3,
            "class_names": ["normal", "T1078", "T1021"],
            "label_to_id": {"normal": 0, "T1078": 1, "T1021": 2},
            "model_state_dict": model.state_dict(),
        },
        bundle / "best_classifier.pt",
    )
    (bundle / "calibrated_thresholds.json").write_text(
        json.dumps({"classifier_attack_threshold": 0.0}),
        encoding="utf-8",
    )
    (bundle / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    (bundle / "model_metadata.json").write_text(
        json.dumps(
            {
                "max_seq_len": 4,
                "model_task": "mitre_multiclass_classification",
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_mitre_dataset_joins_labels_to_tokenized_sessions(tmp_path: Path) -> None:
    manifest = _write_tokenized_manifest(tmp_path)
    labels = _write_labels(tmp_path)

    dataset = MITREClassificationDataset(
        manifest,
        labels,
        split="test",
        class_names=["normal", "T1078", "T1021"],
    )

    assert len(dataset) == 3
    assert dataset.class_names == ["normal", "T1078", "T1021"]
    assert dataset[1]["session_id"] == "s_t1078"
    assert dataset[1]["label"] == 1


def test_summarize_metrics_computes_macro_f1() -> None:
    report = summarize_metrics(
        labels=[0, 1, 2],
        predictions=[0, 1, 1],
        probabilities=[
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
        ],
        class_names=["normal", "T1078", "T1021"],
    )

    assert report["accuracy"] == pytest.approx(0.666667)
    assert report["macro_f1"] == pytest.approx(0.555556)
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_evaluate_mitre_classifier_writes_report(tmp_path: Path) -> None:
    manifest = _write_tokenized_manifest(tmp_path)
    labels = _write_labels(tmp_path)
    bundle = _write_multiclass_bundle(tmp_path)
    output = tmp_path / "eval"

    report = evaluate_mitre_main(
        Namespace(
            classifier=str(bundle / "best_classifier.pt"),
            manifest=[str(manifest)],
            labels=str(labels),
            split="test",
            out=str(output),
            batch_size=2,
            limit_chunks=None,
            device="cpu",
            target_macro_f1=0.75,
        )
    )

    assert "macro_f1" in report
    assert (output / "evaluation_report.json").exists()
    assert (output / "per_class_metrics.csv").exists()
    assert (output / "confusion_matrix.csv").exists()
    assert (output / "mitre_predictions.csv").exists()


def test_phase3_detection_emits_mitre_fields_and_persists_alert(tmp_path: Path) -> None:
    bundle = _write_multiclass_bundle(tmp_path)
    alert_store = InMemoryAlertStore()
    service = Phase3DetectionService(
        resolve_bundle_paths(bundle_dir=bundle),
        threshold=0.0,
        device="cpu",
        alert_store=alert_store,
        dedup_window_secs=0.0,
    )

    rows = service.score_sessions(
        [
            {
                "session_id": "s1",
                "user_id": "u1",
                "host_id": "h1",
                "replay_run_id": "run-1",
                "events": [{"event_id": "NA", "auth_type": "NTLM", "logon_type": "Network"}],
            }
        ]
    )

    assert len(rows) == 1
    assert "technique_id" in rows[0]
    assert "technique_probability" in rows[0]
    assert rows[0]["replay_run_id"] == "run-1"
    assert len(alert_store.alerts) == int(rows[0]["alert_generated"])
    if alert_store.alerts:
        assert alert_store.alerts[0]["replay_run_id"] == "run-1"


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indexed = []
        self.search_calls = []

    def index(self, *, index: str, id: str, document: dict) -> dict:
        self.indexed.append({"index": index, "id": id, "document": document})
        return {"_id": id}

    def search(self, *, index: str, query: dict, sort: list, size: int) -> dict:
        self.search_calls.append(
            {"index": index, "query": query, "sort": sort, "size": size}
        )
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "alert_1",
                        "_source": {
                            "alert_id": "alert_1",
                            "user_id": "u1",
                            "replay_run_id": "run-1",
                        },
                    }
                ]
            }
        }


class FakeApiResponse:
    def __init__(self, body: dict) -> None:
        self.body = body


class FakeElasticsearchObjectResponse(FakeElasticsearch):
    def search(self, *, index: str, query: dict, sort: list, size: int) -> FakeApiResponse:
        self.search_calls.append(
            {"index": index, "query": query, "sort": sort, "size": size}
        )
        return FakeApiResponse(
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "alert_obj",
                            "_source": {
                                "user_id": "u2",
                                "replay_run_id": "run-2",
                            },
                        }
                    ]
                }
            }
        )


def test_elasticsearch_alert_store_uses_daily_argus_alert_index() -> None:
    client = FakeElasticsearch()
    store = ElasticsearchAlertStore(client=client)
    alert = Alert(
        alert_id="alert_1",
        timestamp=1_700_000_000.0,
        user_id="u1",
        host_id="h1",
        session_id="s1",
        anomaly_score=1.0,
        classification="attack",
        classification_confidence=0.9,
        technique_id="T1078",
        technique_severity=0.8,
        user_risk=0.3,
        composite_severity=0.7,
        alert_class="HIGH",
    )

    indexed_id = store.index_alert(alert, extra={"attack_probability": 0.9})

    assert indexed_id == "alert_1"
    assert client.indexed[0]["index"].startswith("argus-alerts-")
    assert client.indexed[0]["document"]["technique_id"] == "T1078"
    assert client.indexed[0]["document"]["attack_probability"] == 0.9


def test_elasticsearch_alert_store_searches_with_filters() -> None:
    client = FakeElasticsearch()
    store = ElasticsearchAlertStore(client=client)

    rows = store.search_alerts(
        user_id="u1",
        replay_run_id="run-1",
        min_severity=0.6,
        limit=25,
    )

    assert rows == [{"alert_id": "alert_1", "user_id": "u1", "replay_run_id": "run-1"}]
    assert client.search_calls[0]["index"] == "argus-alerts-*"
    assert client.search_calls[0]["size"] == 25
    filters = client.search_calls[0]["query"]["bool"]["filter"]
    assert {
        "bool": {
            "should": [
                {"term": {"user_id": "u1"}},
                {"term": {"user_id.keyword": "u1"}},
            ],
            "minimum_should_match": 1,
        }
    } in filters
    assert {
        "bool": {
            "should": [
                {"term": {"replay_run_id": "run-1"}},
                {"term": {"replay_run_id.keyword": "run-1"}},
            ],
            "minimum_should_match": 1,
        }
    } in filters


def test_elasticsearch_alert_store_extracts_object_api_response() -> None:
    store = ElasticsearchAlertStore(client=FakeElasticsearchObjectResponse())

    rows = store.search_alerts(replay_run_id="run-2")

    assert rows == [
        {"alert_id": "alert_obj", "user_id": "u2", "replay_run_id": "run-2"}
    ]
