"""Tests for Phase 3 bundle packaging and runtime detection."""

from __future__ import annotations

from argparse import Namespace
import csv
import json
from pathlib import Path

import pytest
import torch

from scripts.package_phase3_model_bundle import main as package_bundle_main
from scripts.run_detection import main as run_detection_main
from src.api.main import create_app
from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig


VOCAB = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[MASK]": 2,
    "[PAD]": 3,
    "[UNK]": 4,
    "NA_NTLM_Network": 5,
    "NA_Kerberos_Interactive": 6,
}


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_classifier_checkpoint(path: Path) -> Path:
    config = ArgusBertConfig(
        vocab_size=len(VOCAB),
        max_seq_len=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = ARGUSClassifier(config=config, num_classes=2, freeze_layers=0)
    torch.save(
        {
            "config": config,
            "num_classes": 2,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _make_source_artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_classifier_checkpoint(artifacts / "best_classifier.pt")
    _write_json(
        artifacts / "calibrated_thresholds.json",
        {
            "classifier_attack_threshold": 0.7,
            "roc_auc": 0.99,
            "pr_auc": 0.95,
            "operating_threshold_metrics": {
                "threshold": 0.7,
                "precision": 0.9,
                "recall": 0.91,
                "fpr": 0.01,
            },
            "best_f1_threshold_metrics": {"threshold": 0.8, "f1": 0.92},
        },
    )
    _write_json(artifacts / "vocab.json", VOCAB)
    _write_json(
        artifacts / "evaluation_report.json",
        {
            "roc_auc": 0.98,
            "pr_auc": 0.94,
            "operating_threshold": {"threshold": 0.7},
            "best_f1_threshold": {"threshold": 0.8},
        },
    )
    (artifacts / "classifier_scores.csv").write_text(
        "session_id,attack_probability\ns1,0.1\n",
        encoding="utf-8",
    )
    _write_json(artifacts / "finetune_history.json", {"best_epoch": 1})
    torch.save({"model": {}}, artifacts / "checkpoint_step_003501.pt")
    return {
        "classifier": artifacts / "best_classifier.pt",
        "thresholds": artifacts / "calibrated_thresholds.json",
        "vocab": artifacts / "vocab.json",
        "evaluation_report": artifacts / "evaluation_report.json",
        "classifier_scores": artifacts / "classifier_scores.csv",
        "finetune_history": artifacts / "finetune_history.json",
        "base_checkpoint": artifacts / "checkpoint_step_003501.pt",
    }


def _package_bundle(tmp_path: Path) -> Path:
    artifacts = _make_source_artifacts(tmp_path)
    bundle_dir = tmp_path / "bundle"
    package_bundle_main(
        Namespace(
            classifier=str(artifacts["classifier"]),
            thresholds=str(artifacts["thresholds"]),
            vocab=str(artifacts["vocab"]),
            out_dir=str(bundle_dir),
            evaluation_report=str(artifacts["evaluation_report"]),
            classifier_scores=str(artifacts["classifier_scores"]),
            finetune_history=str(artifacts["finetune_history"]),
            base_checkpoint=str(artifacts["base_checkpoint"]),
            max_seq_len=8,
            status="test_bundle",
            no_archive=True,
        )
    )
    return bundle_dir


def test_package_phase3_bundle_writes_metadata_and_artifacts(tmp_path: Path) -> None:
    bundle_dir = _package_bundle(tmp_path)

    expected = {
        "best_classifier.pt",
        "calibrated_thresholds.json",
        "vocab.json",
        "evaluation_report.json",
        "classifier_scores.csv",
        "finetune_history.json",
        "checkpoint_step_003501.pt",
        "model_metadata.json",
    }
    assert expected.issubset({path.name for path in bundle_dir.iterdir()})

    metadata = json.loads((bundle_dir / "model_metadata.json").read_text())
    assert metadata["bundle_format"] == "argus_phase3_model_bundle_v1"
    assert metadata["operating_threshold"] == 0.7
    assert metadata["best_f1_threshold"] == 0.8
    assert metadata["vocab_size"] == len(VOCAB)
    assert metadata["max_seq_len"] == 8
    assert metadata["files"]["metadata"] == "model_metadata.json"


def test_run_detection_cli_scores_jsonl_from_bundle(tmp_path: Path) -> None:
    bundle_dir = _package_bundle(tmp_path)
    sessions_jsonl = tmp_path / "sessions.jsonl"
    sessions_jsonl.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "user_id": "u1",
                "host_id": "h1",
                "events": [
                    {"event_id": "NA", "auth_type": "NTLM", "logon_type": "Network"},
                    {
                        "event_id": "NA",
                        "auth_type": "Kerberos",
                        "logon_type": "Interactive",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "detections.csv"

    run_detection_main(
        Namespace(
            bundle_dir=str(bundle_dir),
            classifier=None,
            thresholds=None,
            vocab=None,
            metadata=None,
            sessions_jsonl=str(sessions_jsonl),
            manifest=None,
            out=str(output),
            threshold=0.0,
            batch_size=4,
            num_workers=0,
            device="cpu",
            max_seq_len=None,
            technique_id="T1078",
            anomaly_ceiling=15.0,
            dedup_window_secs=0.0,
        )
    )

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["prediction"] == "attack"
    assert rows[0]["threshold"] == "0.0"
    assert 0.0 <= float(rows[0]["attack_probability"]) <= 1.0


def test_phase3_api_detects_sessions_from_configured_bundle(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi.testclient")
    bundle_dir = _package_bundle(tmp_path)
    app = create_app(bundle_dir=str(bundle_dir))
    client = fastapi.TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["phase3_model_loaded"] is True

    response = client.post(
        "/phase3/detect",
        json={
            "threshold": 0.0,
            "sessions": [
                {
                    "session_id": "api_s1",
                    "user_id": "u1",
                    "host_id": "h1",
                    "events": [
                        {
                            "event_id": "NA",
                            "auth_type": "NTLM",
                            "logon_type": "Network",
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["threshold"] == 0.0
    assert payload["detections"][0]["session_id"] == "api_s1"
    assert payload["detections"][0]["prediction"] == "attack"


def test_phase3_api_exposes_and_clears_ueba_risks(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi.testclient")
    bundle_dir = _package_bundle(tmp_path)
    app = create_app(bundle_dir=str(bundle_dir))
    client = fastapi.TestClient(app)

    detection = client.post(
        "/phase3/detect",
        json={
            "threshold": 0.0,
            "sessions": [
                {
                    "session_id": "risk_s1",
                    "user_id": "risk_user",
                    "host_id": "risk_host",
                    "events": [
                        {
                            "event_id": "NA",
                            "auth_type": "NTLM",
                            "logon_type": "Network",
                        }
                    ],
                }
            ],
        },
    )
    assert detection.status_code == 200

    all_risks = client.get("/phase3/ueba/risks")
    assert all_risks.status_code == 200
    all_payload = all_risks.json()
    assert all_payload["count"] == 1
    assert "risk_user" in all_payload["risks"]
    assert all_payload["risks"]["risk_user"] > 0.0

    user_risk = client.get("/phase3/ueba/risks/risk_user")
    assert user_risk.status_code == 200
    user_payload = user_risk.json()
    assert user_payload["exists"] is True
    assert user_payload["risk"] == pytest.approx(all_payload["risks"]["risk_user"])

    timeline = client.get("/phase3/ueba/risks/risk_user/timeline")
    assert timeline.status_code == 200
    timeline_payload = timeline.json()
    assert timeline_payload["count"] >= 1
    assert timeline_payload["timeline"][0]["risk"] > 0.0

    missing_risk = client.get("/phase3/ueba/risks/missing_user")
    assert missing_risk.status_code == 200
    assert missing_risk.json()["exists"] is False

    cleared = client.delete("/phase3/ueba/risks")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] == 1
    assert cleared.json()["remaining"] == 0

    after_clear = client.get("/phase3/ueba/risks")
    assert after_clear.json()["risks"] == {}
