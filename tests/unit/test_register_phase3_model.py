"""Tests for Phase 3 MLflow bundle registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.register_phase3_model import (
    collect_metrics,
    collect_params,
    register_phase3_bundle,
    resolve_bundle_files,
)
from tests.unit.test_phase3_bundle_detection import _package_bundle


class FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self) -> "FakeRun":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self.created_registered_models = []
        self.created_model_versions = []
        self.tracking = SimpleNamespace(MlflowClient=self._client_cls)

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, run_name: str) -> FakeRun:
        self.run_name = run_name
        return FakeRun("fake-run-id")

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)

    def log_artifacts(self, local_dir: str, artifact_path: str) -> None:
        self.artifacts.append((local_dir, artifact_path))

    def get_artifact_uri(self, artifact_path: str) -> str:
        return f"fake-artifacts:/fake-run-id/{artifact_path}"

    def _client_cls(self, tracking_uri: str) -> "FakeMlflow":
        self.client_tracking_uri = tracking_uri
        return self

    def create_registered_model(self, name: str) -> SimpleNamespace:
        self.created_registered_models.append(name)
        return SimpleNamespace(name=name)

    def create_model_version(
        self,
        *,
        name: str,
        source: str,
        run_id: str,
    ) -> SimpleNamespace:
        self.created_model_versions.append((name, source, run_id))
        return SimpleNamespace(version="7")


def test_resolve_bundle_files_requires_phase3_bundle(tmp_path: Path) -> None:
    bundle_dir = _package_bundle(tmp_path)

    files = resolve_bundle_files(bundle_dir)

    assert files["classifier"].name == "best_classifier.pt"
    assert files["thresholds"].name == "calibrated_thresholds.json"
    assert files["metadata"].name == "model_metadata.json"


def test_resolve_bundle_files_rejects_missing_required_file(tmp_path: Path) -> None:
    bundle_dir = _package_bundle(tmp_path)
    (bundle_dir / "model_metadata.json").unlink()

    with pytest.raises(FileNotFoundError, match="model_metadata.json"):
        resolve_bundle_files(bundle_dir)


def test_collect_metrics_and_params_from_metadata(tmp_path: Path) -> None:
    metadata = {
        "phase": "3",
        "status": "phase3_batch_eval_complete",
        "bundle_format": "argus_phase3_model_bundle_v1",
        "vocab_size": 1233,
        "max_seq_len": 16,
        "roc_auc": 0.998,
        "pr_auc": 0.964,
        "operating_threshold": 0.999,
        "best_f1_threshold": 0.9995,
        "precision_at_operating_threshold": 0.91,
        "recall_at_operating_threshold": 0.915,
        "fpr_at_operating_threshold": 0.0019,
        "best_f1_threshold_metrics": {"f1": 0.927},
        "operating_threshold_metrics": {"f1": 0.913},
    }

    metrics = collect_metrics(metadata)
    params = collect_params(
        metadata,
        model_name="argus-bert-finetuned",
        bundle_dir=tmp_path,
    )

    assert metrics["roc_auc"] == 0.998
    assert metrics["best_f1"] == 0.927
    assert metrics["f1_at_operating_threshold"] == 0.913
    assert params["model_name"] == "argus-bert-finetuned"
    assert params["phase3_caveat"] == "binary_classifier_not_mitre_multitechnique"


def test_register_phase3_bundle_logs_and_registers_model(tmp_path: Path) -> None:
    bundle_dir = _package_bundle(tmp_path)
    fake_mlflow = FakeMlflow()

    summary = register_phase3_bundle(
        bundle_dir=bundle_dir,
        tracking_uri="http://localhost:5000",
        experiment_name="ARGUS Phase 3",
        model_name="argus-bert-finetuned",
        run_name="test-run",
        mlflow_module=fake_mlflow,
    )

    assert fake_mlflow.tracking_uri == "http://localhost:5000"
    assert fake_mlflow.experiment_name == "ARGUS Phase 3"
    assert fake_mlflow.params["model_name"] == "argus-bert-finetuned"
    assert fake_mlflow.metrics["roc_auc"] == 0.99
    assert fake_mlflow.artifacts == [(str(bundle_dir), "argus_phase3_model_bundle")]
    assert fake_mlflow.created_registered_models == ["argus-bert-finetuned"]
    assert fake_mlflow.created_model_versions == [
        (
            "argus-bert-finetuned",
            "fake-artifacts:/fake-run-id/argus_phase3_model_bundle",
            "fake-run-id",
        )
    ]
    assert summary["model_version"] == "7"
    assert summary["registered"] is True


def test_register_phase3_bundle_can_skip_model_registry(tmp_path: Path) -> None:
    bundle_dir = _package_bundle(tmp_path)
    fake_mlflow = FakeMlflow()

    summary = register_phase3_bundle(
        bundle_dir=bundle_dir,
        tracking_uri="http://localhost:5000",
        experiment_name="ARGUS Phase 3",
        model_name="argus-bert-finetuned",
        run_name="test-run",
        skip_register=True,
        mlflow_module=fake_mlflow,
    )

    assert fake_mlflow.created_model_versions == []
    assert summary["registered"] is False
    assert summary["model_version"] is None
