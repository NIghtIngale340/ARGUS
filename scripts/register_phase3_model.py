"""Register the ARGUS Phase 3 bundle in MLflow.

Example:
    python -m scripts.register_phase3_model \
        --bundle-dir data/argus_phase3_model_bundle \
        --tracking-uri http://localhost:5000 \
        --model-name argus-bert-finetuned
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


REQUIRED_BUNDLE_FILES = {
    "classifier": "best_classifier.pt",
    "thresholds": "calibrated_thresholds.json",
    "vocab": "vocab.json",
    "metadata": "model_metadata.json",
}

OPTIONAL_BUNDLE_FILES = {
    "evaluation_report": "evaluation_report.json",
    "classifier_scores": "classifier_scores.csv",
    "finetune_history": "finetune_history.json",
    "base_mlm_checkpoint": "checkpoint_step_003501.pt",
}

METRIC_KEYS = [
    "roc_auc",
    "pr_auc",
    "operating_threshold",
    "best_f1_threshold",
    "precision_at_operating_threshold",
    "recall_at_operating_threshold",
    "fpr_at_operating_threshold",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def resolve_bundle_files(bundle_dir: str | Path) -> dict[str, Path | None]:
    bundle = Path(bundle_dir)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle}")
    if not bundle.is_dir():
        raise ValueError(f"Bundle path must be a directory: {bundle}")

    resolved: dict[str, Path | None] = {}
    for label, filename in REQUIRED_BUNDLE_FILES.items():
        path = bundle / filename
        if not path.exists():
            raise FileNotFoundError(f"Required bundle file missing: {path}")
        if not path.is_file():
            raise ValueError(f"Required bundle path must be a file: {path}")
        resolved[label] = path

    for label, filename in OPTIONAL_BUNDLE_FILES.items():
        path = bundle / filename
        resolved[label] = path if path.exists() and path.is_file() else None

    return resolved


def coerce_metric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for key in METRIC_KEYS:
        value = coerce_metric(metadata.get(key))
        if value is not None:
            metrics[key] = value

    best_f1_metrics = metadata.get("best_f1_threshold_metrics")
    if isinstance(best_f1_metrics, dict):
        value = coerce_metric(best_f1_metrics.get("f1"))
        if value is not None:
            metrics["best_f1"] = value

    operating_metrics = metadata.get("operating_threshold_metrics")
    if isinstance(operating_metrics, dict):
        value = coerce_metric(operating_metrics.get("f1"))
        if value is not None:
            metrics["f1_at_operating_threshold"] = value

    return metrics


def collect_params(
    metadata: dict[str, Any],
    *,
    model_name: str,
    bundle_dir: Path,
) -> dict[str, str | int | float | bool]:
    return {
        "phase": str(metadata.get("phase", "3")),
        "status": str(metadata.get("status", "unknown")),
        "bundle_format": str(metadata.get("bundle_format", "unknown")),
        "model_name": model_name,
        "bundle_dir": str(bundle_dir),
        "classifier_task": "binary_attack_classification",
        "vocab_size": int(metadata.get("vocab_size", 0) or 0),
        "max_seq_len": int(metadata.get("max_seq_len", 0) or 0),
        "phase3_caveat": "binary_classifier_not_mitre_multitechnique",
    }


def import_mlflow() -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install project dependencies or run "
            "`python -m pip install mlflow>=2.10.0`."
        ) from exc
    return mlflow


def create_registered_model_version(
    *,
    mlflow: Any,
    tracking_uri: str,
    model_name: str,
    source_uri: str,
    run_id: str,
) -> Any:
    """Create a model version without requiring MLflow 3 logged-model APIs."""
    client_cls = getattr(getattr(mlflow, "tracking", None), "MlflowClient", None)
    if client_cls is None:
        return mlflow.register_model(model_uri=source_uri, name=model_name)

    client = client_cls(tracking_uri=tracking_uri)
    try:
        client.create_registered_model(model_name)
    except Exception:
        # Usually RESOURCE_ALREADY_EXISTS. Keep this broad so older/newer
        # MLflow clients do not turn an idempotent registration into a failure.
        pass
    return client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id,
    )


def register_phase3_bundle(
    *,
    bundle_dir: str | Path,
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    run_name: str,
    skip_register: bool = False,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    """Log the Phase 3 bundle to MLflow and optionally create a model version."""
    bundle = Path(bundle_dir)
    files = resolve_bundle_files(bundle)
    metadata = read_json(files["metadata"])
    metrics = collect_metrics(metadata)
    params = collect_params(metadata, model_name=model_name, bundle_dir=bundle)

    mlflow = mlflow_module or import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(bundle), artifact_path="argus_phase3_model_bundle")

        model_uri = mlflow.get_artifact_uri("argus_phase3_model_bundle")
        model_version = None
        if not skip_register:
            registered = create_registered_model_version(
                mlflow=mlflow,
                tracking_uri=tracking_uri,
                model_name=model_name,
                source_uri=model_uri,
                run_id=run_id,
            )
            model_version = getattr(registered, "version", None)

    return {
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "model_uri": model_uri,
        "model_version": model_version,
        "registered": not skip_register,
        "metrics": metrics,
        "params": params,
        "files": {label: str(path) if path else None for label, path in files.items()},
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Register ARGUS Phase 3 bundle in MLflow")
    parser.add_argument("--bundle-dir", required=True, help="Path to argus_phase3_model_bundle")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment-name", default="ARGUS Phase 3")
    parser.add_argument("--model-name", default="argus-bert-finetuned")
    parser.add_argument("--run-name", default="phase3-bundle-registration")
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Log artifacts and metrics without creating a registered model version",
    )
    parser.add_argument("--summary-out", help="Optional path to write registration summary JSON")
    return parser


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()
    summary = register_phase3_bundle(
        bundle_dir=parsed.bundle_dir,
        tracking_uri=parsed.tracking_uri,
        experiment_name=parsed.experiment_name,
        model_name=parsed.model_name,
        run_name=parsed.run_name,
        skip_register=parsed.skip_register,
    )

    if parsed.summary_out:
        summary_path = Path(parsed.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"MLflow run: {summary['run_id']}")
    print(f"Model URI: {summary['model_uri']}")
    if summary["registered"]:
        print(f"Registered model: {summary['model_name']} v{summary['model_version']}")
    else:
        print("Registered model: skipped")
    if parsed.summary_out:
        print(f"Summary: {parsed.summary_out}")


if __name__ == "__main__":
    main()
