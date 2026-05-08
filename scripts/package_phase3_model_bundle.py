"""Package ARGUS Phase 3 artifacts into a deployable model bundle.

Example:
    python scripts/package_phase3_model_bundle.py \
        --classifier /kaggle/working/argus_finetuned/best_classifier.pt \
        --thresholds /kaggle/working/argus_phase3_eval/calibrated_thresholds.json \
        --vocab /kaggle/input/datasets/nightingale21/argus-tokenized-58day-verified/data/vocab.json \
        --evaluation-report /kaggle/working/argus_phase3_eval/evaluation_report.json \
        --classifier-scores /kaggle/working/argus_phase3_eval/classifier_scores.csv \
        --finetune-history /kaggle/working/argus_finetuned/finetune_history.json \
        --base-checkpoint /kaggle/working/argus_mlm_eval_check/checkpoint_step_003501.pt \
        --out-dir /kaggle/working/argus_phase3_model_bundle
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def copy_artifact(src: Path | None, out_dir: Path, name: str) -> str | None:
    if src is None:
        return None
    if not src.exists():
        return None
    dst = out_dir / name
    shutil.copy2(src, dst)
    return dst.name


def build_metadata(
    *,
    copied_files: dict[str, str | None],
    thresholds: dict[str, Any],
    evaluation_report: dict[str, Any],
    vocab_path: Path,
    max_seq_len: int,
    status: str,
) -> dict[str, Any]:
    with vocab_path.open("r", encoding="utf-8") as file_obj:
        vocab = json.load(file_obj)
    if not isinstance(vocab, dict):
        raise ValueError(f"Vocabulary JSON must be an object: {vocab_path}")

    operating_metrics = (
        thresholds.get("operating_threshold_metrics")
        or evaluation_report.get("operating_threshold")
        or {}
    )
    best_f1_metrics = (
        thresholds.get("best_f1_threshold_metrics")
        or evaluation_report.get("best_f1_threshold")
        or {}
    )
    operating_threshold = (
        thresholds.get("classifier_attack_threshold")
        or operating_metrics.get("threshold")
    )

    return {
        "phase": "3",
        "status": status,
        "bundle_format": "argus_phase3_model_bundle_v1",
        "files": copied_files,
        "vocab_size": len(vocab),
        "max_seq_len": max_seq_len,
        "operating_threshold": operating_threshold,
        "best_f1_threshold": best_f1_metrics.get("threshold"),
        "roc_auc": thresholds.get("roc_auc", evaluation_report.get("roc_auc")),
        "pr_auc": thresholds.get("pr_auc", evaluation_report.get("pr_auc")),
        "precision_at_operating_threshold": operating_metrics.get("precision"),
        "recall_at_operating_threshold": operating_metrics.get("recall"),
        "fpr_at_operating_threshold": operating_metrics.get("fpr"),
        "operating_threshold_metrics": operating_metrics,
        "best_f1_threshold_metrics": best_f1_metrics,
        "selection_policy": thresholds.get("selection_policy", {}),
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Package ARGUS Phase 3 model artifacts")
    parser.add_argument("--classifier", required=True, help="Path to best_classifier.pt")
    parser.add_argument("--thresholds", required=True, help="Path to calibrated_thresholds.json")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--out-dir", required=True, help="Output bundle directory")
    parser.add_argument("--evaluation-report", help="Path to evaluation_report.json")
    parser.add_argument("--classifier-scores", help="Path to classifier_scores.csv")
    parser.add_argument("--finetune-history", help="Path to finetune_history.json")
    parser.add_argument("--base-checkpoint", help="Path to Phase 2 MLM checkpoint")
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--status", default="phase3_batch_eval_complete")
    parser.add_argument("--no-archive", action="store_true", help="Do not create a .zip archive")
    return parser


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()

    classifier = require_file(parsed.classifier, "classifier")
    thresholds_path = require_file(parsed.thresholds, "thresholds")
    vocab = require_file(parsed.vocab, "vocab")
    evaluation_report = (
        require_file(parsed.evaluation_report, "evaluation report")
        if parsed.evaluation_report
        else None
    )
    classifier_scores = (
        require_file(parsed.classifier_scores, "classifier scores")
        if parsed.classifier_scores
        else None
    )
    finetune_history = (
        require_file(parsed.finetune_history, "finetune history")
        if parsed.finetune_history
        else None
    )
    base_checkpoint = (
        require_file(parsed.base_checkpoint, "base checkpoint")
        if parsed.base_checkpoint
        else None
    )

    out_dir = Path(parsed.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_files = {
        "classifier_checkpoint": copy_artifact(classifier, out_dir, "best_classifier.pt"),
        "thresholds": copy_artifact(thresholds_path, out_dir, "calibrated_thresholds.json"),
        "vocab": copy_artifact(vocab, out_dir, "vocab.json"),
        "evaluation_report": copy_artifact(evaluation_report, out_dir, "evaluation_report.json"),
        "classifier_scores": copy_artifact(classifier_scores, out_dir, "classifier_scores.csv"),
        "finetune_history": copy_artifact(finetune_history, out_dir, "finetune_history.json"),
        "base_mlm_checkpoint": (
            copy_artifact(base_checkpoint, out_dir, base_checkpoint.name)
            if base_checkpoint
            else None
        ),
    }
    copied_files["metadata"] = "model_metadata.json"

    metadata = build_metadata(
        copied_files=copied_files,
        thresholds=read_json(thresholds_path),
        evaluation_report=read_json(evaluation_report),
        vocab_path=vocab,
        max_seq_len=parsed.max_seq_len,
        status=parsed.status,
    )
    metadata_path = out_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Bundle directory: {out_dir}")
    for label, filename in copied_files.items():
        if filename:
            print(f"  {label}: {filename}")

    if not parsed.no_archive:
        archive_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"Bundle archive: {archive_path}")


if __name__ == "__main__":
    main()
