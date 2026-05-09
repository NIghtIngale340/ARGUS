"""Evaluate ARGUS multi-class MITRE ATT&CK classifier."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig
from src.training.mitre_dataset import MITREClassificationDataset, collate_mitre_batch


def compute_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def compute_per_class_metrics(matrix: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for class_id, class_name in enumerate(class_names):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        support = int(matrix[class_id, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def summarize_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
) -> dict[str, Any]:
    matrix = compute_confusion_matrix(labels, predictions, len(class_names))
    per_class = compute_per_class_metrics(matrix, class_names)
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0
    macro_precision = (
        float(np.mean([row["precision"] for row in per_class])) if per_class else 0.0
    )
    macro_recall = float(np.mean([row["recall"] for row in per_class])) if per_class else 0.0
    accuracy = (
        sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
        / max(len(labels), 1)
    )
    confidences = [
        float(max(row)) if row else 0.0
        for row in probabilities
    ]

    return {
        "sample_count": len(labels),
        "class_names": class_names,
        "accuracy": round(float(accuracy), 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "mean_confidence": round(float(np.mean(confidences)), 6) if confidences else 0.0,
        "per_class_metrics": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def load_classifier(checkpoint_path: Path, device: torch.device) -> tuple[ARGUSClassifier, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, ArgusBertConfig):
        config = ArgusBertConfig()
    class_names = checkpoint.get("class_names")
    num_classes = int(checkpoint.get("num_classes", len(class_names) if class_names else 2))
    if not class_names:
        class_names = ["normal"] + [f"class_{index}" for index in range(1, num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("checkpoint class_names length does not match num_classes")

    model = ARGUSClassifier(config=config, num_classes=num_classes, freeze_layers=0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, list(class_names)


@torch.no_grad()
def run_inference(
    model: ARGUSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    session_ids: list[str] = []
    user_ids: list[str] = []
    host_ids: list[str] = []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        preds = probs.argmax(dim=-1)
        probabilities.extend(probs.tolist())
        predictions.extend(preds.tolist())
        labels.extend(batch["labels"].tolist())
        session_ids.extend(batch["session_id"])
        user_ids.extend(batch["user_id"])
        host_ids.extend(batch["host_id"])

    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "session_ids": session_ids,
        "user_ids": user_ids,
        "host_ids": host_ids,
    }


def write_per_class_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["class_id", "class_name", "precision", "recall", "f1", "support", "tp", "fp", "fn"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def write_predictions(path: Path, results: dict[str, Any], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = [
            "session_id",
            "user_id",
            "host_id",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "confidence",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for index, probs in enumerate(results["probabilities"]):
            true_label = int(results["labels"][index])
            predicted_label = int(results["predictions"][index])
            writer.writerow(
                {
                    "session_id": results["session_ids"][index],
                    "user_id": results["user_ids"][index],
                    "host_id": results["host_ids"][index],
                    "true_label": true_label,
                    "true_class": class_names[true_label],
                    "predicted_label": predicted_label,
                    "predicted_class": class_names[predicted_label],
                    "confidence": round(float(max(probs)), 6),
                }
            )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate ARGUS MITRE multi-class classifier")
    parser.add_argument("--classifier", required=True, help="Path to best_classifier.pt")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Tokenized manifest to evaluate. Repeat for multiple manifests.",
    )
    parser.add_argument("--labels", required=True, help="MITRE labels JSONL/CSV")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--target-macro-f1", type=float, default=0.75)
    return parser


def main(args: Namespace | None = None) -> dict[str, Any]:
    parsed = args or build_parser().parse_args()
    if parsed.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not 0.0 <= parsed.target_macro_f1 <= 1.0:
        raise ValueError("--target-macro-f1 must be in [0, 1]")

    if parsed.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(parsed.device)
    if parsed.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, class_names = load_classifier(Path(parsed.classifier), device)
    dataset = MITREClassificationDataset(
        [Path(path) for path in parsed.manifest],
        parsed.labels,
        split=parsed.split,
        class_names=class_names,
        limit_chunks=parsed.limit_chunks,
    )
    loader = DataLoader(
        dataset,
        batch_size=parsed.batch_size,
        shuffle=False,
        collate_fn=collate_mitre_batch,
    )
    results = run_inference(model, loader, device)
    report = summarize_metrics(
        results["labels"],
        results["predictions"],
        results["probabilities"],
        class_names,
    )
    report.update(
        {
            "classifier_checkpoint": parsed.classifier,
            "labels_path": parsed.labels,
            "split": parsed.split,
            "target_macro_f1": parsed.target_macro_f1,
            "passes_target": report["macro_f1"] >= parsed.target_macro_f1,
            "phase3_metric": "mitre_macro_f1",
        }
    )

    output_dir = Path(parsed.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_per_class_metrics(output_dir / "per_class_metrics.csv", report["per_class_metrics"])
    write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        report["confusion_matrix"],
        class_names,
    )
    write_predictions(output_dir / "mitre_predictions.csv", results, class_names)

    print(
        f"MITRE evaluation: macro-F1={report['macro_f1']:.6f} "
        f"target={parsed.target_macro_f1:.2f} passes={report['passes_target']}"
    )
    print(f"Report written to: {output_dir / 'evaluation_report.json'}")
    return report


if __name__ == "__main__":
    main()
