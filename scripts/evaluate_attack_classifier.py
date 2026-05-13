"""Evaluate ARGUS attack classifier thresholds and alert integration.

Loads the fine-tuned classifier, scores attack plus normal-control sessions,
writes per-session probabilities, computes ROC-AUC / PR-AUC, sweeps classifier
probability thresholds, selects an operating threshold, and feeds real classifier
outputs into the alert engine using that threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.inference.alert_engine import AlertEngine, ScoredSession
from src.models.attack_classifier import ARGUSClassifier


DEFAULT_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
]


class EvalDataset(Dataset):
    """Load tokenized sessions with binary labels for evaluation."""

    def __init__(self) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.attention_masks: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.session_indices: list[int] = []

    def add_from_manifest(
        self,
        manifest_path: str,
        label: int,
        max_sessions: Optional[int] = None,
        limit_chunks: Optional[int] = None,
    ) -> int:
        """Load sessions from a tokenized manifest or legacy flat list."""
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        count = 0
        index_offset = len(self.labels)

        if isinstance(manifest, dict) and "chunks" in manifest:
            base_dir = Path(manifest_path).parent
            chunk_paths = manifest["chunks"]
            chunks_to_load = chunk_paths[:limit_chunks] if limit_chunks else chunk_paths

            for chunk_ref in chunks_to_load:
                chunk = torch.load(base_dir / chunk_ref, map_location="cpu", weights_only=False)
                ids_tensor = chunk["input_ids"]
                mask_tensor = chunk["attention_mask"]
                for row_index in range(ids_tensor.shape[0]):
                    if max_sessions is not None and count >= max_sessions:
                        return count
                    self.input_ids.append(ids_tensor[row_index])
                    self.attention_masks.append(mask_tensor[row_index])
                    self.labels.append(label)
                    self.session_indices.append(index_offset + count)
                    count += 1
            return count

        for item in manifest:
            if max_sessions is not None and count >= max_sessions:
                break
            self.input_ids.append(item["input_ids"])
            self.attention_masks.append(item["attention_mask"])
            self.labels.append(label)
            self.session_indices.append(index_offset + count)
            count += 1
        return count

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
            "index": self.session_indices[idx],
        }


def collate_eval(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


def parse_thresholds(raw_thresholds: str | None) -> list[float]:
    """Parse a comma-separated threshold list or return the default sweep."""
    if raw_thresholds is None:
        return DEFAULT_THRESHOLDS

    thresholds = []
    for raw_value in raw_thresholds.split(","):
        value = raw_value.strip()
        if not value:
            continue
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1]: {threshold}")
        thresholds.append(threshold)

    if not thresholds:
        raise ValueError("--thresholds did not contain any threshold values")
    return sorted(set(thresholds))


@torch.no_grad()
def run_inference(
    model: ARGUSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run classifier on all sessions."""
    model.eval()
    all_probs = []
    all_labels = []
    all_indices = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_labels.extend(batch["labels"].tolist())
        all_indices.extend(batch["indices"].tolist())

    probs_array = np.concatenate(all_probs, axis=0)
    return {
        "attack_prob": probs_array[:, 1],
        "normal_prob": probs_array[:, 0],
        "labels": np.array(all_labels),
        "indices": np.array(all_indices),
    }


def compute_binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute binary classification metrics."""
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    total = len(labels)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(total, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
        "accuracy": round(accuracy, 6),
    }


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC using trapezoidal integration."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    tp = 0
    fp = 0
    tpr_values = [0.0]
    fpr_values = [0.0]
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)

    auc = 0.0
    for index in range(1, len(tpr_values)):
        auc += (
            (fpr_values[index] - fpr_values[index - 1])
            * (tpr_values[index] + tpr_values[index - 1])
            / 2
        )
    return round(float(auc), 6)


def compute_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Precision-Recall AUC using trapezoidal integration."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    positives = labels.sum()
    if positives == 0:
        return 0.0

    tp = 0
    precision_values = []
    recall_values = []
    for index, label in enumerate(sorted_labels, start=1):
        if label == 1:
            tp += 1
        precision_values.append(tp / index)
        recall_values.append(tp / positives)

    auc = 0.0
    for index in range(1, len(precision_values)):
        auc += (
            (recall_values[index] - recall_values[index - 1])
            * (precision_values[index] + precision_values[index - 1])
            / 2
        )
    return round(float(auc), 6)


def threshold_sweep(
    labels: np.ndarray,
    attack_probs: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Compute metrics for every threshold."""
    rows = []
    for threshold in thresholds:
        predictions = (attack_probs >= threshold).astype(int)
        metrics = compute_binary_metrics(labels, predictions)
        metrics["threshold"] = threshold
        rows.append(metrics)
    return rows


def select_operating_threshold(
    sweep: list[dict],
    *,
    min_recall: float,
    max_fpr: float,
    explicit_threshold: float | None,
) -> dict:
    """Choose the threshold used for alert evaluation and deployment config."""
    if explicit_threshold is not None:
        return min(sweep, key=lambda row: abs(row["threshold"] - explicit_threshold))

    eligible = [
        row for row in sweep
        if row["recall"] >= min_recall and row["fpr"] <= max_fpr
    ]
    candidates = eligible or sweep

    return max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["precision"],
            -row["fpr"],
            row["threshold"],
        ),
    )


def write_scores_csv(
    path: Path,
    attack_probs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    operating_threshold: float,
) -> None:
    """Write per-session classification results to CSV."""
    with path.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "session_index",
            "true_label",
            "attack_probability",
            "prediction_05",
            "prediction_operating",
            "operating_threshold",
            "label_name",
        ])
        for index in range(len(labels)):
            prediction_05 = int(attack_probs[index] >= 0.5)
            prediction_operating = int(attack_probs[index] >= operating_threshold)
            label_name = "attack" if labels[index] == 1 else "normal"
            writer.writerow([
                int(indices[index]),
                int(labels[index]),
                round(float(attack_probs[index]), 6),
                prediction_05,
                prediction_operating,
                operating_threshold,
                label_name,
            ])


def run_alert_engine(
    attack_probs: np.ndarray,
    labels: np.ndarray,
    *,
    classification_threshold: float,
    anomaly_ceiling: float = 15.0,
) -> dict:
    """Feed classifier outputs into the alert engine at the chosen threshold."""
    engine = AlertEngine(dedup_window_secs=0, anomaly_ceiling=anomaly_ceiling)

    sessions = []
    for index in range(len(labels)):
        prob = float(attack_probs[index])
        is_attack_prediction = prob >= classification_threshold
        sessions.append(ScoredSession(
            session_id=f"eval_{index:06d}",
            user_id=f"user_{index % 100:04d}",
            host_id=f"host_{index % 20:03d}",
            anomaly_score=prob * anomaly_ceiling,
            classification="attack" if is_attack_prediction else "normal",
            classification_confidence=prob if is_attack_prediction else 0.95,
            technique_id=None,
        ))

    alerts = engine.process_batch(sessions)
    alert_labels = []
    for alert in alerts:
        index = int(alert.session_id.split("_")[1])
        alert_labels.append(labels[index])

    true_attack_alerts = int(sum(alert_labels))
    false_alerts = int(len(alert_labels) - true_attack_alerts)
    return {
        "classification_threshold": classification_threshold,
        "total_alerts": len(alerts),
        "true_attack_alerts": true_attack_alerts,
        "false_alerts": false_alerts,
        "alert_precision": round(true_attack_alerts / max(len(alerts), 1), 4),
        "severity_distribution": {
            level: sum(1 for alert in alerts if alert.alert_class == level)
            for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
        "engine_summary": engine.get_stats(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS attack classifier")
    parser.add_argument("--classifier", required=True, help="Path to best_classifier.pt")
    parser.add_argument("--attack-manifest", required=True, help="Attack sessions manifest")
    parser.add_argument("--normal-manifest", required=True, help="Normal sessions manifest")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--max-normal", type=int, default=10_000)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--thresholds",
        help="Comma-separated thresholds. Defaults include 0.90 through 0.9999.",
    )
    parser.add_argument(
        "--operating-threshold",
        type=float,
        help="Explicit threshold for alert-engine evaluation.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.90,
        help="Minimum recall target for automatic operating threshold selection.",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="Maximum FPR target for automatic operating threshold selection.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.operating_threshold is not None and not 0.0 <= args.operating_threshold <= 1.0:
        raise ValueError("--operating-threshold must be in [0, 1]")
    if not 0.0 <= args.min_recall <= 1.0:
        raise ValueError("--min-recall must be in [0, 1]")
    if not 0.0 <= args.max_fpr <= 1.0:
        raise ValueError("--max-fpr must be in [0, 1]")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    print("\n-- Loading classifier --")
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    model = ARGUSClassifier(
        num_classes=checkpoint.get("num_classes", 2),
        freeze_layers=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
        f"train F1={checkpoint.get('best_f1', 0):.4f}"
    )

    print("\n-- Loading evaluation data --")
    dataset = EvalDataset()
    n_attack = dataset.add_from_manifest(args.attack_manifest, label=1)
    n_normal = dataset.add_from_manifest(
        args.normal_manifest,
        label=0,
        max_sessions=args.max_normal,
        limit_chunks=args.limit_chunks,
    )
    print(f"  Attack: {n_attack:,} | Normal: {n_normal:,} | Total: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("\n-- Running inference --")
    results = run_inference(model, loader, device)
    attack_probs = results["attack_prob"]
    labels = results["labels"]
    indices = results["indices"]
    print(f"  Scored {len(labels):,} sessions")

    attack_scores = attack_probs[labels == 1]
    normal_scores = attack_probs[labels == 0]
    print(
        f"\n  Attack P(attack): mean={attack_scores.mean():.4f} "
        f"std={attack_scores.std():.4f} min={attack_scores.min():.4f} "
        f"max={attack_scores.max():.4f}"
    )
    print(
        f"  Normal P(attack): mean={normal_scores.mean():.4f} "
        f"std={normal_scores.std():.4f} min={normal_scores.min():.4f} "
        f"max={normal_scores.max():.4f}"
    )

    roc_auc = compute_roc_auc(labels, attack_probs)
    pr_auc = compute_pr_auc(labels, attack_probs)
    print("\n-- Global Metrics --")
    print(f"  ROC-AUC: {roc_auc:.6f}")
    print(f"  PR-AUC:  {pr_auc:.6f}")

    print("\n-- Threshold Sweep --")
    thresholds = parse_thresholds(args.thresholds)
    sweep = threshold_sweep(labels, attack_probs, thresholds=thresholds)
    header = (
        f"{'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FPR':>8} "
        f"{'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    )
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for row in sweep:
        print(
            f"  {row['threshold']:>8.4f} {row['precision']:>8.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['fpr']:>8.4f} "
            f"{row['tp']:>6} {row['fp']:>6} {row['fn']:>6} {row['tn']:>6}"
        )

    best_f1_row = max(sweep, key=lambda row: (row["f1"], row["precision"], -row["fpr"]))
    operating_row = select_operating_threshold(
        sweep,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
        explicit_threshold=args.operating_threshold,
    )
    operating_threshold = float(operating_row["threshold"])
    print(
        f"\n  Best threshold by F1: {best_f1_row['threshold']:.4f} -> "
        f"F1={best_f1_row['f1']:.4f} Prec={best_f1_row['precision']:.4f} "
        f"Rec={best_f1_row['recall']:.4f} FPR={best_f1_row['fpr']:.4f}"
    )
    print(
        f"  Operating threshold: {operating_threshold:.4f} -> "
        f"F1={operating_row['f1']:.4f} Prec={operating_row['precision']:.4f} "
        f"Rec={operating_row['recall']:.4f} FPR={operating_row['fpr']:.4f}"
    )
    print(f"  Selection target: recall>={args.min_recall:.2f}, FPR<={args.max_fpr:.2f}")

    csv_path = output_dir / "classifier_scores.csv"
    write_scores_csv(csv_path, attack_probs, labels, indices, operating_threshold)
    print(f"\n  Scores CSV -> {csv_path}")

    print("\n-- Alert Engine (operating threshold) --")
    alert_results = run_alert_engine(
        attack_probs,
        labels,
        classification_threshold=operating_threshold,
    )
    print(f"  Classification threshold: {alert_results['classification_threshold']:.4f}")
    print(f"  Total alerts:        {alert_results['total_alerts']}")
    print(f"  True attack alerts:  {alert_results['true_attack_alerts']}")
    print(f"  False alerts:        {alert_results['false_alerts']}")
    print(f"  Alert precision:     {alert_results['alert_precision']:.4f}")
    print(f"  Severity dist:       {alert_results['severity_distribution']}")

    report = {
        "classifier_checkpoint": args.classifier,
        "n_attack": n_attack,
        "n_normal": n_normal,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "attack_prob_stats": {
            "mean": round(float(attack_scores.mean()), 6),
            "std": round(float(attack_scores.std()), 6),
            "min": round(float(attack_scores.min()), 6),
            "max": round(float(attack_scores.max()), 6),
        },
        "normal_prob_stats": {
            "mean": round(float(normal_scores.mean()), 6),
            "std": round(float(normal_scores.std()), 6),
            "min": round(float(normal_scores.min()), 6),
            "max": round(float(normal_scores.max()), 6),
        },
        "best_f1_threshold": best_f1_row,
        "operating_threshold": operating_row,
        "threshold_sweep": sweep,
        "alert_engine_results": alert_results,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Full report -> {report_path}")

    threshold_config = {
        "classifier_attack_threshold": operating_threshold,
        "selection_policy": {
            "min_recall": args.min_recall,
            "max_fpr": args.max_fpr,
            "explicit_threshold": args.operating_threshold,
        },
        "operating_threshold_metrics": operating_row,
        "best_f1_threshold": best_f1_row["threshold"],
        "best_f1_threshold_metrics": best_f1_row,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    threshold_path = output_dir / "calibrated_thresholds.json"
    threshold_path.write_text(json.dumps(threshold_config, indent=2))
    print(f"  Thresholds -> {threshold_path}")

    print("\nPhase 3 evaluation complete.")


if __name__ == "__main__":
    main()
