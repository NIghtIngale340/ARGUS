"""ARGUS Phase 3 — Classifier evaluation, threshold calibration, and alert-engine integration.

Loads the fine-tuned classifier, scores attack + same-day normal control
sessions, produces a full probability CSV, and computes ROC-AUC, PR-AUC,
confusion matrices, and threshold sweep tables.

Usage (Kaggle):
    python -m scripts.evaluate_attack_classifier \
        --classifier /kaggle/working/argus_finetuned/best_classifier.pt \
        --attack-manifest /kaggle/working/attack_sessions/attack_sessions.pt \
        --normal-manifest /kaggle/input/.../sessions_val.pt \
        --out /kaggle/working/argus_phase3_eval \
        --max-normal 10000 --limit-chunks 20
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig
from src.inference.alert_engine import AlertEngine, ScoredSession


# ---------------------------------------------------------------------------
# Dataset (reuse pattern from finetune.py, but label-aware)
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """Load tokenized sessions with labels for evaluation."""

    def __init__(self) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.attention_masks: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.session_indices: list[int] = []  # original index for tracing

    def add_from_manifest(
        self,
        manifest_path: str,
        label: int,
        max_sessions: Optional[int] = None,
        limit_chunks: Optional[int] = None,
    ) -> int:
        """Load sessions from a manifest. Returns count added."""
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        count = 0
        idx_offset = len(self.labels)

        if isinstance(manifest, dict) and "chunks" in manifest:
            base_dir = Path(manifest_path).parent
            chunk_paths = manifest["chunks"]
            chunks_to_load = chunk_paths[:limit_chunks] if limit_chunks else chunk_paths

            for chunk_rel in chunks_to_load:
                chunk = torch.load(base_dir / chunk_rel, map_location="cpu", weights_only=False)
                ids_t = chunk["input_ids"]
                mask_t = chunk["attention_mask"]
                for i in range(ids_t.shape[0]):
                    if max_sessions and count >= max_sessions:
                        return count
                    self.input_ids.append(ids_t[i])
                    self.attention_masks.append(mask_t[i])
                    self.labels.append(label)
                    self.session_indices.append(idx_offset + count)
                    count += 1
        else:
            # Flat list / legacy
            for item in manifest:
                if max_sessions and count >= max_sessions:
                    break
                self.input_ids.append(item["input_ids"])
                self.attention_masks.append(item["attention_mask"])
                self.labels.append(label)
                self.session_indices.append(idx_offset + count)
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
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: ARGUSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run classifier on all sessions. Returns arrays of probabilities and labels."""
    model.eval()
    all_probs = []
    all_labels = []
    all_indices = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=-1)  # (B, 2)

        all_probs.append(probs.cpu().numpy())
        all_labels.extend(batch["labels"].tolist())
        all_indices.extend(batch["indices"].tolist())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, 2)
    return {
        "attack_prob": all_probs[:, 1],       # P(attack)
        "normal_prob": all_probs[:, 0],       # P(normal)
        "labels": np.array(all_labels),        # 0=normal, 1=attack
        "indices": np.array(all_indices),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute TP, FP, FN, TN, precision, recall, F1."""
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    n = len(labels)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(n, 1)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
        "accuracy": round(accuracy, 6),
    }


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC using the trapezoidal rule (no sklearn dependency)."""
    # Sort by descending score
    desc = np.argsort(-scores)
    sorted_labels = labels[desc]
    sorted_scores = scores[desc]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    return round(auc, 6)


def compute_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Precision-Recall AUC."""
    desc = np.argsort(-scores)
    sorted_labels = labels[desc]

    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0

    precision_list = []
    recall_list = []
    tp = 0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        precision_list.append(tp / (i + 1))
        recall_list.append(tp / n_pos)

    # Trapezoidal rule on P-R curve
    auc = 0.0
    for i in range(1, len(precision_list)):
        auc += (recall_list[i] - recall_list[i-1]) * (precision_list[i] + precision_list[i-1]) / 2
    return round(auc, 6)


def threshold_sweep(
    labels: np.ndarray,
    attack_probs: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> list[dict]:
    """Sweep probability thresholds and compute metrics at each."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    for t in thresholds:
        preds = (attack_probs >= t).astype(int)
        metrics = compute_binary_metrics(labels, preds)
        metrics["threshold"] = t
        rows.append(metrics)
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_scores_csv(
    path: Path,
    attack_probs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Write per-session classification results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_index", "true_label", "attack_probability",
            "prediction_05", "label_name",
        ])
        for i in range(len(labels)):
            pred = 1 if attack_probs[i] >= 0.5 else 0
            label_name = "attack" if labels[i] == 1 else "normal"
            writer.writerow([
                int(indices[i]),
                int(labels[i]),
                round(float(attack_probs[i]), 6),
                pred,
                label_name,
            ])


# ---------------------------------------------------------------------------
# Alert engine integration
# ---------------------------------------------------------------------------

def run_alert_engine(
    attack_probs: np.ndarray,
    labels: np.ndarray,
    anomaly_ceiling: float = 1.0,
) -> dict:
    """Feed real classifier outputs into the alert engine."""
    engine = AlertEngine(dedup_window_secs=0, anomaly_ceiling=anomaly_ceiling)

    sessions = []
    for i in range(len(labels)):
        prob = float(attack_probs[i])
        classification = "attack" if prob >= 0.5 else "normal"
        sessions.append(ScoredSession(
            session_id=f"eval_{i:06d}",
            user_id=f"user_{i % 100:04d}",
            host_id=f"host_{i % 20:03d}",
            anomaly_score=prob * 15.0,  # scale to typical anomaly range
            classification=classification,
            classification_confidence=prob if prob >= 0.5 else (1 - prob),
            technique_id="T1078" if classification == "attack" else None,
        ))

    alerts = engine.process_batch(sessions)

    # Check alert quality
    alert_labels = []
    for alert in alerts:
        idx = int(alert.session_id.split("_")[1])
        alert_labels.append(labels[idx])

    true_attack_alerts = sum(alert_labels)
    false_alerts = len(alert_labels) - true_attack_alerts

    return {
        "total_alerts": len(alerts),
        "true_attack_alerts": int(true_attack_alerts),
        "false_alerts": int(false_alerts),
        "alert_precision": round(true_attack_alerts / max(len(alerts), 1), 4),
        "severity_distribution": {
            level: sum(1 for a in alerts if a.alert_class == level)
            for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
        "engine_summary": engine.get_stats(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS attack classifier")
    parser.add_argument("--classifier", required=True, help="Path to best_classifier.pt")
    parser.add_argument("--attack-manifest", required=True, help="Attack sessions manifest")
    parser.add_argument("--normal-manifest", required=True, help="Normal sessions manifest")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--max-normal", type=int, default=10_000)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # ── Load classifier ──────────────────────────────────────────────
    print("\n── Loading classifier ──")
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    model = ARGUSClassifier(
        num_classes=checkpoint.get("num_classes", 2),
        freeze_layers=0,  # doesn't matter for eval
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
          f"train F1={checkpoint.get('best_f1', 0):.4f}")

    # ── Load evaluation data ─────────────────────────────────────────
    print("\n── Loading evaluation data ──")
    dataset = EvalDataset()
    n_attack = dataset.add_from_manifest(args.attack_manifest, label=1)
    n_normal = dataset.add_from_manifest(
        args.normal_manifest, label=0,
        max_sessions=args.max_normal, limit_chunks=args.limit_chunks,
    )
    print(f"  Attack: {n_attack:,} | Normal: {n_normal:,} | Total: {len(dataset):,}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_eval, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Run inference ────────────────────────────────────────────────
    print("\n── Running inference ──")
    results = run_inference(model, loader, device)
    attack_probs = results["attack_prob"]
    labels = results["labels"]
    indices = results["indices"]
    print(f"  Scored {len(labels):,} sessions")

    # Score distribution
    attack_scores = attack_probs[labels == 1]
    normal_scores = attack_probs[labels == 0]
    print(f"\n  Attack P(attack):  mean={attack_scores.mean():.4f} "
          f"std={attack_scores.std():.4f} min={attack_scores.min():.4f} "
          f"max={attack_scores.max():.4f}")
    print(f"  Normal P(attack):  mean={normal_scores.mean():.4f} "
          f"std={normal_scores.std():.4f} min={normal_scores.min():.4f} "
          f"max={normal_scores.max():.4f}")

    # ── Write scores CSV ─────────────────────────────────────────────
    csv_path = output_dir / "classifier_scores.csv"
    write_scores_csv(csv_path, attack_probs, labels, indices)
    print(f"\n  Scores CSV → {csv_path}")

    # ── ROC-AUC and PR-AUC ───────────────────────────────────────────
    roc_auc = compute_roc_auc(labels, attack_probs)
    pr_auc = compute_pr_auc(labels, attack_probs)
    print(f"\n── Global Metrics ──")
    print(f"  ROC-AUC: {roc_auc:.6f}")
    print(f"  PR-AUC:  {pr_auc:.6f}")

    # ── Threshold sweep ──────────────────────────────────────────────
    print(f"\n── Threshold Sweep ──")
    sweep = threshold_sweep(labels, attack_probs)
    header = f"{'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FPR':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for row in sweep:
        print(f"  {row['threshold']:>8.2f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
              f"{row['f1']:>8.4f} {row['fpr']:>8.4f} {row['tp']:>6} {row['fp']:>6} "
              f"{row['fn']:>6} {row['tn']:>6}")

    # Find best threshold by F1
    best_thresh_row = max(sweep, key=lambda r: r["f1"])
    print(f"\n  Best threshold by F1: {best_thresh_row['threshold']:.2f} → "
          f"F1={best_thresh_row['f1']:.4f} Prec={best_thresh_row['precision']:.4f} "
          f"Rec={best_thresh_row['recall']:.4f}")

    # ── Alert engine integration ─────────────────────────────────────
    print(f"\n── Alert Engine (real classifier outputs) ──")
    alert_results = run_alert_engine(attack_probs, labels)
    print(f"  Total alerts:        {alert_results['total_alerts']}")
    print(f"  True attack alerts:  {alert_results['true_attack_alerts']}")
    print(f"  False alerts:        {alert_results['false_alerts']}")
    print(f"  Alert precision:     {alert_results['alert_precision']:.4f}")
    print(f"  Severity dist:       {alert_results['severity_distribution']}")

    # ── Save full report ─────────────────────────────────────────────
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
        "best_threshold": best_thresh_row,
        "threshold_sweep": sweep,
        "alert_engine_results": alert_results,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Full report → {report_path}")

    # Save threshold config for deployment
    threshold_config = {
        "best_f1_threshold": best_thresh_row["threshold"],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    threshold_path = output_dir / "calibrated_thresholds.json"
    threshold_path.write_text(json.dumps(threshold_config, indent=2))
    print(f"  Thresholds → {threshold_path}")

    print(f"\n✓ Phase 3 evaluation complete.")


if __name__ == "__main__":
    main()
