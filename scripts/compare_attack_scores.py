"""Compare normal vs attack session scores and compute detection metrics.

Usage
-----
python scripts/compare_attack_scores.py \
    --normal-scores /path/to/argus_val_scores.csv \
    --attack-scores /path/to/argus_attack_scores.csv \
    --thresholds /path/to/thresholds.json \
    --out /path/to/comparison_report.json
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.thresholds import load_thresholds, classify_sessions


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare normal vs attack anomaly scores")
    parser.add_argument("--normal-scores", required=True,
                        help="CSV of normal/validation session scores.")
    parser.add_argument("--attack-scores", required=True,
                        help="CSV of attack/red-team session scores.")
    parser.add_argument("--thresholds", required=True,
                        help="Thresholds JSON from calibrate_thresholds.py.")
    parser.add_argument("--out", required=True,
                        help="Output path for comparison report JSON.")
    parser.add_argument("--score-column", default="anomaly_score")
    return parser


def compute_roc_auc(normal: np.ndarray, attack: np.ndarray) -> float:
    """Compute ROC-AUC from two score arrays (normal=0, attack=1)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return _mannwhitney_auc(normal, attack)

    labels = np.concatenate([np.zeros(len(normal)), np.ones(len(attack))])
    scores = np.concatenate([normal, attack])
    return float(roc_auc_score(labels, scores))


def _mannwhitney_auc(normal: np.ndarray, attack: np.ndarray) -> float:
    """Fallback AUC via Mann-Whitney U statistic (no sklearn needed)."""
    n_normal = len(normal)
    n_attack = len(attack)
    if n_normal == 0 or n_attack == 0:
        return 0.0

    all_scores = np.concatenate([normal, attack])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_attack)])

    order = all_scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)

    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    u_attack = ranks[labels == 1].sum() - n_attack * (n_attack + 1) / 2.0
    return float(u_attack / (n_normal * n_attack))


def threshold_metrics(
    normal: np.ndarray,
    attack: np.ndarray,
    threshold: float,
) -> dict:
    """Compute precision, recall, FPR at a single threshold."""
    normal_flagged = classify_sessions(normal, threshold)
    attack_flagged = classify_sessions(attack, threshold)

    tp = int(attack_flagged.sum())
    fp = int(normal_flagged.sum())
    fn = int((~attack_flagged).sum())
    tn = int((~normal_flagged).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "fpr": round(fpr, 6),
        "f1": round(f1, 6),
    }


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()

    col = parsed.score_column
    normal_df = pd.read_csv(parsed.normal_scores)
    attack_df = pd.read_csv(parsed.attack_scores)

    for label, df in [("normal", normal_df), ("attack", attack_df)]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in {label} CSV. "
                             f"Available: {list(df.columns)}")

    normal = normal_df[col].to_numpy(dtype=np.float64)
    attack = attack_df[col].to_numpy(dtype=np.float64)
    thresholds = load_thresholds(parsed.thresholds)

    print(f"Normal sessions:  {len(normal):,}")
    print(f"Attack sessions:  {len(attack):,}")

    roc_auc = compute_roc_auc(normal, attack)
    print(f"ROC-AUC:          {roc_auc:.6f}")

    normal_mean = float(normal.mean())
    attack_mean = float(attack.mean())
    pooled_std = float(np.sqrt((normal.std() ** 2 + attack.std() ** 2) / 2))
    cohens_d = ((attack_mean - normal_mean) / pooled_std
                if pooled_std > 0 else 0.0)

    print(f"Normal mean:      {normal_mean:.4f}")
    print(f"Attack mean:      {attack_mean:.4f}")
    print(f"Cohen's d:        {cohens_d:.4f}")

    per_threshold = {}
    for key, thr_val in thresholds.percentiles.items():
        metrics = threshold_metrics(normal, attack, thr_val)
        per_threshold[key] = metrics
        print(f"  {key:>8s} (thr={thr_val:.4f}): "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.4f}")
    report = {
        "normal_count": len(normal),
        "attack_count": len(attack),
        "roc_auc": round(roc_auc, 6),
        "normal_mean": round(normal_mean, 6),
        "attack_mean": round(attack_mean, 6),
        "cohens_d": round(cohens_d, 6),
        "per_threshold": per_threshold,
    }

    if thresholds.operational_threshold is not None:
        op = threshold_metrics(normal, attack, thresholds.operational_threshold)
        report["operational"] = {
            "percentile": thresholds.operational_percentile,
            **op,
        }

    out_path = Path(parsed.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
