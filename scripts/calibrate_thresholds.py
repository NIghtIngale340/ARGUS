"""Compute and save anomaly-score thresholds from a scored CSV.

Usage
-----
python scripts/calibrate_thresholds.py \
    --scores /path/to/argus_val_scores.csv \
    --out /path/to/thresholds.json \
    --percentiles 50,90,95,99,99.5,99.9 \
    --operational-pct p95
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.thresholds import (
    calibrate_thresholds,
    save_thresholds,
    set_operational_threshold,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Calibrate anomaly score thresholds")
    parser.add_argument(
        "--scores", required=True,
        help="Path to scored CSV (must have 'anomaly_score' column).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for thresholds JSON.",
    )
    parser.add_argument(
        "--percentiles", default="50,90,95,99,99.5,99.9",
        help="Comma-separated percentile list (0-100 scale).",
    )
    parser.add_argument(
        "--operational-pct", default=None,
        help="Percentile key to use as operational threshold (e.g. p95).",
    )
    parser.add_argument(
        "--score-column", default="anomaly_score",
        help="Column name containing scores.",
    )
    return parser


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()

    df = pd.read_csv(parsed.scores)
    if parsed.score_column not in df.columns:
        raise ValueError(
            f"Column {parsed.score_column!r} not found. "
            f"Available: {list(df.columns)}"
        )

    scores = df[parsed.score_column].to_numpy()
    percentiles = [float(p) for p in parsed.percentiles.split(",")]

    print(f"Loaded {len(scores):,} scores from {parsed.scores}")
    print(f"Percentiles: {percentiles}")

    result = calibrate_thresholds(scores, percentiles)

    if parsed.operational_pct:
        result = set_operational_threshold(result, parsed.operational_pct)
        print(f"Operational threshold: {parsed.operational_pct} = "
              f"{result.operational_threshold:.6f}")

    out_path = save_thresholds(result, parsed.out)
    print(f"\n--- Calibration Summary ---")
    print(f"  Sessions:  {result.count:,}")
    print(f"  Mean:      {result.mean:.6f}")
    print(f"  Std:       {result.std:.6f}")
    print(f"  Min:       {result.min_score:.6f}")
    print(f"  Max:       {result.max_score:.6f}")
    print()
    for key, val in result.percentiles.items():
        print(f"  {key:>8s}:  {val:.6f}")

    vals = list(result.percentiles.values())
    unique = len(set(vals))
    if unique < len(vals):
        print(f"\n⚠  {len(vals) - unique} duplicate threshold(s) detected.")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
