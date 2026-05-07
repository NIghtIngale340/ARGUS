"""Threshold calibration and session classification for anomaly scoring."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9)


@dataclass
class ThresholdResult:
    """Container for calibrated threshold values and summary statistics."""

    percentiles: dict[str, float] = field(default_factory=dict)
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    operational_threshold: float | None = None
    operational_percentile: str | None = None


def calibrate_thresholds(
    scores: np.ndarray | Sequence[float],
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> ThresholdResult:
    """Compute percentile thresholds from an array of anomaly scores.

    Parameters
    ----------
    scores:
        1-D array of per-session anomaly scores.
    percentiles:
        Percentile breakpoints (0-100 scale).

    Returns
    -------
    ThresholdResult with computed percentile map and summary stats.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot calibrate thresholds from an empty score array.")

    values = np.percentile(arr, list(percentiles))
    pct_map = {f"p{p:g}": float(v) for p, v in zip(percentiles, values)}

    return ThresholdResult(
        percentiles=pct_map,
        count=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std()),
        min_score=float(arr.min()),
        max_score=float(arr.max()),
    )


def set_operational_threshold(
    result: ThresholdResult,
    percentile_key: str,
) -> ThresholdResult:
    """Mark one percentile as the operational decision boundary.

    Parameters
    ----------
    result:
        A previously calibrated ThresholdResult.
    percentile_key:
        Key such as ``"p95"`` or ``"p99"`` present in ``result.percentiles``.
    """
    if percentile_key not in result.percentiles:
        raise KeyError(
            f"{percentile_key!r} not in calibrated percentiles: "
            f"{list(result.percentiles.keys())}"
        )
    result.operational_threshold = result.percentiles[percentile_key]
    result.operational_percentile = percentile_key
    return result


def classify_sessions(
    scores: np.ndarray | Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Return a boolean mask where True = anomalous (score >= threshold)."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    return arr >= threshold


def save_thresholds(result: ThresholdResult, path: str | Path) -> Path:
    """Serialize a ThresholdResult to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out


def load_thresholds(path: str | Path) -> ThresholdResult:
    """Deserialize a ThresholdResult from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ThresholdResult(**raw)
