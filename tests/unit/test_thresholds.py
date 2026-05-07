"""Unit tests for src.inference.thresholds."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.inference.thresholds import (
    ThresholdResult,
    calibrate_thresholds,
    classify_sessions,
    load_thresholds,
    save_thresholds,
    set_operational_threshold,
)


class TestCalibrateThresholds:
    def test_basic_percentiles(self):
        scores = np.arange(1, 101, dtype=float)  # 1..100
        result = calibrate_thresholds(scores, percentiles=[50, 90, 99])
        assert result.count == 100
        assert abs(result.mean - 50.5) < 0.01
        assert result.min_score == 1.0
        assert result.max_score == 100.0
        # p50 of 1..100 ≈ 50.5
        assert 49 <= result.percentiles["p50"] <= 51

    def test_custom_percentiles(self):
        scores = np.random.RandomState(42).normal(5.0, 2.0, size=10_000)
        result = calibrate_thresholds(scores, percentiles=[25, 75])
        assert "p25" in result.percentiles
        assert "p75" in result.percentiles
        assert result.percentiles["p25"] < result.percentiles["p75"]

    def test_single_value_array(self):
        result = calibrate_thresholds([7.0], percentiles=[50, 99])
        assert result.count == 1
        assert result.percentiles["p50"] == 7.0
        assert result.percentiles["p99"] == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calibrate_thresholds([])

    def test_list_input(self):
        result = calibrate_thresholds([1.0, 2.0, 3.0], percentiles=[50])
        assert result.count == 3


class TestSetOperationalThreshold:
    def test_sets_value(self):
        result = calibrate_thresholds(np.arange(100.0), percentiles=[95])
        result = set_operational_threshold(result, "p95")
        assert result.operational_threshold is not None
        assert result.operational_percentile == "p95"

    def test_invalid_key_raises(self):
        result = calibrate_thresholds(np.arange(100.0), percentiles=[95])
        with pytest.raises(KeyError, match="p50"):
            set_operational_threshold(result, "p50")


class TestClassifySessions:
    def test_basic_classification(self):
        scores = [1.0, 5.0, 10.0, 15.0]
        flags = classify_sessions(scores, threshold=10.0)
        np.testing.assert_array_equal(flags, [False, False, True, True])

    def test_all_below(self):
        flags = classify_sessions([1.0, 2.0], threshold=100.0)
        assert not flags.any()

    def test_all_above(self):
        flags = classify_sessions([50.0, 60.0], threshold=1.0)
        assert flags.all()

    def test_boundary_included(self):
        flags = classify_sessions([5.0], threshold=5.0)
        assert flags[0] is np.True_


class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        result = calibrate_thresholds(
            np.arange(1000.0), percentiles=[50, 90, 95, 99]
        )
        result = set_operational_threshold(result, "p95")

        path = tmp_path / "thresholds.json"
        save_thresholds(result, path)

        loaded = load_thresholds(path)
        assert loaded.count == result.count
        assert loaded.percentiles == result.percentiles
        assert loaded.operational_threshold == result.operational_threshold
        assert loaded.operational_percentile == result.operational_percentile

    def test_json_is_valid(self, tmp_path: Path):
        result = calibrate_thresholds([1.0, 2.0, 3.0], percentiles=[50])
        path = tmp_path / "out.json"
        save_thresholds(result, path)
        raw = json.loads(path.read_text())
        assert "percentiles" in raw
        assert "count" in raw

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "thresholds.json"
        result = calibrate_thresholds([1.0], percentiles=[50])
        save_thresholds(result, deep)
        assert deep.exists()
