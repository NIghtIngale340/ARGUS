"""Unit tests for the direct Phase 4 streaming consumer runner."""

from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.run_phase4_streaming_consumer import normalize_bootstrap, run_consumer


def _args(**overrides):
    values = {
        "bootstrap_server": "kafka://localhost:29092",
        "raw_topic": "argus.raw-logs",
        "detections_topic": "argus.detections",
        "dead_letter_topic": "argus.dead-letter",
        "group_id": "test-group",
        "from_beginning": False,
        "poll_timeout": 1.0,
        "duration_seconds": 1.0,
        "max_events": None,
        "max_errors": 100,
        "score_batch_size": 16,
        "skip_topic_create": True,
        "topic_ready_timeout": 60.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_normalize_bootstrap_strips_faust_scheme() -> None:
    assert normalize_bootstrap("kafka://kafka:9092") == "kafka:9092"
    assert normalize_bootstrap("localhost:29092") == "localhost:29092"


def test_run_consumer_rejects_invalid_topic_timeout() -> None:
    with pytest.raises(ValueError, match="topic-ready-timeout"):
        run_consumer(_args(topic_ready_timeout=0))
