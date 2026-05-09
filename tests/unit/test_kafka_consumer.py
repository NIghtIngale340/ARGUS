"""Unit tests for the Phase 4 streaming Kafka consumer core."""

from __future__ import annotations

import pytest
import torch

from src.inference.kafka_consumer import (
    StreamingSessionProcessor,
    extract_user_host,
)
from src.inference.session_tracker import SessionTracker


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}

    def rpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).append(str(value))
        return len(self.lists[key])

    def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.lists:
                deleted += 1
                self.lists.pop(key, None)
        return deleted

    def expire(self, key: str, ttl_seconds: int) -> bool:
        return True


class FakeTokenizer:
    max_len = 6
    cls_token = 0
    sep_token = 1
    pad_token = 3

    def encode_event(self, event: dict) -> int:
        return int(event.get("encoded_token", 42))


class FakeDetector:
    def __init__(self) -> None:
        self.max_seq_len = 6
        self.tokenizer = FakeTokenizer()
        self.batches: list[list[dict]] = []

    def score_items(self, batch: list[dict]) -> list[dict]:
        self.batches.append(batch)
        item = batch[0]
        return [
            {
                "session_id": item["session_id"],
                "user_id": item["user_id"],
                "host_id": item["host_id"],
                "attack_probability": 0.99,
                "prediction": "attack",
                "threshold": 0.999,
                "alert_generated": True,
                "alert_class": "HIGH",
                "composite_severity": 0.7,
            }
        ]


def make_processor(max_tokens: int = 4) -> tuple[StreamingSessionProcessor, FakeDetector]:
    detector = FakeDetector()
    tracker = SessionTracker(
        client=FakeRedis(),
        key_prefix="test:session",
        max_tokens=max_tokens,
    )
    processor = StreamingSessionProcessor(tracker=tracker, detector=detector)
    return processor, detector


def test_processor_scores_only_when_session_is_complete() -> None:
    processor, detector = make_processor(max_tokens=4)

    base_event = {"user_id": "user_a", "host_id": "host_1"}
    assert processor.process_event({**base_event, "token_id": 10}) == []
    assert processor.process_event({**base_event, "token_id": 11}) == []
    assert processor.process_event({**base_event, "token_id": 12}) == []

    rows = processor.process_event({**base_event, "token_id": 13})

    assert len(rows) == 1
    assert rows[0]["prediction"] == "attack"
    assert rows[0]["stream_token_count"] == 4
    assert rows[0]["user_id"] == "user_a"
    assert rows[0]["host_id"] == "host_1"
    assert len(detector.batches) == 1

    scored_item = detector.batches[0][0]
    assert scored_item["input_ids"].tolist() == [0, 10, 11, 12, 13, 1]
    assert scored_item["attention_mask"].tolist() == [True] * 6


def test_processor_pads_short_flushed_sessions() -> None:
    processor, detector = make_processor(max_tokens=2)

    processor.process_event({"user_id": "user_a", "host_id": "host_1", "token_id": 21})
    rows = processor.process_event({"user_id": "user_a", "host_id": "host_1", "token_id": 22})

    assert len(rows) == 1
    scored_item = detector.batches[0][0]
    assert scored_item["input_ids"].tolist() == [0, 21, 22, 1, 3, 3]
    assert scored_item["attention_mask"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_processor_isolates_sessions_by_user_and_host() -> None:
    processor, detector = make_processor(max_tokens=2)

    assert processor.process_event({"user_id": "u1", "host_id": "h1", "token_id": 1}) == []
    assert processor.process_event({"user_id": "u1", "host_id": "h2", "token_id": 2}) == []
    assert processor.process_event({"user_id": "u2", "host_id": "h1", "token_id": 3}) == []

    rows = processor.process_event({"user_id": "u1", "host_id": "h1", "token_id": 4})

    assert len(rows) == 1
    assert detector.batches[0][0]["user_id"] == "u1"
    assert detector.batches[0][0]["host_id"] == "h1"
    assert detector.batches[0][0]["input_ids"].tolist() == [0, 1, 4, 1, 3, 3]
    assert processor.process_event({"user_id": "u1", "host_id": "h2", "token_id": 5})
    assert processor.process_event({"user_id": "u2", "host_id": "h1", "token_id": 6})


def test_processor_can_encode_parsed_event_when_token_id_is_missing() -> None:
    processor, detector = make_processor(max_tokens=1)

    rows = processor.process_event(
        {
            "user": "user_a",
            "computer": "host_1",
            "encoded_token": 77,
            "timestamp": 12345,
        }
    )

    assert len(rows) == 1
    assert rows[0]["last_event_time"] == 12345
    assert detector.batches[0][0]["input_ids"].tolist() == [0, 77, 1, 3, 3, 3]


def test_source_session_id_is_preserved() -> None:
    processor, detector = make_processor(max_tokens=1)

    processor.process_event(
        {
            "session_id": "upstream_session_1",
            "user_id": "user_a",
            "host_id": "host_1",
            "token_id": 31,
        }
    )

    assert detector.batches[0][0]["session_id"] == "upstream_session_1"


def test_extract_user_host_accepts_common_field_names() -> None:
    assert extract_user_host({"user": "u1", "computer": "h1"}) == ("u1", "h1")
    assert extract_user_host({"src_user": "u2", "src_host": "h2"}) == ("u2", "h2")
    assert extract_user_host({"user_id": "u3", "host": "h3"}) == ("u3", "h3")


def test_invalid_events_raise_clear_errors() -> None:
    processor, _ = make_processor(max_tokens=1)
    processor.tokenizer = None

    with pytest.raises(ValueError, match="user_id"):
        processor.process_event({"host_id": "host_1", "token_id": 1})

    with pytest.raises(ValueError, match="host_id"):
        processor.process_event({"user_id": "user_a", "token_id": 1})

    with pytest.raises(ValueError, match="token_id"):
        processor.process_event({"user_id": "user_a", "host_id": "host_1"})


def test_detection_tensors_are_torch_tensors() -> None:
    processor, detector = make_processor(max_tokens=1)

    processor.process_event({"user_id": "user_a", "host_id": "host_1", "token_id": 9})

    item = detector.batches[0][0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.bool
