"""Kafka/Faust streaming bridge for ARGUS Phase 4 detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from src.inference.alert_store import ElasticsearchAlertStore
from src.inference.phase3_detection import Phase3DetectionService
from src.inference.session_tracker import SessionTracker

try:
    from faust import App
except Exception:
    App = None  # type: ignore[assignment]


DEFAULT_RAW_TOPIC = "argus.raw-logs"
DEFAULT_DETECTIONS_TOPIC = "argus.detections"


class StreamingSessionProcessor:
    """Accumulate streaming events and score completed sessions."""

    def __init__(
        self,
        *,
        tracker: SessionTracker,
        detector: Any,
        tokenizer: Any | None = None,
        max_seq_len: int | None = None,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        self.tracker = tracker
        self.detector = detector
        self.tokenizer = tokenizer or getattr(detector, "tokenizer", None)
        self.max_seq_len = int(
            max_seq_len
            or getattr(detector, "max_seq_len", 0)
            or getattr(self.tokenizer, "max_len", 0)
            or 16
        )
        if self.max_seq_len < 3:
            raise ValueError("max_seq_len must be >= 3")
        self.cls_token_id = int(
            cls_token_id
            if cls_token_id is not None
            else getattr(self.tokenizer, "cls_token", 0)
        )
        self.sep_token_id = int(
            sep_token_id
            if sep_token_id is not None
            else getattr(self.tokenizer, "sep_token", 1)
        )
        self.pad_token_id = int(
            pad_token_id
            if pad_token_id is not None
            else getattr(self.tokenizer, "pad_token", 3)
        )
        self._session_counter = 0

    def process_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Process one parsed event and return detections for flushed sessions."""
        user_id, host_id = extract_user_host(event)
        token_id = self.extract_token_id(event)
        replay_run_id = extract_replay_run_id(event)

        self.tracker.append(
            user_id,
            host_id,
            token_id,
            replay_run_id=replay_run_id,
        )
        if not self.tracker.is_complete(user_id, host_id, replay_run_id=replay_run_id):
            return []

        token_ids = self.tracker.flush(user_id, host_id, replay_run_id=replay_run_id)
        if not token_ids:
            return []

        item = self.build_detection_item(
            token_ids,
            user_id=user_id,
            host_id=host_id,
            source_event=event,
        )
        rows = self.detector.score_items([item])
        return [self._annotate_detection(row, token_ids, event) for row in rows]

    def extract_token_id(self, event: Mapping[str, Any]) -> int:
        """Get a token ID from the event or encode it with the detector tokenizer."""
        for field in ("token_id", "event_token_id"):
            if field in event and event[field] is not None:
                return int(event[field])

        if self.tokenizer is None or not hasattr(self.tokenizer, "encode_event"):
            raise ValueError("event must include token_id or a tokenizer must be provided")
        return int(self.tokenizer.encode_event(event))

    def build_detection_item(
        self,
        token_ids: list[int],
        *,
        user_id: str,
        host_id: str,
        source_event: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the tensor item expected by Phase3DetectionService.score_items."""
        input_ids, attention_mask = self._build_model_inputs(token_ids)
        session_id = source_event.get("session_id") or self._next_session_id(
            user_id,
            host_id,
        )
        return {
            "session_id": str(session_id),
            "user_id": user_id,
            "host_id": host_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            **_detection_metadata_from_event(source_event),
        }

    def _build_model_inputs(self, token_ids: list[int]) -> tuple[list[int], list[int]]:
        max_events = self.max_seq_len - 2
        event_tokens = [int(token_id) for token_id in token_ids[-max_events:]]
        input_ids = [self.cls_token_id] + event_tokens + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)

        padding_needed = self.max_seq_len - len(input_ids)
        if padding_needed > 0:
            input_ids.extend([self.pad_token_id] * padding_needed)
            attention_mask.extend([0] * padding_needed)
        return input_ids, attention_mask

    def _next_session_id(self, user_id: str, host_id: str) -> str:
        self._session_counter += 1
        return f"stream_{user_id}_{host_id}_{self._session_counter:08d}"

    @staticmethod
    def _annotate_detection(
        row: dict[str, Any],
        token_ids: list[int],
        source_event: Mapping[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(row)
        enriched["stream_token_count"] = len(token_ids)
        if "event_time" in source_event:
            enriched["last_event_time"] = source_event["event_time"]
        elif "timestamp" in source_event:
            enriched["last_event_time"] = source_event["timestamp"]
        replay_run_id = extract_replay_run_id(source_event)
        if replay_run_id is not None:
            enriched["replay_run_id"] = replay_run_id
        return enriched


def extract_user_host(event: Mapping[str, Any]) -> tuple[str, str]:
    """Extract stable user/host identifiers from common parsed log fields."""
    user_id = event.get("user_id") or event.get("user") or event.get("src_user")
    host_id = (
        event.get("host_id")
        or event.get("host")
        or event.get("computer")
        or event.get("src_host")
    )
    if user_id is None or str(user_id) == "":
        raise ValueError("event is missing user_id/user/src_user")
    if host_id is None or str(host_id) == "":
        raise ValueError("event is missing host_id/host/computer/src_host")
    return str(user_id), str(host_id)


def extract_replay_run_id(event: Mapping[str, Any]) -> str | None:
    """Extract the optional replay run identifier from a streaming event."""
    replay_run_id = event.get("replay_run_id") or event.get("run_id")
    if replay_run_id is None or str(replay_run_id) == "":
        return None
    return str(replay_run_id)


def _detection_metadata_from_event(event: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    replay_run_id = extract_replay_run_id(event)
    if replay_run_id is not None:
        metadata["replay_run_id"] = replay_run_id
    return metadata


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def create_alert_store_from_env() -> ElasticsearchAlertStore | None:
    """Create optional Elasticsearch alert persistence from env vars."""
    if not _env_bool("ARGUS_USE_ELASTICSEARCH_ALERTS", False):
        return None
    elasticsearch_url = (
        os.getenv("ELASTICSEARCH_URL")
        or os.getenv("ES_URL")
        or "http://elasticsearch:9200"
    )
    return ElasticsearchAlertStore(
        elasticsearch_url,
        index_prefix=os.getenv("ARGUS_ALERT_INDEX_PREFIX", "argus-alerts"),
    )


def create_streaming_processor_from_env() -> StreamingSessionProcessor:
    """Create the Redis session tracker and Phase 3 detector from env vars."""
    bundle_dir = os.getenv("ARGUS_PHASE3_BUNDLE_DIR")
    if not bundle_dir:
        raise RuntimeError("ARGUS_PHASE3_BUNDLE_DIR is required for Kafka detection")

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    alert_store = create_alert_store_from_env()
    detector = Phase3DetectionService.from_bundle_dir(
        Path(bundle_dir),
        threshold=_env_optional_float("ARGUS_PHASE3_THRESHOLD"),
        redis_url=redis_url if _env_bool("ARGUS_USE_REDIS_UEBA", True) else None,
        redis_key_prefix=os.getenv("ARGUS_UEBA_REDIS_PREFIX", "argus:ueba"),
        alert_store=alert_store,
    )
    max_events = _env_optional_int("ARGUS_STREAM_SESSION_MAX_TOKENS")
    if max_events is None:
        max_events = max(int(detector.max_seq_len) - 2, 1)

    tracker = SessionTracker(
        redis_url,
        key_prefix=os.getenv("ARGUS_STREAM_SESSION_PREFIX", "argus:session"),
        max_tokens=max_events,
        ttl_seconds=_env_optional_int("ARGUS_STREAM_SESSION_TTL_SECONDS") or 1800,
    )
    return StreamingSessionProcessor(
        tracker=tracker,
        detector=detector,
        max_seq_len=int(detector.max_seq_len),
    )


_processor: StreamingSessionProcessor | None = None


def get_streaming_processor() -> StreamingSessionProcessor:
    """Return a lazily initialized process-local streaming processor."""
    global _processor
    if _processor is None:
        _processor = create_streaming_processor_from_env()
    return _processor


def reset_streaming_processor() -> None:
    """Clear the cached processor. Intended for tests and process reloads."""
    global _processor
    _processor = None


def create_faust_app() -> Any:
    """Create the Faust app. Returns None when faust-streaming is unavailable."""
    if App is None:
        return None
    return App(
        "argus-consumer",
        broker=os.getenv("KAFKA_BOOTSTRAP", "kafka://kafka:9092"),
        store=os.getenv("FAUST_STORE", os.getenv("REDIS_URL", "redis://redis:6379/0")),
    )


app = create_faust_app()

if app is not None:
    raw_logs = app.topic(
        os.getenv("ARGUS_RAW_LOG_TOPIC", DEFAULT_RAW_TOPIC),
        value_serializer="json",
    )
    detections = app.topic(
        os.getenv("ARGUS_DETECTIONS_TOPIC", DEFAULT_DETECTIONS_TOPIC),
        value_serializer="json",
    )

    @app.agent(raw_logs)
    async def consume(stream: Iterable[Mapping[str, Any]]) -> None:
        async for event in stream:
            for detection in get_streaming_processor().process_event(event):
                await detections.send(value=detection)
else:
    raw_logs = None
    detections = None

    async def consume(stream: Iterable[Mapping[str, Any]]) -> None:
        raise RuntimeError("Install faust-streaming to run the Kafka consumer")


__all__ = [
    "DEFAULT_DETECTIONS_TOPIC",
    "DEFAULT_RAW_TOPIC",
    "StreamingSessionProcessor",
    "app",
    "consume",
    "create_alert_store_from_env",
    "create_streaming_processor_from_env",
    "extract_replay_run_id",
    "extract_user_host",
    "get_streaming_processor",
    "reset_streaming_processor",
]
