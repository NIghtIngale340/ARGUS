import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

Event = Dict[str, Any]

SENSITIVE_EVENT_FIELDS = {
    "user",
    "host",
    "username",
    "source_user",
    "source_host",
    "src_user",
    "dst_user",
    "src_host",
    "src_computer",
    "dst_computer",
    "template_params",
}


@lru_cache(maxsize=4096)
def _hash_normalized_id(normalized: str) -> str:
    hash_obj = hashlib.sha256(normalized.encode("utf-8"))
    return hash_obj.hexdigest()[:8]


@dataclass
class Session:
    user_id: str
    host_id: str
    start_ts: int
    end_ts: int
    events: List[Event]
    label: Optional[int] = None


class SessionBuilder:
    def __init__(
        self,
        window_mins: int = 30,
        stride_mins: int = 15,
        min_events: int = 3,
        max_tokens: int = 512,
        timestamp_unit: str = "seconds",
    ):
        self.window_mins = window_mins
        self.stride_mins = stride_mins
        self.min_events = min_events
        self.max_tokens = max_tokens
        self.timestamp_unit = timestamp_unit
        self._validate_config()

    def _validate_config(self) -> None:
        if self.window_mins <= 0:
            raise ValueError("window_mins must be > 0")
        if self.stride_mins <= 0:
            raise ValueError("stride_mins must be > 0")
        if self.min_events <= 0:
            raise ValueError("min_events must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.timestamp_unit not in {"seconds", "milliseconds"}:
            raise ValueError("timestamp_unit must be either 'seconds' or 'milliseconds'")

    def _minutes_to_time_units(self, minutes: int) -> int:
        scale = 60 if self.timestamp_unit == "seconds" else 60_000
        return minutes * scale

    def _hash_id(self, raw_id: Any) -> str:
        normalized = str(raw_id).strip() if raw_id is not None else ""
        if not normalized:
            return "UNKNOWN"
        return _hash_normalized_id(normalized)

    def _extract_timestamp(self, event: Event) -> Optional[int]:
        for key in ("time", "timestamp"):
            value = event.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _extract_group_key(self, event: Event) -> Tuple[str, str]:
        raw_user = (
            event.get("user")
            or event.get("src_user")
            or event.get("source_user")
            or event.get("username")
            or "UNKNOWN"
        )
        raw_host = (
            event.get("host")
            or event.get("src_computer")
            or event.get("src_host")
            or event.get("source_host")
            or "UNKNOWN"
        )
        return self._hash_id(raw_user), self._hash_id(raw_host)

    def _sanitize_event(self, event: Event) -> Event:
        sanitized = dict(event)

        if "event_id" not in sanitized or sanitized.get("event_id") in {None, ""}:
            sanitized["event_id"] = str(sanitized.get("template_id", "UNK"))

        for field in SENSITIVE_EVENT_FIELDS:
            sanitized.pop(field, None)

        return sanitized

    def _build_group_sessions(self, user_id: str, host_id: str, events: List[Event]) -> List[Session]:
        if not events:
            return []

        events.sort(key=lambda event: event["time"])

        window_size = self._minutes_to_time_units(self.window_mins)
        stride_size = self._minutes_to_time_units(self.stride_mins)

        sessions: List[Session] = []
        window_start = events[0]["time"]
        last_event_time = events[-1]["time"]
        start_idx = 0
        end_idx = 0
        total_events = len(events)

        while window_start <= last_event_time:
            window_end = window_start + window_size

            while start_idx < total_events and events[start_idx]["time"] < window_start:
                start_idx += 1

            if end_idx < start_idx:
                end_idx = start_idx

            while end_idx < total_events and events[end_idx]["time"] < window_end:
                end_idx += 1

            event_count = end_idx - start_idx
            if event_count >= self.min_events:
                window_events = events[start_idx:end_idx]
                if event_count > self.max_tokens:
                    window_events = window_events[-self.max_tokens :]

                sessions.append(
                    Session(
                        user_id=user_id,
                        host_id=host_id,
                        start_ts=window_start,
                        end_ts=window_end,
                        events=window_events,
                    )
                )

            window_start += stride_size

        return sessions

    def build_sessions(self, events: List[Event]) -> List[Session]:
        grouped_events: Dict[Tuple[str, str], List[Event]] = defaultdict(list)

        for event in events:
            event_time = self._extract_timestamp(event)
            if event_time is None:
                continue
            normalized_event = dict(event)
            normalized_event["time"] = event_time

            user_id, host_id = self._extract_group_key(normalized_event)
            sanitized_event = self._sanitize_event(normalized_event)
            grouped_events[(user_id, host_id)].append(sanitized_event)

        sessions: List[Session] = []
        for (user_id, host_id), user_host_events in grouped_events.items():
            sessions.extend(self._build_group_sessions(user_id, host_id, user_host_events))

        return sessions
