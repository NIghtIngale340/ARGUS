from collections import Counter

import pytest

import src.parsing.session_builder as session_builder_module
from src.parsing.session_builder import SessionBuilder


def test_groups_by_user_and_host_and_drops_sparse_groups() -> None:
    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=3, timestamp_unit="seconds")
    events = [
        {"user": "alice", "host": "pc1", "time": 0, "event_id": "A0"},
        {"user": "alice", "host": "pc1", "time": 300, "event_id": "A1"},
        {"user": "alice", "host": "pc1", "time": 900, "event_id": "A2"},
        {"user": "alice", "host": "pc2", "time": 0, "event_id": "B0"},
        {"user": "alice", "host": "pc2", "time": 200, "event_id": "B1"},
        {"user": "alice", "host": "pc2", "time": 400, "event_id": "B2"},
        {"user": "bob", "host": "pc9", "time": 0, "event_id": "C0"},
        {"user": "bob", "host": "pc9", "time": 600, "event_id": "C1"},
    ]

    sessions = builder.build_sessions(events)

    assert len(sessions) == 2
    assert {session.start_ts for session in sessions} == {0}
    assert {session.end_ts for session in sessions} == {1800}

    expected_keys = {
        (builder._hash_id("alice"), builder._hash_id("pc1")),
        (builder._hash_id("alice"), builder._hash_id("pc2")),
    }
    produced_keys = {(session.user_id, session.host_id) for session in sessions}
    assert produced_keys == expected_keys


def test_stride_creates_overlapping_sessions() -> None:
    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=3, timestamp_unit="seconds")
    events = [
        {"user": "alice", "host": "pc1", "time": 0, "event_id": "E0"},
        {"user": "alice", "host": "pc1", "time": 600, "event_id": "E1"},
        {"user": "alice", "host": "pc1", "time": 1200, "event_id": "E2"},
        {"user": "alice", "host": "pc1", "time": 1500, "event_id": "E3"},
        {"user": "alice", "host": "pc1", "time": 1800, "event_id": "E4"},
        {"user": "alice", "host": "pc1", "time": 2100, "event_id": "E5"},
        {"user": "alice", "host": "pc1", "time": 2400, "event_id": "E6"},
    ]

    sessions = builder.build_sessions(events)

    assert len(sessions) >= 2
    assert sessions[0].start_ts == 0
    assert sessions[1].start_ts == 900

    first_times = {event["time"] for event in sessions[0].events}
    second_times = {event["time"] for event in sessions[1].events}
    assert first_times.intersection(second_times)


def test_sessions_with_less_than_min_events_are_dropped() -> None:
    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=3, timestamp_unit="seconds")
    events = [
        {"user": "alice", "host": "pc1", "time": 0, "event_id": "E0"},
        {"user": "alice", "host": "pc1", "time": 600, "event_id": "E1"},
    ]

    sessions = builder.build_sessions(events)
    assert sessions == []


def test_sessions_are_truncated_to_last_max_tokens() -> None:
    builder = SessionBuilder(
        window_mins=60,
        stride_mins=60,
        min_events=1,
        max_tokens=5,
        timestamp_unit="seconds",
    )
    events = [
        {"user": "alice", "host": "pc1", "time": idx, "event_id": f"E{idx}"}
        for idx in range(10)
    ]

    sessions = builder.build_sessions(events)

    assert len(sessions) == 1
    assert len(sessions[0].events) == 5
    assert [event["time"] for event in sessions[0].events] == [5, 6, 7, 8, 9]


def test_invalid_builder_configuration_raises() -> None:
    with pytest.raises(ValueError):
        SessionBuilder(window_mins=0)


def test_sensitive_event_fields_are_removed_and_event_id_is_backfilled() -> None:
    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=1, timestamp_unit="seconds")
    events = [
        {
            "user": "alice",
            "host": "pc1",
            "src_user": "alice",
            "dst_user": "bob",
            "src_computer": "pc1",
            "dst_computer": "pc2",
            "template_params": ["alice", "pc1"],
            "template_id": 42,
            "auth_type": "Kerberos",
            "logon_type": "Network",
            "time": 0,
        }
    ]

    sessions = builder.build_sessions(events)

    assert len(sessions) == 1
    event = sessions[0].events[0]
    assert event["event_id"] == "42"
    assert event["auth_type"] == "Kerberos"
    assert event["logon_type"] == "Network"

    for key in (
        "user",
        "host",
        "src_user",
        "dst_user",
        "src_computer",
        "dst_computer",
        "template_params",
    ):
        assert key not in event


def test_overlapping_windows_do_not_resanitize_events() -> None:
    class CountingSessionBuilder(SessionBuilder):
        def __init__(self) -> None:
            super().__init__(window_mins=30, stride_mins=15, min_events=3, timestamp_unit="seconds")
            self.sanitized_event_ids: list[str] = []

        def _sanitize_event(self, event):  # type: ignore[override]
            self.sanitized_event_ids.append(str(event["event_id"]))
            return super()._sanitize_event(event)

    builder = CountingSessionBuilder()
    events = [
        {"user": "alice", "host": "pc1", "time": 0, "event_id": "E0"},
        {"user": "alice", "host": "pc1", "time": 600, "event_id": "E1"},
        {"user": "alice", "host": "pc1", "time": 1200, "event_id": "E2"},
        {"user": "alice", "host": "pc1", "time": 1500, "event_id": "E3"},
        {"user": "alice", "host": "pc1", "time": 1800, "event_id": "E4"},
        {"user": "alice", "host": "pc1", "time": 2100, "event_id": "E5"},
        {"user": "alice", "host": "pc1", "time": 2400, "event_id": "E6"},
    ]

    sessions = builder.build_sessions(events)

    assert len(sessions) >= 2
    assert sum(len(session.events) for session in sessions) > len(events)
    assert Counter(builder.sanitized_event_ids) == Counter(event["event_id"] for event in events)


def test_hash_id_cache_preserves_outputs_for_normal_and_blank_ids() -> None:
    session_builder_module._hash_normalized_id.cache_clear()
    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=1, timestamp_unit="seconds")

    alice_hash = builder._hash_id("alice")

    assert builder._hash_id("alice") == alice_hash
    assert builder._hash_id(None) == "UNKNOWN"
    assert builder._hash_id("   ") == "UNKNOWN"

    cache_info = session_builder_module._hash_normalized_id.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits >= 1
