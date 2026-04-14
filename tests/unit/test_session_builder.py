import pytest

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
