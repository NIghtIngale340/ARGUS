"""Unit tests for streaming session token tracking."""

from __future__ import annotations

import pytest

from src.inference.session_tracker import SessionTracker


class FakePipeline:
    def __init__(self, client: "FakeRedis") -> None:
        self.client = client
        self.commands: list[tuple[str, tuple]] = []

    def lrange(self, key: str, start: int, end: int) -> "FakePipeline":
        self.commands.append(("lrange", (key, start, end)))
        return self

    def delete(self, *keys: str) -> "FakePipeline":
        self.commands.append(("delete", keys))
        return self

    def execute(self) -> list:
        results = []
        for command, args in self.commands:
            if command == "lrange":
                results.append(self.client.lrange(*args))
            elif command == "delete":
                results.append(self.client.delete(*args))
        return results


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.expirations: dict[str, int] = {}

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
            self.expirations.pop(key, None)
        return deleted

    def expire(self, key: str, ttl_seconds: int) -> bool:
        self.expirations[key] = int(ttl_seconds)
        return True

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


def test_append_tracks_completion_and_flushes_tokens() -> None:
    client = FakeRedis()
    tracker = SessionTracker(client=client, max_tokens=3, ttl_seconds=60)

    assert tracker.append("user_a", "host_1", 10) == 1
    assert not tracker.is_complete("user_a", "host_1")
    assert tracker.append("user_a", "host_1", 11) == 2
    assert not tracker.is_complete("user_a", "host_1")
    assert tracker.append("user_a", "host_1", 12) == 3
    assert tracker.is_complete("user_a", "host_1")

    assert tracker.flush("user_a", "host_1") == [10, 11, 12]
    assert not tracker.is_complete("user_a", "host_1")
    assert tracker.flush("user_a", "host_1") == []


def test_sessions_are_isolated_by_user_and_host() -> None:
    client = FakeRedis()
    tracker = SessionTracker(client=client, max_tokens=2)

    tracker.append("user_a", "host_1", 100)
    tracker.append("user_a", "host_2", 200)
    tracker.append("user_b", "host_1", 300)
    tracker.append("user_a", "host_1", 101)

    assert tracker.is_complete("user_a", "host_1")
    assert not tracker.is_complete("user_a", "host_2")
    assert not tracker.is_complete("user_b", "host_1")
    assert tracker.flush("user_a", "host_1") == [100, 101]
    assert tracker.flush("user_a", "host_2") == [200]
    assert tracker.flush("user_b", "host_1") == [300]


def test_sessions_are_isolated_by_replay_run_id() -> None:
    client = FakeRedis()
    tracker = SessionTracker(client=client, max_tokens=2)

    tracker.append("user_a", "host_1", 100, replay_run_id="run_a")
    tracker.append("user_a", "host_1", 200, replay_run_id="run_b")
    tracker.append("user_a", "host_1", 101, replay_run_id="run_a")

    assert tracker.is_complete("user_a", "host_1", replay_run_id="run_a")
    assert not tracker.is_complete("user_a", "host_1", replay_run_id="run_b")
    assert tracker.flush("user_a", "host_1", replay_run_id="run_a") == [100, 101]
    assert tracker.flush("user_a", "host_1", replay_run_id="run_b") == [200]


def test_state_persists_across_tracker_instances_with_same_client() -> None:
    client = FakeRedis()
    first = SessionTracker(client=client, max_tokens=2, key_prefix="test:sessions")
    second = SessionTracker(client=client, max_tokens=2, key_prefix="test:sessions")

    first.append("user_a", "host_1", 7)
    first.append("user_a", "host_1", 8)

    assert second.is_complete("user_a", "host_1")
    assert second.flush("user_a", "host_1") == [7, 8]
    assert not first.is_complete("user_a", "host_1")


def test_append_refreshes_ttl_when_enabled() -> None:
    client = FakeRedis()
    tracker = SessionTracker(client=client, ttl_seconds=120)

    tracker.append("user_a", "host_1", 5)
    key = tracker.session_key("user_a", "host_1")

    assert client.expirations[key] == 120


def test_ttl_can_be_disabled() -> None:
    client = FakeRedis()
    tracker = SessionTracker(client=client, ttl_seconds=None)

    tracker.append("user_a", "host_1", 5)

    assert client.expirations == {}


def test_clear_deletes_one_session() -> None:
    client = FakeRedis()
    tracker = SessionTracker(client=client)

    tracker.append("user_a", "host_1", 1)
    tracker.append("user_a", "host_2", 2)
    tracker.clear("user_a", "host_1")

    assert tracker.flush("user_a", "host_1") == []
    assert tracker.flush("user_a", "host_2") == [2]


def test_invalid_configuration_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        SessionTracker(client=FakeRedis(), max_tokens=0)

    with pytest.raises(ValueError, match="ttl_seconds"):
        SessionTracker(client=FakeRedis(), ttl_seconds=0)

    with pytest.raises(ValueError, match="redis_url"):
        SessionTracker()
