"""Redis-backed streaming session token accumulation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class SessionTracker:
    """Accumulate event token IDs per user/host until a session is complete.

    The tracker stores one Redis list per ``(user_id, host_id)`` pair. It is
    intentionally small: Kafka or API code can append parsed event tokens, call
    ``is_complete()``, then ``flush()`` the token sequence into the Phase 3
    detector when the configured token limit is reached.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: Any | None = None,
        key_prefix: str = "argus:session",
        max_tokens: int = 16,
        ttl_seconds: int | None = 1800,
        socket_timeout: float = 2.0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when provided")

        self.key_prefix = key_prefix.rstrip(":")
        self.max_tokens = int(max_tokens)
        self.ttl_seconds = ttl_seconds

        if client is None:
            if not redis_url:
                raise ValueError("redis_url is required when client is not provided")
            try:
                import redis
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install redis to use SessionTracker") from exc
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
            )
        self.client = client

    def session_key(
        self,
        user_id: str,
        host_id: str,
        *,
        replay_run_id: str | None = None,
    ) -> str:
        """Return the Redis key for a user/host streaming session."""
        raw = json.dumps(
            [str(user_id), str(host_id), str(replay_run_id or "")],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:tokens:{digest}"

    def append(
        self,
        user_id: str,
        host_id: str,
        token_id: int,
        *,
        replay_run_id: str | None = None,
    ) -> int:
        """Append one token and return the current token count."""
        key = self.session_key(user_id, host_id, replay_run_id=replay_run_id)
        count = self.client.rpush(key, str(int(token_id)))
        if self.ttl_seconds is not None:
            self.client.expire(key, self.ttl_seconds)
        if count is None:
            return int(self.client.llen(key))
        return int(count)

    def is_complete(
        self,
        user_id: str,
        host_id: str,
        *,
        replay_run_id: str | None = None,
    ) -> bool:
        """Return True when the user/host session reached ``max_tokens``."""
        key = self.session_key(user_id, host_id, replay_run_id=replay_run_id)
        return int(self.client.llen(key)) >= self.max_tokens

    def flush(
        self,
        user_id: str,
        host_id: str,
        *,
        replay_run_id: str | None = None,
    ) -> list[int]:
        """Return accumulated token IDs and delete the stored session."""
        key = self.session_key(user_id, host_id, replay_run_id=replay_run_id)

        if hasattr(self.client, "pipeline"):
            pipe = self.client.pipeline()
            pipe.lrange(key, 0, -1)
            pipe.delete(key)
            values = pipe.execute()[0]
        else:
            values = self.client.lrange(key, 0, -1)
            self.client.delete(key)

        return [self._coerce_token_id(value) for value in values]

    def clear(
        self,
        user_id: str,
        host_id: str,
        *,
        replay_run_id: str | None = None,
    ) -> None:
        """Delete any accumulated token IDs for a user/host pair."""
        self.client.delete(self.session_key(user_id, host_id, replay_run_id=replay_run_id))

    @staticmethod
    def _coerce_token_id(value: Any) -> int:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return int(value)


__all__ = ["SessionTracker"]
