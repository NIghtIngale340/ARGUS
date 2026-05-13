"""Security and operational middleware for the ARGUS API."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hmac
import json
import logging
import os
import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware


LOGGER = logging.getLogger("argus.api")
DEFAULT_EXEMPT_PATHS = {"/health", "/ready"}


def configure_logging() -> None:
    """Configure a compact JSON logger when the host app has not configured one."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")


def log_json(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    LOGGER.log(level, json.dumps(payload, sort_keys=True, default=str))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _looks_like_browser_accept(value: str | None) -> bool:
    return bool(value and "text/html" in value.lower())


@dataclass(frozen=True)
class ApiSecuritySettings:
    api_key: str | None
    admin_api_key: str | None
    dashboard_username: str
    cors_origins: list[str]
    trusted_hosts: list[str]
    max_request_bytes: int
    rate_limit_enabled: bool
    rate_limit_per_minute: int
    rate_limit_redis_url: str | None

    @classmethod
    def from_env(cls) -> "ApiSecuritySettings":
        return cls(
            api_key=os.getenv("ARGUS_API_KEY") or None,
            admin_api_key=os.getenv("ARGUS_ADMIN_API_KEY") or None,
            dashboard_username=os.getenv("ARGUS_DASHBOARD_USERNAME", "argus"),
            cors_origins=_env_csv(
                "ARGUS_CORS_ORIGINS",
                "http://127.0.0.1:8000,http://localhost:8000",
            ),
            trusted_hosts=_env_csv(
                "ARGUS_TRUSTED_HOSTS",
                "127.0.0.1,localhost,testserver",
            ),
            max_request_bytes=_env_int("ARGUS_MAX_REQUEST_BYTES", 1_048_576),
            rate_limit_enabled=_env_bool("ARGUS_RATE_LIMIT_ENABLED", True),
            rate_limit_per_minute=_env_int("ARGUS_RATE_LIMIT_PER_MINUTE", 120),
            rate_limit_redis_url=(
                os.getenv("ARGUS_RATE_LIMIT_REDIS_URL")
                if _env_bool("ARGUS_RATE_LIMIT_USE_REDIS", False)
                else None
            ),
        )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request IDs, enforce body size, and emit structured request logs."""

    def __init__(
        self,
        app: Any,
        *,
        max_request_bytes: int,
    ) -> None:
        super().__init__(app)
        self.max_request_bytes = int(max_request_bytes)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                byte_count = int(content_length)
            except ValueError:
                byte_count = -1
            if byte_count < 0 or byte_count > self.max_request_bytes:
                return _json_error(
                    413,
                    "request_too_large",
                    f"request body exceeds {self.max_request_bytes} bytes",
                    request_id=request_id,
                )

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        response.headers["X-Request-ID"] = request_id
        log_json(
            logging.INFO,
            "api_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_ms,
        )
        return response


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require an API key for every non-probe endpoint."""

    def __init__(
        self,
        app: Any,
        *,
        settings: ApiSecuritySettings,
        exempt_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self.exempt_paths = exempt_paths or DEFAULT_EXEMPT_PATHS

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        if request.method == "OPTIONS" or request.url.path in self.exempt_paths:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", uuid4().hex)
        if not self.settings.api_key:
            return _json_error(
                503,
                "api_key_not_configured",
                "ARGUS_API_KEY is required for protected endpoints.",
                request_id=request_id,
            )

        token = extract_auth_token(request, dashboard_username=self.settings.dashboard_username)
        if token is None or not _constant_time_equal(token, self.settings.api_key):
            headers = {}
            if _looks_like_browser_accept(request.headers.get("accept")):
                headers["WWW-Authenticate"] = 'Basic realm="ARGUS Dashboard"'
            return _json_error(
                401,
                "unauthorized",
                "valid ARGUS API key required",
                request_id=request_id,
                headers=headers,
            )

        request.state.argus_api_key_authenticated = True
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Small fixed-window rate limiter with Redis when configured."""

    def __init__(
        self,
        app: Any,
        *,
        settings: ApiSecuritySettings,
        exempt_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self.exempt_paths = exempt_paths or DEFAULT_EXEMPT_PATHS
        self._memory_counts: dict[str, tuple[int, float]] = {}
        self._redis_client = self._create_redis_client(settings.rate_limit_redis_url)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        if (
            not self.settings.rate_limit_enabled
            or request.method == "OPTIONS"
            or request.url.path in self.exempt_paths
        ):
            return await call_next(request)

        limit = max(int(self.settings.rate_limit_per_minute), 1)
        identity = _rate_limit_identity(request)
        allowed, remaining = self._increment(identity, limit)
        if not allowed:
            return _json_error(
                429,
                "rate_limit_exceeded",
                f"rate limit exceeded: {limit} requests per minute",
                request_id=getattr(request.state, "request_id", None),
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))
        return response

    def _create_redis_client(self, redis_url: str | None) -> Any | None:
        if not redis_url:
            return None
        try:
            import redis

            return redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
        except Exception as exc:
            log_json(logging.WARNING, "rate_limit_redis_unavailable", error=str(exc))
            return None

    def _increment(self, identity: str, limit: int) -> tuple[bool, int]:
        now = int(time.time())
        window = now // 60
        if self._redis_client is not None:
            key = f"argus:api:rate:{window}:{identity}"
            try:
                count = int(self._redis_client.incr(key))
                if count == 1:
                    self._redis_client.expire(key, 60)
                return count <= limit, limit - count
            except Exception as exc:
                log_json(logging.WARNING, "rate_limit_redis_failed", error=str(exc))

        memory_key = f"{window}:{identity}"
        count, expires_at = self._memory_counts.get(memory_key, (0, time.time() + 60.0))
        count += 1
        self._memory_counts[memory_key] = (count, expires_at)
        self._memory_counts = {
            key: value for key, value in self._memory_counts.items() if value[1] > time.time()
        }
        return count <= limit, limit - count


def install_security_middleware(app: Any) -> ApiSecuritySettings:
    settings = ApiSecuritySettings.from_env()
    allow_all_origins = settings.cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all_origins else settings.cors_origins,
        allow_credentials=not allow_all_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-ARGUS-API-Key", "X-Request-ID"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(ApiKeyAuthMiddleware, settings=settings)
    app.add_middleware(RequestContextMiddleware, max_request_bytes=settings.max_request_bytes)
    app.state.security_settings = settings
    return settings


def extract_auth_token(request: Request, *, dashboard_username: str = "argus") -> str | None:
    header_key = request.headers.get("X-ARGUS-API-Key")
    if header_key:
        return header_key

    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    if authorization.lower().startswith("basic "):
        encoded = authorization.split(" ", 1)[1].strip()
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except Exception:
            return None
        username, _, password = decoded.partition(":")
        if username == dashboard_username and password:
            return password
    return None


def require_admin_request(request: Request) -> None:
    settings: ApiSecuritySettings = request.app.state.security_settings
    request_id = getattr(request.state, "request_id", None)
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "admin_key_not_configured",
                "message": "ARGUS_ADMIN_API_KEY is required for this operation.",
                "request_id": request_id,
            },
        )

    admin_token = (
        request.headers.get("X-ARGUS-Admin-Key")
        or request.headers.get("X-ARGUS-API-Key")
        or extract_auth_token(request, dashboard_username=settings.dashboard_username)
    )
    if admin_token is None or not _constant_time_equal(admin_token, settings.admin_api_key):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "admin_key_required",
                "message": "admin API key required",
                "request_id": request_id,
            },
        )


def _rate_limit_identity(request: Request) -> str:
    token = request.headers.get("X-ARGUS-API-Key") or request.headers.get("Authorization")
    if token:
        return f"token:{hash(token)}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _json_error(
    status_code: int,
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "detail": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


__all__ = [
    "ApiSecuritySettings",
    "configure_logging",
    "extract_auth_token",
    "install_security_middleware",
    "log_json",
    "require_admin_request",
]
