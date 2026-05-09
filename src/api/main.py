"""ARGUS API entrypoint."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.phase3_detection import Phase3DetectionService


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class SessionRequest(BaseModel):
    session_id: str | int | None = None
    user_id: str = ""
    host_id: str = ""
    events: list[dict[str, Any]] = Field(default_factory=list)


class DetectionRequest(BaseModel):
    sessions: list[SessionRequest]
    threshold: float | None = None
    technique_id: str | None = None


def create_app(bundle_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="ARGUS API", version="0.1.0")
    configured_bundle = bundle_dir or os.getenv("ARGUS_PHASE3_BUNDLE_DIR")
    redis_url = os.getenv("REDIS_URL") if _env_flag("ARGUS_USE_REDIS_UEBA") else None
    redis_key_prefix = os.getenv("ARGUS_UEBA_REDIS_PREFIX", "argus:ueba")

    if configured_bundle:
        app.state.phase3_detector = Phase3DetectionService.from_bundle_dir(
            configured_bundle,
            redis_url=redis_url,
            redis_key_prefix=redis_key_prefix,
        )
        app.state.phase3_bundle_dir = configured_bundle
        app.state.redis_ueba_enabled = redis_url is not None
    else:
        app.state.phase3_detector = None
        app.state.phase3_bundle_dir = None
        app.state.redis_ueba_enabled = False

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "ARGUS API",
            "phase3_model_loaded": app.state.phase3_detector is not None,
            "phase3_bundle_dir": app.state.phase3_bundle_dir,
            "redis_ueba_enabled": app.state.redis_ueba_enabled,
        }

    @app.post("/phase3/detect")
    def detect(payload: DetectionRequest) -> dict[str, Any]:
        detector: Phase3DetectionService | None = app.state.phase3_detector
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Phase 3 detector is not configured. Set ARGUS_PHASE3_BUNDLE_DIR.",
            )
        if not payload.sessions:
            raise HTTPException(status_code=400, detail="sessions must not be empty")

        rows = detector.score_sessions(
            [session.model_dump() for session in payload.sessions],
            threshold=payload.threshold,
            technique_id=payload.technique_id,
        )
        return {
            "count": len(rows),
            "threshold": payload.threshold if payload.threshold is not None else detector.threshold,
            "alerts_generated": sum(1 for row in rows if row["alert_generated"]),
            "detections": rows,
        }

    def _require_detector() -> Phase3DetectionService:
        detector: Phase3DetectionService | None = app.state.phase3_detector
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Phase 3 detector is not configured. Set ARGUS_PHASE3_BUNDLE_DIR.",
            )
        return detector

    @app.get("/phase3/ueba/risks")
    def list_ueba_risks() -> dict[str, Any]:
        detector = _require_detector()
        risks = detector.alert_engine.risk_store.get_all_risks()
        return {
            "count": len(risks),
            "redis_ueba_enabled": app.state.redis_ueba_enabled,
            "risks": risks,
        }

    @app.get("/phase3/ueba/risks/{user_id}")
    def get_ueba_risk(user_id: str) -> dict[str, Any]:
        detector = _require_detector()
        risks = detector.alert_engine.risk_store.get_all_risks()
        return {
            "user_id": user_id,
            "risk": detector.alert_engine.risk_store.get_risk(user_id),
            "exists": user_id in risks,
            "redis_ueba_enabled": app.state.redis_ueba_enabled,
        }

    @app.delete("/phase3/ueba/risks")
    def clear_ueba_risks() -> dict[str, Any]:
        detector = _require_detector()
        before = len(detector.alert_engine.risk_store.get_all_risks())
        detector.alert_engine.risk_store.clear()
        return {
            "cleared": before,
            "remaining": len(detector.alert_engine.risk_store.get_all_risks()),
            "redis_ueba_enabled": app.state.redis_ueba_enabled,
        }

    return app


app = create_app()
