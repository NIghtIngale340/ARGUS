"""ARGUS API entrypoint."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.alert_store import AlertStore, ElasticsearchAlertStore
from src.inference.phase3_detection import Phase3DetectionService


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


class SessionRequest(BaseModel):
    session_id: str | int | None = None
    user_id: str = ""
    host_id: str = ""
    replay_run_id: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class DetectionRequest(BaseModel):
    sessions: list[SessionRequest]
    threshold: float | None = None
    technique_id: str | None = None


def create_app(
    bundle_dir: str | None = None,
    *,
    alert_store: AlertStore | None = None,
) -> FastAPI:
    app = FastAPI(title="ARGUS API", version="0.1.0")
    configured_bundle = bundle_dir or os.getenv("ARGUS_PHASE3_BUNDLE_DIR")
    redis_url = os.getenv("REDIS_URL") if _env_flag("ARGUS_USE_REDIS_UEBA") else None
    redis_key_prefix = os.getenv("ARGUS_UEBA_REDIS_PREFIX", "argus:ueba")
    if alert_store is None and _env_flag("ARGUS_USE_ELASTICSEARCH_ALERTS"):
        alert_store = ElasticsearchAlertStore(
            os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
            index_prefix=os.getenv("ARGUS_ALERT_INDEX_PREFIX", "argus-alerts"),
        )

    if configured_bundle:
        app.state.phase3_detector = Phase3DetectionService.from_bundle_dir(
            configured_bundle,
            threshold=_env_optional_float("ARGUS_PHASE3_THRESHOLD"),
            redis_url=redis_url,
            redis_key_prefix=redis_key_prefix,
            alert_store=alert_store,
        )
        app.state.phase3_bundle_dir = configured_bundle
        app.state.redis_ueba_enabled = redis_url is not None
        app.state.elasticsearch_alerts_enabled = alert_store is not None
        app.state.alert_store = alert_store
    else:
        app.state.phase3_detector = None
        app.state.phase3_bundle_dir = None
        app.state.redis_ueba_enabled = False
        app.state.elasticsearch_alerts_enabled = False
        app.state.alert_store = alert_store

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "ARGUS API",
            "phase3_model_loaded": app.state.phase3_detector is not None,
            "phase3_bundle_dir": app.state.phase3_bundle_dir,
            "redis_ueba_enabled": app.state.redis_ueba_enabled,
            "elasticsearch_alerts_enabled": app.state.elasticsearch_alerts_enabled,
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

    def _require_alert_store() -> AlertStore:
        alert_store: AlertStore | None = app.state.alert_store
        if alert_store is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Elasticsearch alert querying is not configured. "
                    "Set ARGUS_USE_ELASTICSEARCH_ALERTS=true."
                ),
            )
        return alert_store

    @app.get("/phase3/alerts")
    def list_phase3_alerts(
        limit: int = 50,
        user_id: str | None = None,
        host_id: str | None = None,
        session_id: str | None = None,
        alert_class: str | None = None,
        technique_id: str | None = None,
        replay_run_id: str | None = None,
        min_severity: float | None = None,
    ) -> dict[str, Any]:
        if limit <= 0 or limit > 500:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
        if min_severity is not None and not 0.0 <= min_severity <= 1.0:
            raise HTTPException(status_code=400, detail="min_severity must be in [0, 1]")

        alert_store = _require_alert_store()
        alerts = alert_store.search_alerts(
            limit=limit,
            user_id=user_id,
            host_id=host_id,
            session_id=session_id,
            alert_class=alert_class,
            technique_id=technique_id,
            replay_run_id=replay_run_id,
            min_severity=min_severity,
        )
        return {
            "count": len(alerts),
            "limit": limit,
            "elasticsearch_alerts_enabled": app.state.elasticsearch_alerts_enabled,
            "alerts": alerts,
        }

    @app.get("/phase3/alerts/{alert_id}")
    def get_phase3_alert(alert_id: str) -> dict[str, Any]:
        alert_store = _require_alert_store()
        alerts = alert_store.search_alerts(alert_id=alert_id, limit=1)
        if not alerts:
            raise HTTPException(status_code=404, detail="alert not found")
        return {"alert": alerts[0]}

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

    @app.get("/phase3/ueba/risks/{user_id}/timeline")
    def get_ueba_risk_timeline(user_id: str, days: int = 30) -> dict[str, Any]:
        detector = _require_detector()
        if days <= 0:
            raise HTTPException(status_code=400, detail="days must be positive")
        timeline = detector.alert_engine.risk_store.get_risk_timeline(
            user_id,
            days=days,
        )
        return {
            "user_id": user_id,
            "days": days,
            "count": len(timeline),
            "timeline": timeline,
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
