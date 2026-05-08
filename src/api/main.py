"""ARGUS API entrypoint."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.phase3_detection import Phase3DetectionService


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

    if configured_bundle:
        app.state.phase3_detector = Phase3DetectionService.from_bundle_dir(
            configured_bundle
        )
        app.state.phase3_bundle_dir = configured_bundle
    else:
        app.state.phase3_detector = None
        app.state.phase3_bundle_dir = None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "ARGUS API",
            "phase3_model_loaded": app.state.phase3_detector is not None,
            "phase3_bundle_dir": app.state.phase3_bundle_dir,
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

    return app


app = create_app()
