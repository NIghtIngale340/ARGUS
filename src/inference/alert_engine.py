"""ARGUS Phase 3.3 + 3.4 — Alert engine with composite severity and UEBA risk.

Combines:
- MLM anomaly score (unsupervised)
- Classification confidence (supervised)
- Technique severity (static lookup)
- User historical risk (EWMA in Redis)

Into a single composite severity score for alert prioritization.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np


@dataclass
class Alert:
    """A generated security alert."""
    alert_id: str
    timestamp: float
    user_id: str
    host_id: str
    session_id: str
    anomaly_score: float
    classification: str
    classification_confidence: float
    technique_id: Optional[str]
    technique_severity: float
    user_risk: float
    composite_severity: float
    alert_class: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredSession:
    """Input to the alert engine — a session that has been scored."""
    session_id: str
    user_id: str
    host_id: str
    anomaly_score: float
    classification: str
    classification_confidence: float
    technique_id: Optional[str] = None


DEFAULT_TECHNIQUE_SEVERITY = {
    "T1003": 0.95,
    "T1078": 0.80,
    "T1021": 0.75,
    "T1059": 0.70,
    "T1543": 0.65,
    "T1136": 0.60,
    "T1110": 0.85,
    "T1071": 0.55,
    "T1055": 0.90,
    "T1105": 0.60,
}


def load_technique_severity(path: Optional[str] = None) -> dict[str, float]:
    """Load technique severity from JSON file, or use defaults."""
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return DEFAULT_TECHNIQUE_SEVERITY.copy()


class RiskStore(Protocol):
    """Storage interface for per-user UEBA risk."""

    def get_risk(self, user_id: str) -> float:
        ...

    def update_risk(self, user_id: str, anomaly_score: float) -> float:
        ...

    def get_all_risks(self) -> dict[str, float]:
        ...


def normalize_anomaly_score(anomaly_score: float, ceiling: float = 15.0) -> float:
    """Normalize an unbounded anomaly score into [0, 1]."""
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")
    return min(max(float(anomaly_score) / ceiling, 0.0), 1.0)


class UEBARiskStore:
    """Per-user risk tracking using Exponentially Weighted Moving Average.

    In production, this would be backed by Redis. For Kaggle/batch evaluation,
    we use an in-memory dict with the same interface.

    EWMA formula:
        new_risk = old_risk * decay + new_score * (1 - decay)

    A user who repeatedly triggers high anomaly scores has a rising risk.
    A user who returns to normal has a decaying risk.
    """

    def __init__(self, decay: float = 0.95, default_risk: float = 0.0) -> None:
        self.decay = decay
        self.default_risk = default_risk
        self._store: dict[str, float] = {}
        self._last_update: dict[str, float] = {}

    def get_risk(self, user_id: str) -> float:
        """Get current risk score for a user."""
        return self._store.get(user_id, self.default_risk)

    def update_risk(self, user_id: str, anomaly_score: float) -> float:
        """Update and return the new EWMA risk score.

        Args:
            user_id: Hashed user identifier.
            anomaly_score: Raw anomaly score from the model (unbounded positive).

        Returns:
            Updated risk score.
        """
        old_risk = self._store.get(user_id, self.default_risk)
        normalized = normalize_anomaly_score(anomaly_score)
        new_risk = old_risk * self.decay + normalized * (1.0 - self.decay)
        self._store[user_id] = new_risk
        self._last_update[user_id] = time.time()
        return new_risk

    def get_all_risks(self) -> dict[str, float]:
        """Return all user risk scores."""
        return dict(self._store)


class RedisUEBARiskStore:
    """Redis-backed per-user UEBA risk tracking.

    Stores the same EWMA risk values as ``UEBARiskStore`` but persists them in
    Redis hashes so API and worker processes can share user risk state.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: Any | None = None,
        decay: float = 0.95,
        default_risk: float = 0.0,
        key_prefix: str = "argus:ueba",
        socket_timeout: float = 2.0,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.decay = decay
        self.default_risk = default_risk
        self.key_prefix = key_prefix.rstrip(":")
        self.risk_key = f"{self.key_prefix}:risk"
        self.last_update_key = f"{self.key_prefix}:last_update"

        if client is None:
            if not redis_url:
                raise ValueError("redis_url is required when client is not provided")
            try:
                import redis
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install redis to use RedisUEBARiskStore") from exc
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
            )
        self.client = client

    def ping(self) -> bool:
        """Return True when Redis responds to PING."""
        return bool(self.client.ping())

    def get_risk(self, user_id: str) -> float:
        stored = self.client.hget(self.risk_key, str(user_id))
        if stored is None:
            return self.default_risk
        try:
            return float(stored)
        except (TypeError, ValueError):
            return self.default_risk

    def update_risk(self, user_id: str, anomaly_score: float) -> float:
        old_risk = self.get_risk(user_id)
        normalized = normalize_anomaly_score(anomaly_score)
        new_risk = old_risk * self.decay + normalized * (1.0 - self.decay)
        now = time.time()
        self.client.hset(self.risk_key, str(user_id), repr(new_risk))
        self.client.hset(self.last_update_key, str(user_id), repr(now))
        return new_risk

    def get_all_risks(self) -> dict[str, float]:
        raw = self.client.hgetall(self.risk_key)
        risks: dict[str, float] = {}
        for user_id, value in raw.items():
            try:
                risks[str(user_id)] = float(value)
            except (TypeError, ValueError):
                continue
        return risks

    def clear(self) -> None:
        """Delete stored UEBA risk state for tests or local resets."""
        self.client.delete(self.risk_key, self.last_update_key)


class AlertEngine:
    """Processes scored sessions into prioritized alerts.

    Combines multiple signals into a composite severity:
        severity = w_anomaly * anomaly_norm
                 + w_confidence * classification_confidence
                 + w_technique * technique_severity
                 + w_user * user_risk
    """

    SEVERITY_WEIGHTS = {
        "anomaly": 0.35,
        "confidence": 0.25,
        "technique": 0.20,
        "user_risk": 0.20,
    }

    SEVERITY_THRESHOLDS = {
        "CRITICAL": 0.80,
        "HIGH": 0.60,
        "MEDIUM": 0.40,
        "LOW": 0.0,
    }

    def __init__(
        self,
        technique_severity_path: Optional[str] = None,
        risk_store: Optional[RiskStore] = None,
        anomaly_ceiling: float = 15.0,
        dedup_window_secs: float = 300.0,
    ) -> None:
        self.technique_severity = load_technique_severity(technique_severity_path)
        self.risk_store = risk_store or UEBARiskStore()
        self.anomaly_ceiling = anomaly_ceiling
        self.dedup_window = dedup_window_secs
        self._recent_alerts: dict[str, float] = {}
        self._alert_counter = 0

    def process_session(self, session: ScoredSession) -> Optional[Alert]:
        """Score a session and optionally generate an alert.

        Returns None if:
        - Classification is "normal" with high confidence
        - Alert is deduplicated (same user within dedup window)

        Returns an Alert if the session warrants analyst attention.
        """
        if session.classification == "normal" and session.classification_confidence > 0.9:
            self.risk_store.update_risk(session.user_id, session.anomaly_score)
            return None

        user_risk = self.risk_store.update_risk(session.user_id, session.anomaly_score)

        tech_sev = self.technique_severity.get(
            session.technique_id or "", 0.5
        )

        anomaly_norm = min(session.anomaly_score / self.anomaly_ceiling, 1.0)
        composite = (
            self.SEVERITY_WEIGHTS["anomaly"] * anomaly_norm
            + self.SEVERITY_WEIGHTS["confidence"] * session.classification_confidence
            + self.SEVERITY_WEIGHTS["technique"] * tech_sev
            + self.SEVERITY_WEIGHTS["user_risk"] * user_risk
        )

        alert_class = "LOW"
        for cls, threshold in sorted(
            self.SEVERITY_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if composite >= threshold:
                alert_class = cls
                break

        now = time.time()
        last_alert = self._recent_alerts.get(session.user_id, 0.0)
        if now - last_alert < self.dedup_window:
            return None

        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter:06d}"

        alert = Alert(
            alert_id=alert_id,
            timestamp=now,
            user_id=session.user_id,
            host_id=session.host_id,
            session_id=session.session_id,
            anomaly_score=session.anomaly_score,
            classification=session.classification,
            classification_confidence=session.classification_confidence,
            technique_id=session.technique_id,
            technique_severity=tech_sev,
            user_risk=user_risk,
            composite_severity=composite,
            alert_class=alert_class,
        )

        self._recent_alerts[session.user_id] = now
        return alert

    def process_batch(self, sessions: list[ScoredSession]) -> list[Alert]:
        """Process a batch of sessions and return all generated alerts."""
        alerts = []
        for session in sessions:
            alert = self.process_session(session)
            if alert:
                alerts.append(alert)
        return alerts

    def get_stats(self) -> dict:
        """Return engine statistics."""
        risks = self.risk_store.get_all_risks()
        return {
            "total_alerts_generated": self._alert_counter,
            "users_alerted": len(self._recent_alerts),
            "avg_user_risk": float(np.mean(list(risks.values()))) if risks else 0.0,
        }

    def summary(self) -> dict:
        """Backward-compatible alias for older notebook cells."""
        return self.get_stats()


def save_alerts_to_csv(alerts: list[Alert], output_path: str) -> None:
    """Save alerts to CSV."""
    if not alerts:
        print(f"No alerts to save.")
        return

    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "alert_id", "timestamp", "user_id", "host_id", "session_id",
            "anomaly_score", "classification", "classification_confidence",
            "technique_id", "technique_severity", "user_risk",
            "composite_severity", "alert_class"
        ])
        writer.writeheader()
        for alert in alerts:
            writer.writerow(alert.to_dict())

    print(f"Saved {len(alerts)} alerts → {output_path}")
