"""Alert persistence backends for ARGUS detection output."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Protocol

from src.inference.alert_engine import Alert


class AlertStore(Protocol):
    def index_alert(self, alert: Alert, *, extra: dict[str, Any] | None = None) -> str:
        ...

    def search_alerts(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        host_id: str | None = None,
        session_id: str | None = None,
        alert_id: str | None = None,
        alert_class: str | None = None,
        technique_id: str | None = None,
        replay_run_id: str | None = None,
        min_severity: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


class InMemoryAlertStore:
    """Test/local alert store with the same interface as Elasticsearch."""

    def __init__(self) -> None:
        self.alerts: list[dict[str, Any]] = []

    def index_alert(self, alert: Alert, *, extra: dict[str, Any] | None = None) -> str:
        document = alert.to_dict()
        if extra:
            document.update(extra)
        self.alerts.append(document)
        return str(alert.alert_id)

    def search_alerts(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        host_id: str | None = None,
        session_id: str | None = None,
        alert_id: str | None = None,
        alert_class: str | None = None,
        technique_id: str | None = None,
        replay_run_id: str | None = None,
        min_severity: float | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            dict(alert)
            for alert in self.alerts
            if _matches_alert(
                alert,
                user_id=user_id,
                host_id=host_id,
                session_id=session_id,
                alert_id=alert_id,
                alert_class=alert_class,
                technique_id=technique_id,
                replay_run_id=replay_run_id,
                min_severity=min_severity,
            )
        ]
        rows.sort(key=lambda row: float(row.get("timestamp", 0.0)), reverse=True)
        return rows[:limit]


class ElasticsearchAlertStore:
    """Persist alerts into daily Elasticsearch indices."""

    def __init__(
        self,
        elasticsearch_url: str | None = None,
        *,
        client: Any | None = None,
        index_prefix: str = "argus-alerts",
        request_timeout: float = 5.0,
    ) -> None:
        if client is None:
            if not elasticsearch_url:
                raise ValueError("elasticsearch_url is required when client is not provided")
            try:
                from elasticsearch import Elasticsearch
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install elasticsearch to use ElasticsearchAlertStore") from exc
            client = Elasticsearch(elasticsearch_url, request_timeout=request_timeout)
        self.client = client
        self.index_prefix = index_prefix.rstrip("-")

    def index_alert(self, alert: Alert, *, extra: dict[str, Any] | None = None) -> str:
        document = alert.to_dict()
        if extra:
            document.update(extra)
        document["@timestamp"] = datetime.fromtimestamp(
            float(alert.timestamp),
            tz=timezone.utc,
        ).isoformat()
        index_name = self.index_for_timestamp(float(alert.timestamp))
        response = self.client.index(
            index=index_name,
            id=alert.alert_id,
            document=document,
        )
        return str(response.get("_id", alert.alert_id)) if isinstance(response, dict) else str(alert.alert_id)

    def index_for_timestamp(self, timestamp: float) -> str:
        day = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).strftime("%Y.%m.%d")
        return f"{self.index_prefix}-{day}"

    @property
    def index_pattern(self) -> str:
        return f"{self.index_prefix}-*"

    def search_alerts(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        host_id: str | None = None,
        session_id: str | None = None,
        alert_id: str | None = None,
        alert_class: str | None = None,
        technique_id: str | None = None,
        replay_run_id: str | None = None,
        min_severity: float | None = None,
    ) -> list[dict[str, Any]]:
        query = _build_alert_query(
            user_id=user_id,
            host_id=host_id,
            session_id=session_id,
            alert_id=alert_id,
            alert_class=alert_class,
            technique_id=technique_id,
            replay_run_id=replay_run_id,
            min_severity=min_severity,
        )
        try:
            response = self.client.search(
                index=self.index_pattern,
                query=query,
                sort=[{"@timestamp": {"order": "desc"}}],
                size=limit,
            )
        except TypeError:
            response = self.client.search(
                index=self.index_pattern,
                body={
                    "query": query,
                    "sort": [{"@timestamp": {"order": "desc"}}],
                    "size": limit,
                },
            )
        return _extract_search_sources(response)


def _build_alert_query(
    *,
    user_id: str | None = None,
    host_id: str | None = None,
    session_id: str | None = None,
    alert_id: str | None = None,
    alert_class: str | None = None,
    technique_id: str | None = None,
    replay_run_id: str | None = None,
    min_severity: float | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    for field, value in {
        "user_id": user_id,
        "host_id": host_id,
        "session_id": session_id,
        "alert_id": alert_id,
        "alert_class": alert_class,
        "technique_id": technique_id,
        "replay_run_id": replay_run_id,
    }.items():
        if value is not None and str(value) != "":
            filters.append(_exact_text_filter(field, str(value)))
    if min_severity is not None:
        filters.append({"range": {"composite_severity": {"gte": float(min_severity)}}})
    if not filters:
        return {"match_all": {}}
    return {"bool": {"filter": filters}}


def _exact_text_filter(field: str, value: str) -> dict[str, Any]:
    """Match exact text across explicit keyword and dynamic text.keyword mappings."""
    return {
        "bool": {
            "should": [
                {"term": {field: value}},
                {"term": {f"{field}.keyword": value}},
            ],
            "minimum_should_match": 1,
        }
    }


def _extract_search_sources(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping):
        response = getattr(response, "body", response)
    if not isinstance(response, Mapping):
        return []
    hits = response.get("hits", {}).get("hits", [])
    rows: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source")
        if isinstance(source, dict):
            row = dict(source)
            if "_id" in hit and "alert_id" not in row:
                row["alert_id"] = str(hit["_id"])
            rows.append(row)
    return rows


def _matches_alert(
    alert: dict[str, Any],
    *,
    user_id: str | None = None,
    host_id: str | None = None,
    session_id: str | None = None,
    alert_id: str | None = None,
    alert_class: str | None = None,
    technique_id: str | None = None,
    replay_run_id: str | None = None,
    min_severity: float | None = None,
) -> bool:
    for field, value in {
        "user_id": user_id,
        "host_id": host_id,
        "session_id": session_id,
        "alert_id": alert_id,
        "alert_class": alert_class,
        "technique_id": technique_id,
        "replay_run_id": replay_run_id,
    }.items():
        if value is not None and str(value) != "" and str(alert.get(field, "")) != str(value):
            return False
    if min_severity is not None and float(alert.get("composite_severity", 0.0)) < float(min_severity):
        return False
    return True


__all__ = [
    "AlertStore",
    "ElasticsearchAlertStore",
    "InMemoryAlertStore",
]
