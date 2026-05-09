"""Alert persistence backends for ARGUS detection output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from src.inference.alert_engine import Alert


class AlertStore(Protocol):
    def index_alert(self, alert: Alert, *, extra: dict[str, Any] | None = None) -> str:
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


__all__ = ["AlertStore", "ElasticsearchAlertStore", "InMemoryAlertStore"]
