"""Verify Phase 4 Kafka detections and Elasticsearch alert persistence.

Run this after the Faust consumer is running and replay traffic has been sent.
It checks the detection topic and, unless disabled, the ``argus-alerts-*``
Elasticsearch index pattern.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import os
import sys
import time
from typing import Any
from uuid import uuid4


DEFAULT_BOOTSTRAP = "localhost:29092"
DEFAULT_DETECTIONS_TOPIC = "argus.detections"
DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_ALERT_INDEX = "argus-alerts-*"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Verify ARGUS Phase 4 streaming outputs")
    parser.add_argument(
        "--bootstrap-server",
        default=os.getenv("KAFKA_BOOTSTRAP", DEFAULT_BOOTSTRAP),
    )
    parser.add_argument(
        "--detections-topic",
        default=os.getenv("ARGUS_DETECTIONS_TOPIC", DEFAULT_DETECTIONS_TOPIC),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--min-detections", type=int, default=1)
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--skip-kafka", action="store_true")
    parser.add_argument("--skip-elasticsearch", action="store_true")
    parser.add_argument(
        "--es-url",
        default=os.getenv("ELASTICSEARCH_URL") or os.getenv("ES_URL") or DEFAULT_ES_URL,
    )
    parser.add_argument("--alert-index", default=DEFAULT_ALERT_INDEX)
    parser.add_argument("--min-alerts", type=int, default=0)
    return parser


def consume_detection_samples(
    *,
    bootstrap_server: str,
    topic: str,
    timeout_seconds: float,
    min_detections: int,
    from_beginning: bool = False,
) -> list[dict[str, Any]]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if min_detections < 0:
        raise ValueError("min_detections must be >= 0")

    try:
        from confluent_kafka import Consumer, KafkaException
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install confluent-kafka to verify Kafka detections "
            "(for example: pip install confluent-kafka)."
        ) from exc

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_server,
            "group.id": f"argus-phase4-verify-{uuid4()}",
            "auto.offset.reset": "earliest" if from_beginning else "latest",
            "enable.auto.commit": False,
        }
    )
    rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    consumer.subscribe([topic])
    try:
        while time.monotonic() < deadline and len(rows) < min_detections:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            value = msg.value()
            if value is None:
                continue
            payload = json.loads(value.decode("utf-8"))
            if isinstance(payload, dict):
                rows.append(payload)
    finally:
        consumer.close()
    return rows


def count_elasticsearch_alerts(*, es_url: str, index_pattern: str) -> int:
    try:
        from elasticsearch import Elasticsearch, NotFoundError
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install elasticsearch to verify alert persistence "
            "(for example: pip install elasticsearch)."
        ) from exc

    client = Elasticsearch(es_url)
    try:
        response = client.count(index=index_pattern)
    except NotFoundError:
        return 0
    return int(response.get("count", 0))


def main(args: Namespace | None = None) -> int:
    parsed = args or build_parser().parse_args()
    failures: list[str] = []

    if not parsed.skip_kafka:
        detections = consume_detection_samples(
            bootstrap_server=parsed.bootstrap_server,
            topic=parsed.detections_topic,
            timeout_seconds=parsed.timeout_seconds,
            min_detections=parsed.min_detections,
            from_beginning=parsed.from_beginning,
        )
        print(
            f"Kafka detections: {len(detections):,} "
            f"from topic {parsed.detections_topic}"
        )
        for row in detections[:3]:
            print(json.dumps(row, sort_keys=True))
        if len(detections) < parsed.min_detections:
            failures.append(
                f"expected at least {parsed.min_detections} Kafka detection(s)"
            )

    if not parsed.skip_elasticsearch:
        alert_count = count_elasticsearch_alerts(
            es_url=parsed.es_url,
            index_pattern=parsed.alert_index,
        )
        print(f"Elasticsearch alerts: {alert_count:,} in {parsed.alert_index}")
        if alert_count < parsed.min_alerts:
            failures.append(
                f"expected at least {parsed.min_alerts} Elasticsearch alert(s)"
            )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1

    print("[OK] Phase 4 streaming outputs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
