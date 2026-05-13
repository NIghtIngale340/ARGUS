#!/usr/bin/env python3
"""Create ARGUS Kafka topics with explicit retention policies.

Idempotent: existing topics are skipped without error.
Requires: confluent-kafka.
"""

from __future__ import annotations

import os
import sys


DEFAULT_KAFKA_BOOTSTRAP = "localhost:29092"


TOPIC_SPECS = [
    {
        "name": "argus.raw-logs",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "259200000",  # 3 days
        },
    },
    {
        "name": "argus.detections",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",  # 7 days
        },
    },
    {
        "name": "argus.dead-letter",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "1209600000",  # 14 days
        },
    },
    {
        "name": "logs.raw",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "259200000",  # 3 days
        },
    },
    {
        "name": "logs.parsed",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "259200000",  # 3 days
        },
    },
    {
        "name": "logs.anomalies",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",  # 7 days
        },
    },
    {
        "name": "logs.alerts",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",  # 7 days
        },
    },
]


def create_topics(bootstrap: str | None = None) -> None:
    """Create all topics defined in TOPIC_SPECS via the Kafka AdminClient."""
    try:
        from confluent_kafka.admin import AdminClient, NewTopic
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install confluent-kafka to create topics "
            "(for example: pip install confluent-kafka)."
        ) from exc

    admin = AdminClient(
        {
            "bootstrap.servers": bootstrap
            or os.getenv("KAFKA_BOOTSTRAP", DEFAULT_KAFKA_BOOTSTRAP)
        }
    )

    new_topics = [
        NewTopic(
            topic=spec["name"],
            num_partitions=spec["partitions"],
            replication_factor=spec["replication_factor"],
            config=spec.get("config", {}),
        )
        for spec in TOPIC_SPECS
    ]

    try:
        futures = admin.create_topics(new_topics)
        for topic_name, future in futures.items():
            try:
                future.result()
                print(f"  [OK] Created topic: {topic_name}")
            except Exception as exc:
                if "TOPIC_ALREADY_EXISTS" in str(exc):
                    print(f"  [SKIP] Topic already exists: {topic_name}")
                else:
                    print(f"  [FAIL] {topic_name}: {exc}")
                    raise
    except Exception as exc:
        print(f"[FATAL] Failed to create topics: {exc}")
        sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("  ARGUS Kafka Topic Provisioning")
    print("=" * 60)
    print()

    create_topics()
    print("\n  [DONE] All topics provisioned successfully.")


if __name__ == "__main__":
    main()
