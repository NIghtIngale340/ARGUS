#!/usr/bin/env python3
"""
ARGUS — Kafka Topic Provisioning

Creates ARGUS topics with explicit partition counts and retention policies.
Idempotent: existing topics are skipped without error.

Requires: confluent-kafka (poetry add confluent-kafka)
"""

import sys

try:
    from src.config import settings
except Exception as e:
    print(f"[FATAL] Configuration load failed: {e}")
    sys.exit(1)


TOPIC_SPECS = [
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


def create_topics():
    """Create all topics defined in TOPIC_SPECS via the Kafka AdminClient."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})

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
            except Exception as e:
                if "TOPIC_ALREADY_EXISTS" in str(e):
                    print(f"  [SKIP] Topic already exists: {topic_name}")
                else:
                    print(f"  [FAIL] {topic_name}: {e}")
                    raise
    except Exception as e:
        print(f"[FATAL] Failed to create topics: {e}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  ARGUS — Kafka Topic Provisioning")
    print("=" * 60)
    print()

    create_topics()
    print("\n  [DONE] All topics provisioned successfully.")


if __name__ == "__main__":
    main()
