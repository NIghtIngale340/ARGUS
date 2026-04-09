#!/usr/bin/env python3
"""
ARGUS — Pre-flight Service Health Check

Validates connectivity to Elasticsearch, Kafka, Redis, and MLflow
before starting any ARGUS service. Runs all checks regardless of
individual failures so you see every problem in a single pass.

Exit codes: 0 = all healthy, 1 = one or more failed.
"""

import sys
from pathlib import Path

# Allow imports like `from src...` when executed as `python scripts/health_check.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import settings
except Exception as e:
    print(f"[FATAL] Failed to load configuration: {e}")
    sys.exit(1)


def check_elasticsearch() -> bool:
    """Check Elasticsearch cluster health (green/yellow = OK, red = FAIL)."""
    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(settings.es_url, verify_certs=False)
        health = es.cluster.health()
        status = health.get("status")
        if status in ("green", "yellow"):
            print(f"  [OK] Elasticsearch: cluster status: {status}")
            return True
        print(f"  [FAIL] Elasticsearch: cluster status: {status}")
        return False
    except Exception as e:
        print(f"  [FAIL] Elasticsearch: {e}")
        return False


def check_kafka() -> bool:
    """TCP socket probe against the Kafka bootstrap server."""
    import socket

    try:
        host, port_str = settings.kafka_bootstrap.split(":")
        port = int(port_str)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        sock.close()
        print(f"  [OK] Kafka: reachable at {host}:{port}")
        return True
    except Exception as e:
        print(f"  [FAIL] Kafka: {e}")
        return False


def check_redis() -> bool:
    """Redis PING/PONG liveness check."""
    try:
        import redis as redis_lib

        r = redis_lib.Redis.from_url(settings.redis_url)
        if r.ping():
            print("  [OK] Redis: responded to PING")
            return True
        return False
    except Exception as e:
        print(f"  [FAIL] Redis: {e}")
        return False


def check_mlflow() -> bool:
    """HTTP GET to MLflow /health endpoint."""
    import urllib.request
    import urllib.error

    try:
        url = f"{settings.mlflow_uri}/health"
        response = urllib.request.urlopen(url, timeout=5)
        if response.status == 200:
            print("  [OK] MLflow: responded to health check")
            return True
        return False
    except Exception as e:
        print(f"  [FAIL] MLflow: {e}")
        return False


def main():
    """Run all health checks and exit with appropriate code."""
    print("=" * 60)
    print("  ARGUS — Service Health Check")
    print("=" * 60)
    print()

    checks = {
        "Elasticsearch": check_elasticsearch,
        "Kafka": check_kafka,
        "Redis": check_redis,
        "MLflow": check_mlflow,
    }

    results = {}
    for name, check_func in checks.items():
        results[name] = check_func()

    passed = sum(results.values())
    total = len(results)
    print(f"\nResult: {passed}/{total} services healthy")

    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
