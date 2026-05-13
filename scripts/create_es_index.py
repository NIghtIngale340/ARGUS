#!/usr/bin/env python3
import os
import sys

DEFAULT_ES_URL = "http://localhost:9200"


INDEX_TEMPLATES = {
    "logs-raw": {
        "index_patterns": ["logs-raw-*"],
        "template": {
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 0,
                "refresh_interval": "5s",
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "raw_message": {"type": "text"},
                    "source": {"type": "keyword"},
                    "log_level": {"type": "keyword"},
                    "hostname": {"type": "keyword"},
                    "source_ip": {"type": "ip"},
                },
            },
        },
    },

    "logs-parsed": {
        "index_patterns": ["logs-parsed-*"],
        "template": {
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 0,
                "refresh_interval": "5s",
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "raw_message": {"type": "text"},
                    "template_id": {"type": "keyword"},
                    "template": {"type": "text"},
                    "parsed_fields": {"type": "object", "dynamic": True},
                    "source": {"type": "keyword"},
                    "hostname": {"type": "keyword"},
                    "log_level": {"type": "keyword"},
                },
            },
        },
    },

    "alerts": {
        "index_patterns": ["alerts-*", "argus-alerts-*"],
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "alert_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "host_id": {"type": "keyword"},
                    "session_id": {"type": "keyword"},
                    "replay_run_id": {"type": "keyword"},
                    "anomaly_score": {"type": "double"},
                    "attack_probability": {"type": "double"},
                    "technique_probability": {"type": "double"},
                    "technique_id": {"type": "keyword"},
                    "technique_source": {"type": "keyword"},
                    "fallback_technique_id": {"type": "keyword"},
                    "threshold": {"type": "double"},
                    "threshold_source": {"type": "keyword"},
                    "model_task": {"type": "keyword"},
                    "alert_persistence_status": {"type": "keyword"},
                    "classification": {"type": "keyword"},
                    "classification_confidence": {"type": "double"},
                    "user_risk": {"type": "double"},
                    "composite_severity": {"type": "double"},
                    "alert_class": {"type": "keyword"},
                    "severity": {"type": "keyword"},
                    "is_anomaly": {"type": "boolean"},
                    "raw_message": {"type": "text"},
                    "template_id": {"type": "keyword"},
                    "model_version": {"type": "keyword"},
                },
            },
        },
    },
}


def create_index_templates():
    """Apply all index templates to Elasticsearch."""
    try:
        from elasticsearch import Elasticsearch

        es_url = os.getenv("ELASTICSEARCH_URL") or os.getenv("ES_URL") or DEFAULT_ES_URL
        es = Elasticsearch(es_url)

        count = 0
        for template_name, template_body in INDEX_TEMPLATES.items():
            es.indices.put_index_template(
                name=template_name,
                **template_body,
            )
            print(f"  [OK] Applied template: {template_name}")
            count += 1
        return count
    except Exception as e:
        print(f"[FATAL] Failed to create index templates: {e}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  ARGUS - Elasticsearch Index Template Provisioning")
    print("=" * 60)
    print()

    applied = create_index_templates()
    print(f"\n  [DONE] {applied} index templates applied successfully.")


if __name__ == "__main__":
    main()
