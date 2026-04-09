#!/usr/bin/env python3
import sys

try:
    from src.config import settings
except Exception as e:
    print(f"[FATAL] Configuration load failed: {e}")
    sys.exit(1)


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
        "index_patterns": ["alerts-*"],
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "anomaly_score": {"type": "double"},
                    "severity": {"type": "keyword"},
                    "is_anomaly": {"type": "boolean"},
                    "raw_message": {"type": "text"},
                    "template_id": {"type": "keyword"},
                    "mitre": {
                        "type": "object",
                        "properties": {
                            "tactic": {"type": "keyword"},
                            "technique": {"type": "keyword"},
                            "technique_id": {"type": "keyword"},
                        },
                    },
                    "model_version": {"type": "keyword"},
                },
            },
        },
    },

    "risk-profiles": {
        "index_patterns": ["risk-profiles-*"],
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "10s",
            },
            "mappings": {
                "properties": {
                    "entity_id": {"type": "keyword"},
                    "entity_type": {"type": "keyword"},
                    "risk_score": {"type": "double"},
                    "anomaly_count": {"type": "integer"},
                    "last_seen": {"type": "date"},
                    "updated_at": {"type": "date"},
                },
            },
        },
    },
}


def create_index_templates():
    """Apply all index templates to Elasticsearch."""
    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(settings.es_url)

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
    print("  ARGUS — Elasticsearch Index Template Provisioning")
    print("=" * 60)
    print()

    applied = create_index_templates()
    print(f"\n  [DONE] {applied} index templates applied successfully.")


if __name__ == "__main__":
    main()
