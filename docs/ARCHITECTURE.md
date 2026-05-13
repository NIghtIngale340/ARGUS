# ARGUS Architecture

ARGUS is organized around the `src/` package with a single canonical implementation path. Legacy compatibility directories have been removed.

## Source Modules

| Module | Purpose |
|---|---|
| `src/parsing` | Drain3 log parsing, session building, vocabulary construction, tokenization |
| `src/training` | Manifest/chunk datasets, MLM pretraining loop, binary attack-classifier fine-tuning |
| `src/models` | ARGUS-BERT config dataclass, MLM wrapper, binary classifier head |
| `src/inference` | Phase 3 bundle loading, anomaly scoring, alert engine, UEBA risk stores (in-memory + Redis), Elasticsearch alert store, Kafka session processing |
| `src/api` | FastAPI runtime, security middleware (auth, CORS, rate limiting, trusted hosts), SOC dashboard |

## Runtime Flow

```text
LANL auth logs / session parquets
  → scripts/build_sessions.py        (parsing + Drain3 + sessionization)
  → scripts/build_vocab_and_tokenize.py  (vocab + manifest/chunk tokenization)
  → ARGUS-BERT MLM pretraining       (Kaggle/Colab notebooks)
  → Phase 3 binary classifier        (fine-tuning notebook)
  → Phase3DetectionService            (bundle loading + inference)
  → FastAPI / CLI / Kafka consumer    (serving layer)
  → Redis UEBA + Elasticsearch alerts (state + persistence)
  → /dashboard                        (read-only SOC view)
```

## Streaming Flow

```text
scripts/replay_sessions_to_kafka.py
  → Kafka argus.raw-logs topic
  → scripts/run_phase4_streaming_consumer.py
  → StreamingSessionProcessor + SessionTracker (Redis-backed)
  → Phase3DetectionService.score_items()
  → Kafka argus.detections topic
  → Elasticsearch argus-alerts-* daily indices
  → FastAPI /phase3/alerts + /dashboard
```

## Detection Semantics

The deployable model is a binary attack-vs-normal classifier. Runtime output rows include:

| Field | Description |
|---|---|
| `model_task` | Always `binary_attack_detection` in this release |
| `attack_probability` | Softmax probability for the attack class |
| `prediction` | `attack` or `normal` based on calibrated threshold |
| `threshold_source` | `bundle`, `env_override`, `cli_override`, or `request_override` |
| `technique_source` | `none` or `fallback` — this release has no trained MITRE classifier |
| `fallback_technique_id` | Static ID provided by the caller; not model output |
| `composite_severity` | Weighted combination of anomaly, confidence, technique severity, and user risk |
| `alert_class` | `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` based on composite severity thresholds |

## Alert Engine

Composite severity is a weighted sum of four signals:

- **Anomaly score** (0.35): normalized MLM reconstruction loss
- **Classification confidence** (0.25): attack probability from the classifier
- **Technique severity** (0.20): static lookup from `configs/technique_severity.json`
- **User risk** (0.20): EWMA risk score from Redis UEBA

## Infrastructure

Docker Compose provides Elasticsearch, Kibana, Kafka (with Zookeeper), Redis, MLflow, the API container, and the consumer container.

This is a **local demo stack**. Production hardening (Elasticsearch security, Kafka TLS/SASL, Redis auth, proper secrets management) is not applied.

Kibana is included as an optional Elasticsearch browser for debugging. No ARGUS Kibana dashboards are shipped.
