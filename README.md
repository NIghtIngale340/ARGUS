# ARGUS Log Intelligence Platform

A cybersecurity ML log-intelligence platform with LANL-scale parsing, sessionization, tokenization, ARGUS-BERT MLM pretraining lineage, binary attack-vs-normal detection, FastAPI serving, Redis UEBA hot state, Elasticsearch alert persistence, and Kafka replay/streaming foundations.

Built as a serious engineering side-project and cybersecurity ML research prototype.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                     LANL Authentication Logs                     │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │   Drain3 Log Parsing   │
              │  + Session Building    │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  Vocab + Tokenization  │
              │  (manifest/chunk .pt)  │
              └────────────┬───────────┘
                           ▼
         ┌─────────────────────────────────┐
         │   ARGUS-BERT MLM Pretraining    │
         │   (Kaggle/Colab notebooks)      │
         └─────────────────┬───────────────┘
                           ▼
         ┌─────────────────────────────────┐
         │  Phase 3 Binary Classifier      │
         │  attack-vs-normal fine-tuning   │
         └─────────────────┬───────────────┘
                           ▼
    ┌──────────────────────────────────────────────┐
    │          Phase3DetectionService               │
    │  calibrated threshold · composite severity    │
    └──────┬───────────────────┬───────────────────┘
           │                   │
           ▼                   ▼
    ┌──────────────┐    ┌──────────────────┐
    │  FastAPI /    │    │ Kafka Streaming  │
    │  CLI / Batch  │    │ Consumer         │
    └──────┬───────┘    └──────┬───────────┘
           │                   │
           ▼                   ▼
    ┌──────────────────────────────────────┐
    │  Redis UEBA    │  Elasticsearch     │
    │  (risk state)  │  (alert storage)   │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  SOC Dashboard   │
    │  /dashboard      │
    └──────────────────┘
```

## What Works

- **Data pipeline**: LANL-scale parsing, Drain3 enrichment, sessionization, vocabulary building, and manifest/chunk tokenized datasets across 58 days of authentication logs.
- **MLM pretraining**: ARGUS-BERT masked-language-model pretraining with resumable Kaggle/Colab notebooks.
- **Binary detection**: Phase 3 attack-vs-normal classifier with calibrated thresholding (ROC-AUC 0.998, PR-AUC 0.964).
- **API serving**: FastAPI with API-key auth, CORS allowlist, rate limiting, trusted hosts, request size limits, and `/health` + `/ready` probes.
- **UEBA risk**: Per-user Exponentially Weighted Moving Average risk tracking via Redis.
- **Alert persistence**: Elasticsearch daily-index alert storage with composite severity scoring and query APIs.
- **Streaming**: Kafka replay pipeline with run-ID filtering, dead-letter topic, and Redis session accumulation.
- **Dashboard**: Local read-only SOC dashboard at `/dashboard` with alert table, severity metrics, UEBA risk panel, and auto-refresh.

## Not Implemented

- Real MITRE ATT&CK multi-technique classification.
- Attention or attribution explainability.
- Contrastive learning.
- Production RBAC or analyst workflow actions.
- Production endpoint collection via Filebeat/Winlogbeat.
- Durable Elasticsearch `risk-profiles-*` UEBA history.

These are documented as future work, not shipped features.

## Repository Structure

```text
src/                    # Canonical implementation
  api/                  #   FastAPI, middleware, dashboard
  inference/            #   Detection service, alert engine, UEBA, Kafka consumer
  models/               #   ARGUS-BERT config, MLM model, attack classifier
  parsing/              #   Drain3 parser, session builder, tokenizer, vocab builder
  training/             #   Dataset loaders, MLM pretraining, fine-tuning
scripts/                # CLI entrypoints and operational tools
configs/                # Model configs, Drain3 settings, technique severity
docker/                 # Dockerfiles (API, consumer), Logstash config
docs/                   # Architecture, demo guide, model card, security notes
notebooks/              # Kaggle/Colab training notebooks
tests/                  # Unit tests (127 passing)
requirements/           # pip-fallback requirements
.github/                # CI workflow, Dependabot
```

`src/` is the single canonical code path. Legacy `backend/`, `ml/`, and `data_pipeline/` directories have been removed.

## Local Setup

Requires Python 3.11 and Docker Desktop.

```bash
git clone https://github.com/<your-username>/argus-log-intelligence-platform.git
cd argus-log-intelligence-platform
```

### Poetry (recommended)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip poetry==2.3.2
.\.venv\Scripts\poetry install --with dev
Copy-Item .env.example .env
```

### pip fallback

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements\dev.txt
Copy-Item .env.example .env
```

Edit `.env` and set `ARGUS_API_KEY` and `ARGUS_ADMIN_API_KEY` before exposing the API.

## Model Bundle

The Phase 3 model bundle is a release artifact, not tracked in Git. Place it at:

```text
data/argus_phase3_model_bundle/
```

Required files: `best_classifier.pt`, `calibrated_thresholds.json`, `vocab.json`, `model_metadata.json`.

For public GitHub use, publish the bundle as a GitHub Release asset, Hugging Face artifact, or Kaggle output. Then extract:

```powershell
New-Item -ItemType Directory -Force data\argus_phase3_model_bundle
Expand-Archive .\argus_phase3_model_bundle.zip -DestinationPath data\argus_phase3_model_bundle -Force
```

Bundle zip SHA256 (local artifact):

```text
E25F1A5EBC99337069A98AB6F539F3809B229E673D082027514E5C249F87D44C
```

## Demo

### Option A — Local development (recommended)

Uses Docker for infrastructure only. Faster iteration, no image rebuilds.

```powershell
# 1. Start infrastructure
docker compose --env-file .env up -d elasticsearch kibana zookeeper kafka redis mlflow

# 2. Initialize Elasticsearch index and Kafka topics
.\.venv\Scripts\python -m scripts.create_es_index
.\.venv\Scripts\python -m scripts.create_kafka_topics

# 3. Start the API server
.\.venv\Scripts\uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 4. In another terminal — start the streaming consumer
.\.venv\Scripts\python -m scripts.run_phase4_streaming_consumer --from-beginning

# 5. Replay sessions into Kafka
.\.venv\Scripts\python -m scripts.replay_sessions_to_kafka `
  --manifest data\tokenized\sessions_test.pt `
  --limit-sessions 25 `
  --replay-run-id demo-001

# 6. Verify detections and alerts
.\.venv\Scripts\python -m scripts.verify_phase4_streaming `
  --from-beginning `
  --replay-run-id demo-001 `
  --min-detections 1 `
  --min-alerts 1
```

### Option B — Full Docker Compose

Clean container deployment for demo recordings or reproducibility checks.

```powershell
docker compose --env-file .env down --remove-orphans
docker compose --env-file .env up -d elasticsearch kibana zookeeper kafka redis mlflow
.\.venv\Scripts\python -m scripts.create_es_index
.\.venv\Scripts\python -m scripts.create_kafka_topics
docker compose --env-file .env build api consumer
docker compose --env-file .env up -d api consumer
.\.venv\Scripts\python -m scripts.replay_sessions_to_kafka `
  --manifest data\tokenized\sessions_test.pt `
  --limit-sessions 25 `
  --replay-run-id demo-clean-001
.\.venv\Scripts\python -m scripts.verify_phase4_streaming `
  --from-beginning `
  --replay-run-id demo-clean-001 `
  --min-detections 1 `
  --min-alerts 1
```

### Parquet fallback

If tokenized chunks are not available, replay from session parquet instead. This validates Kafka, the consumer, Elasticsearch persistence, and API/dashboard access, but does not replace calibrated model evaluation.

```powershell
$env:ARGUS_PHASE3_THRESHOLD = "0.0"
docker compose --env-file .env up -d --force-recreate api consumer
Remove-Item Env:\ARGUS_PHASE3_THRESHOLD
.\.venv\Scripts\python -m scripts.replay_sessions_to_kafka `
  --sessions-parquet data\sessions\day_01.parquet `
  --limit-sessions 25 `
  --replay-run-id demo-parquet-001
.\.venv\Scripts\python -m scripts.verify_phase4_streaming `
  --from-beginning `
  --replay-run-id demo-parquet-001 `
  --min-detections 1 `
  --min-alerts 1
```

### Dashboard

Open `http://127.0.0.1:8000/dashboard`. Username: `argus`, password: your `ARGUS_API_KEY`.

## API Overview

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | None | Liveness probe |
| `/ready` | GET | None | Readiness probe with dependency checks |
| `/phase3/detect` | POST | API key | Score sessions for attack detection |
| `/phase3/alerts` | GET | API key | Query stored alerts with filters |
| `/phase3/alerts/{id}` | GET | API key | Get a single alert by ID |
| `/phase3/ueba/risks` | GET | API key | List all user risk scores |
| `/phase3/ueba/risks/{user_id}` | GET | API key | Get risk for a specific user |
| `/phase3/ueba/risks/{user_id}/timeline` | GET | API key | User risk timeline |
| `/phase3/ueba/risks` | DELETE | Admin key | Clear all UEBA risk state |
| `/dashboard` | GET | Basic auth | SOC alert dashboard |

Auth methods: `X-ARGUS-API-Key` header, `Authorization: Bearer` header, or HTTP Basic (dashboard).

## Tests and Checks

```powershell
poetry check                                         # pyproject.toml validation
.\.venv\Scripts\python -m pytest -q                  # 127 unit tests
docker compose config --quiet                        # Docker Compose validation
.\.venv\Scripts\python -m scripts.health_check       # service connectivity
```

## Docker Cleanup

If Docker Desktop keeps a large WSL disk after pruning:

```powershell
docker compose down --remove-orphans
docker builder prune -af
docker image prune -af
docker volume prune -f
wsl --shutdown
Optimize-VHD -Path "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx" -Mode Full
```

## Security

This stack is configured for **local demos only**. See [docs/SECURITY.md](docs/SECURITY.md) for details.

- All non-probe API routes require `ARGUS_API_KEY`.
- Destructive endpoints require `ARGUS_ADMIN_API_KEY`.
- Dashboard uses HTTP Basic auth.
- Elasticsearch, Kafka, and Redis run without TLS/auth for local convenience.
- `.env` is gitignored. Never commit real secrets.

## Limitations

- The binary classifier is trained and evaluated on LANL authentication log sessions. Same-day eval metrics are strong (ROC-AUC 0.998) but do not constitute broad held-out production proof.
- The attack set is small and likely training-adjacent. Real production validation would require leakage-resistant held-out data and broader attack coverage.
- Technique metadata in detection output uses static fallback IDs, not a trained MITRE classifier.
- UEBA risk scores are EWMA-based hot state in Redis. There is no durable historical risk profile storage.
- The Kafka consumer uses a polling loop with `confluent-kafka`, not the Faust streaming agent.
- The dashboard is a single-page read-only view. There are no analyst workflow actions.

## Future Work

- Multi-technique MITRE ATT&CK classification with a dedicated attack taxonomy head.
- Attention-based explainability for session-level detection decisions.
- Contrastive pretraining objectives for improved anomaly separation.
- Durable Elasticsearch-backed UEBA risk profile history.
- Production RBAC with role-based API access control.
- Endpoint log adapters (Filebeat/Winlogbeat integration).

## License

[MIT](LICENSE) — Mark Christian Anub
