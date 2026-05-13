# ARGUS Demo Guide

This guide walks through a controlled local demo. It is not production deployment guidance.

## Prerequisites

- Python 3.11
- Docker Desktop (running)
- The Phase 3 model bundle extracted at `data/argus_phase3_model_bundle/`

## 1. Install Dependencies

```bash
git clone https://github.com/<your-username>/argus-log-intelligence-platform.git
cd argus-log-intelligence-platform
```

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip poetry==2.3.2
.\.venv\Scripts\poetry install --with dev
```

## 2. Configure Environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set `ARGUS_API_KEY` and `ARGUS_ADMIN_API_KEY`. Confirm `ARGUS_PHASE3_BUNDLE_DIR=data/argus_phase3_model_bundle`.

## 3. Prepare Artifacts

Place the Phase 3 model bundle at `data/argus_phase3_model_bundle/`. Required files:

- `best_classifier.pt`
- `calibrated_thresholds.json`
- `vocab.json`
- `model_metadata.json`

For the tokenized replay path, also place the test manifest and chunks:

```text
data/tokenized/sessions_test.pt
data/tokenized/sessions_test_chunks/chunk_*.pt
```

If tokenized chunks are not available, use the parquet fallback (see below).

## 4. Start Docker Services

```powershell
docker compose --env-file .env up -d elasticsearch kibana zookeeper kafka redis mlflow
```

Wait for services to be healthy. Kibana is optional and has no ARGUS dashboards — the supported dashboard is at `/dashboard`.

## 5. Initialize Elasticsearch and Kafka

```powershell
.\.venv\Scripts\python -m scripts.create_es_index
.\.venv\Scripts\python -m scripts.create_kafka_topics
```

## 6. Start the API

```powershell
.\.venv\Scripts\uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

## 7. Start the Streaming Consumer

In a separate terminal:

```powershell
.\.venv\Scripts\python -m scripts.run_phase4_streaming_consumer --from-beginning
```

## 8. Replay Sessions

**Tokenized path (recommended):**

```powershell
.\.venv\Scripts\python -m scripts.replay_sessions_to_kafka `
  --manifest data\tokenized\sessions_test.pt `
  --limit-sessions 25 `
  --replay-run-id demo-001
```

**Parquet fallback (plumbing smoke check only):**

```powershell
$env:ARGUS_PHASE3_THRESHOLD = "0.0"
.\.venv\Scripts\python -m scripts.replay_sessions_to_kafka `
  --sessions-parquet data\sessions\day_01.parquet `
  --limit-sessions 25 `
  --replay-run-id demo-parquet-001
Remove-Item Env:\ARGUS_PHASE3_THRESHOLD
```

A low `ARGUS_PHASE3_THRESHOLD` is acceptable only for plumbing checks. It will be reported as `threshold_source=env_override`.

## 9. Verify Outputs

```powershell
.\.venv\Scripts\python -m scripts.verify_phase4_streaming `
  --from-beginning `
  --replay-run-id demo-001 `
  --min-detections 1 `
  --min-alerts 1
```

## 10. View Dashboard

Open `http://127.0.0.1:8000/dashboard`.

- Username: `argus`
- Password: value of `ARGUS_API_KEY`

## 11. Run Tests

```powershell
.\.venv\Scripts\python -m pytest -q
docker compose config --quiet
```

## Script Inventory

| Script | Purpose |
|---|---|
| `build_sessions.py` | LANL log parsing, Drain3 enrichment, session parquet output |
| `build_vocab_and_tokenize.py` | Vocabulary building and manifest/chunk tokenization |
| `train_mlm.py` | ARGUS-BERT MLM pretraining entrypoint |
| `score_sessions.py` | Phase 2.5 MLM anomaly scoring |
| `calibrate_thresholds.py` | Anomaly score threshold calibration |
| `compare_attack_scores.py` | Attack-vs-normal score distribution analysis |
| `evaluate_attack_classifier.py` | Phase 3 classifier evaluation with metrics |
| `package_phase3_model_bundle.py` | Bundle packaging for deployment |
| `register_phase3_model.py` | Optional MLflow model registration |
| `run_detection.py` | Local CLI inference from Phase 3 bundle |
| `create_es_index.py` | Elasticsearch index setup |
| `create_kafka_topics.py` | Kafka topic creation |
| `replay_sessions_to_kafka.py` | Session replay into Kafka |
| `run_phase4_streaming_consumer.py` | Streaming detection consumer |
| `verify_phase4_streaming.py` | End-to-end streaming verification |
| `health_check.py` | Service connectivity check |
