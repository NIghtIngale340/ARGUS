# ARGUS Current Engineering Handoff

Last updated: 2026-05-10  
Workspace: `C:\Users\Mark Christian Anub\Desktop\newproject\argus-log-intelligence-platform`

This document is the current continuity handoff for ARGUS, a transformer-based cybersecurity log-intelligence platform for LANL-style authentication logs. It is written for a new AI or engineer with no prior context.

## 1. Current Project Goal

ARGUS aims to turn large authentication-log streams into model-scored cybersecurity detections:

```text
raw auth logs
-> Drain3 parsing/enrichment
-> sessionization
-> tokenization
-> ARGUS-BERT MLM pretraining
-> anomaly scoring
-> supervised attack classification
-> calibrated thresholds
-> alert severity + UEBA risk
-> Kafka/Redis/API/Elasticsearch/MLflow deployment
-> dashboard/SOC workflows
```

The current practical goal is **not more model experimentation**. The current goal is to make the already-trained binary Phase 3 detector operational in a Phase 4 streaming foundation.

## 2. Current Phase and Truth Boundary

Current phase: **Phase 4 streaming/production foundation**.

Important nuance:

```text
Operational binary Phase 3 detector: mostly complete and live locally
Full mentor-guide MITRE multi-technique Phase 3: not complete
```

The project has a deployable binary attack-vs-normal classifier, calibrated threshold, model bundle, FastAPI endpoint, Redis UEBA state, Elasticsearch alert persistence, MLflow registration, Kafka replay/consumer utilities, and Docker Compose runtime. The true MITRE multi-technique classifier is paused because the current LANL parquet shards do not contain reliable MITRE technique labels.

## 3. Current Live Local Runtime State

As of this handoff, the local Docker stack was verified healthy:

```text
api             healthy, port 8000
consumer        healthy, direct Phase 4 Kafka consumer
kafka           healthy, ports 9092 in Docker and 29092 on host
redis           healthy, port 6379
elasticsearch   healthy, port 9200
```

Current `.env` important values:

```env
ES_URL=http://localhost:9200
KAFKA_BOOTSTRAP=localhost:29092
REDIS_URL=redis://localhost:6379/0
MLFLOW_URI=http://localhost:5000
ARGUS_PHASE3_BUNDLE_DIR=data/argus_phase3_model_bundle
ARGUS_USE_REDIS_UEBA=true
ARGUS_USE_ELASTICSEARCH_ALERTS=true
ARGUS_PHASE3_THRESHOLD=
ARGUS_STREAM_SESSION_MAX_TOKENS=
```

Blank `ARGUS_PHASE3_THRESHOLD` means the runtime uses the calibrated bundle threshold, currently `0.9990`. Blank `ARGUS_STREAM_SESSION_MAX_TOKENS` means the streaming tracker defaults to `max_seq_len - 2`, currently `14` event tokens for `max_seq_len=16`.

Recent real-threshold replay:

```text
input: data/sessions/day_01.parquet
published: 11,758 events to argus.raw-logs
latest detection sample threshold: 0.999
latest predictions: normal
latest alerts: 0
stream_token_count: 14
```

This is expected: normal replay data usually should not generate attack alerts at threshold `0.9990`.

## 4. Technical Stack

Core:

```text
Python 3.11 target, local host currently also uses Python 3.10 for some helper commands
PyTorch
Hugging Face transformers
PyArrow / Parquet
Pandas / NumPy
Drain3
FastAPI / Uvicorn
Redis
Kafka / Confluent Kafka Python client
Elasticsearch / Kibana
MLflow
Docker Compose
pytest
Kaggle for heavy data/model execution
Windows local workspace
```

Docker note: Docker was reset to free disk space, which wiped images/containers/volumes. The stack was rebuilt afterward. Kafka topics must be recreated after resets. The direct consumer now auto-creates/waits for its required topics to avoid crashing on missing topics.

## 5. Repository Structure

Important folders:

```text
src/parsing/
  raw parsing, Drain3 enrichment, vocab/tokenizer, session builder

src/training/
  tokenized manifest dataset, MLM collator, pretraining, fine-tuning, MITRE dataset scaffolding

src/models/
  ARGUS-BERT config, MLM model wrapper, classifier head

src/inference/
  anomaly scorer, Phase3DetectionService, alert engine, UEBA stores,
  Redis session tracker, Kafka/streaming processor, alert persistence

src/api/
  FastAPI app exposing health, detection, and UEBA risk endpoints

scripts/
  build/session/tokenize/score/evaluate/package/register/replay/verify utilities

notebooks/
  Kaggle and Colab execution notebooks

docker/
  API and consumer Dockerfiles

data/
  local bundle, local LANL session shards, local logs/artifacts
```

New/important Phase 4 files:

```text
scripts/replay_sessions_to_kafka.py
scripts/run_phase4_streaming_consumer.py
scripts/verify_phase4_streaming.py
src/inference/session_tracker.py
src/inference/kafka_consumer.py
src/inference/alert_store.py
```

## 6. Current Architecture

The central deployment object is:

```python
Phase3DetectionService
```

It loads:

```text
best_classifier.pt
calibrated_thresholds.json
vocab.json
model_metadata.json
```

and owns:

```text
LogTokenizer
ARGUSClassifier
AlertEngine
optional RedisUEBARiskStore
optional ElasticsearchAlertStore
```

Why this matters: all runtime paths should call `Phase3DetectionService` instead of duplicating bundle loading, tokenization, thresholding, or alert logic. CLI, API, and streaming consumer are intended to share the same scoring semantics.

Current Phase 4 streaming path:

```text
parsed/tokenized replay event
-> Kafka topic argus.raw-logs
-> scripts.run_phase4_streaming_consumer
-> StreamingSessionProcessor
-> SessionTracker in Redis
-> completed token sequence
-> Phase3DetectionService.score_items()
-> detection row
-> Kafka topic argus.detections
-> Elasticsearch alert persistence if alert_generated
-> Redis UEBA risk update
```

The Docker consumer does **not** use the Faust CLI anymore. Faust was attempted but had dependency/runtime issues. The working container command is:

```text
/usr/local/bin/python -m scripts.run_phase4_streaming_consumer
```

## 7. Current Pipeline Flow

Working binary flow:

```text
auth.txt
-> scripts/build_sessions.py
-> data/sessions/day_XX.parquet
-> scripts/build_vocab_and_tokenize.py
-> data/tokenized/sessions_* manifests + sessions_*_chunks/
-> src.training.pretrain
-> anomaly scoring
-> binary attack classifier fine-tuning
-> scripts/evaluate_attack_classifier.py
-> calibrated_thresholds.json
-> scripts/package_phase3_model_bundle.py
-> data/argus_phase3_model_bundle/
-> scripts/run_detection.py or FastAPI /phase3/detect
-> AlertEngine + Redis UEBA risk
-> Elasticsearch alert persistence
-> MLflow registration
```

Working Phase 4 local replay flow:

```text
data/sessions/day_01.parquet
-> scripts/replay_sessions_to_kafka.py
-> Kafka argus.raw-logs
-> scripts/run_phase4_streaming_consumer.py in consumer container
-> Redis SessionTracker
-> Phase3DetectionService
-> Kafka argus.detections
-> Elasticsearch argus-alerts-* when alert_generated=true
```

## 8. Preprocessing Logic

`scripts/build_sessions.py` parses LANL authentication logs. It supports 9-column and 7-column LANL variants and extracts fields such as:

```text
time
user
host
auth_type
logon_type
success
auth_orientation
event_id
template_id
```

Drain3 enrichment adds:

```text
template_id
event_id
template_params
```

Known weakness: token construction compresses event context heavily. It currently uses only event ID, auth type, and logon type. It ignores many fields that likely matter for attack behavior, such as source/destination directionality, temporal cadence, user/host role, and rare host/user transitions.

## 9. Sessionization

Session defaults:

```python
window_mins = 30
stride_mins = 15
min_events = 3
max_tokens = 512
timestamp_unit = "seconds"
```

Why day sharding exists:

```text
LANL scale is too large for full in-memory grouping.
Day shards bound memory, support resumability, and make Kaggle recovery feasible.
```

Expected logical session fields:

```text
session_id, user_id, host_id, start_ts, end_ts, events
```

Local verified session shards exist under:

```text
data/sessions/day_01.parquet ... day_58.parquet
```

Some local shard rows are flattened event rows rather than nested session objects. Replay handles this by converting parquet rows/events into stream events with `user_id`, `host_id`, `session_id`, and `token_id`.

## 10. Tokenization

Event token rule:

```python
def build_event_token(event):
    event_id = str(event.get("event_id", "NA"))
    auth_type = str(event.get("auth_type", "NA"))
    logon_type = str(event.get("logon_type", "NA"))
    return f"{event_id}_{auth_type}_{logon_type}"
```

Special token IDs:

```text
[CLS]  = 0
[SEP]  = 1
[MASK] = 2
[PAD]  = 3
[UNK]  = 4
```

Current verified model/tokenizer config:

```text
vocab_size = 1233
max_seq_len = 16
pad_token_id = 3
mask_token_id = 2
vocab path = data/argus_phase3_model_bundle/vocab.json locally
```

Why `max_seq_len=16`: full LANL-scale tokenization/training with `512` was infeasible on Kaggle due to disk/RAM. `16` made full training possible with compact dtypes but loses long-range behavior.

Tokenized artifact contract:

```text
sessions_train.pt
sessions_val.pt
sessions_test.pt
```

These are manifests, not monolithic tensor files. Real tensors live under:

```text
sessions_train_chunks/chunk_*.pt
sessions_val_chunks/chunk_*.pt
sessions_test_chunks/chunk_*.pt
```

Important constants:

```python
TOKENIZED_CHUNK_MANIFEST_FORMAT = "tokenized_session_chunk_manifest_v1"
TOKENIZED_CHUNK_FORMAT = "tokenized_session_chunk_v1"
```

Chunk payload:

```python
{
    "format": "tokenized_session_chunk_v1",
    "session_ids": [...],
    "input_ids": Tensor[N, max_len],
    "attention_mask": Tensor[N, max_len],
}
```

Kaggle compact dtypes:

```text
input_ids: int16
attention_mask: bool
```

Important local issue: `data/tokenized/sessions_test.pt` alone is not enough for manifest replay. The matching `data/tokenized/sessions_test_chunks/` directory must also exist. Without the chunk directory, replay from manifest fails by design with a clear error. Parquet replay works locally.

## 11. MLM Masking Logic

`src/training/mlm_collator.py` implements BERT-style masking:

```text
mask_probability = 0.15
mask_token_probability = 0.8
random_token_probability = 0.1
unchanged_probability = 0.1
ignore_index = -100
```

Rules:

```text
never mask [CLS], [SEP], [PAD]
mask 15% of real event tokens
80% replace selected tokens with [MASK]
10% replace selected tokens with random token
10% leave selected tokens unchanged
labels for unmasked positions = -100
```

Robustness fix: for tiny batches where random sampling masks nothing, force at least one maskable token target to avoid `nan` loss.

## 12. Model Configuration

Canonical ARGUS-BERT config:

```python
vocab_size = 1233
max_seq_len = 16
hidden_size = 256
num_hidden_layers = 6
num_attention_heads = 8
intermediate_size = 1024
dropout = 0.1
pad_token_id = 3
mask_token_id = 2
```

Use:

```text
src/models/config.py
configs/argus_bert_phase2.yaml
data/argus_phase3_model_bundle/model_metadata.json
```

Do not trust:

```text
configs/model.yaml
```

It is stale and has older generic values.

## 13. Models and Checkpoints

MLM model:

```text
src/models/argus_bert.py
ArgusBertForMaskedLM wrapping Hugging Face BertForMaskedLM
```

Classifier:

```text
src/models/attack_classifier.py
input_ids + attention_mask
-> BertModel
-> CLS hidden state
-> dropout
-> Linear(hidden_size, num_classes)
-> logits
```

The classifier supports `num_classes`, so binary attack-vs-normal and future MITRE multi-class are structurally possible.

Best known MLM checkpoint:

```text
checkpoint_step_003501.pt
val_loss = 0.4215 over 500 validation batches
```

Classifier bundle path:

```text
data/argus_phase3_model_bundle/
```

Bundle contents:

```text
best_classifier.pt
calibrated_thresholds.json
vocab.json
evaluation_report.json
classifier_scores.csv
finetune_history.json
checkpoint_step_003501.pt
model_metadata.json
```

## 14. Phase 2 / Anomaly Scoring Status

Phase 2 pretraining completed enough to produce a usable base checkpoint, but the broader Phase 2/2.5 evaluation showed that MLM anomaly score alone was not strong enough for operational attack detection.

Phase 2.5 anomaly scoring evidence:

```text
Attack sessions: 212
Normal sessions: 500,000
ROC-AUC: 0.865646
Cohen's d: 1.4412
```

Weakness: MLM anomaly score upper-tail thresholds were not enough to catch attacks reliably, motivating supervised binary classification.

## 15. Phase 3 Binary Classifier Status

Fine-tune dataset:

```text
10,000 normal sessions
212 attack sessions
```

Fine-tune result:

```text
best epoch: 19
val F1: 0.4556
precision: 0.2971
recall: 0.9762
TP=41 FP=97 FN=1 TN=1903
```

Same-day eval:

```text
ROC-AUC = 0.998133
PR-AUC  = 0.964293
```

Operational threshold:

```text
threshold = 0.9990
precision = 0.9108
recall = 0.9151
FPR = 0.0019
```

Best-F1 threshold:

```text
threshold = 0.9995
F1 = 0.9268
precision = 0.9596
recall = 0.8962
FPR = 0.0008
```

Brutal caveat: these metrics are strong smoke/evaluation evidence, not scientific generalization. The attack set is tiny (`212` sessions), likely training-adjacent, and one previous eval path printed `Attack windows loaded: 0`, indicating label-window filtering was not fully trustworthy in that path.

## 16. Detection Output Schema

Current detection rows include:

```text
session_id
user_id
host_id
attack_probability
prediction
technique_id
technique_probability
threshold
alert_generated
alert_class
composite_severity
stream_token_count for streaming rows
```

For the current binary model, `technique_id` is a fallback static value, usually `T1078`, when attack is predicted. It is not a true MITRE technique classifier output yet.

## 17. Alert Engine and UEBA

`src/inference/alert_engine.py` includes:

```text
Alert
ScoredSession
UEBARiskStore
RedisUEBARiskStore
AlertEngine
```

Severity formula:

```text
severity =
  0.35 * anomaly_norm
+ 0.25 * classification_confidence
+ 0.20 * technique_severity
+ 0.20 * user_risk
```

Severity cutoffs:

```python
CRITICAL = 0.80
HIGH     = 0.60
MEDIUM   = 0.40
LOW      = 0.0
```

Technique severity examples:

```python
T1003 = 0.95
T1078 = 0.80
T1021 = 0.75
T1110 = 0.85
```

Redis keys:

```text
argus:ueba:risk
argus:ueba:last_update
argus:ueba:risk_history:{user_id}
```

API UEBA endpoints:

```text
GET    /phase3/ueba/risks
GET    /phase3/ueba/risks/{user_id}
GET    /phase3/ueba/risks/{user_id}/timeline
DELETE /phase3/ueba/risks
```

## 18. API Surface

`src/api/main.py` exposes:

```text
GET  /health
POST /phase3/detect
GET  /phase3/ueba/risks
GET  /phase3/ueba/risks/{user_id}
GET  /phase3/ueba/risks/{user_id}/timeline
DELETE /phase3/ueba/risks
```

The API loads the bundle from `ARGUS_PHASE3_BUNDLE_DIR`. It supports optional `ARGUS_PHASE3_THRESHOLD` override for smoke tests, but real mode leaves this blank so the bundle threshold is used.

## 19. Kafka / Streaming Runtime Details

Kafka topics:

```text
argus.raw-logs
argus.detections
logs.raw
logs.parsed
logs.anomalies
logs.alerts
```

Host scripts use:

```text
localhost:29092
```

Docker containers use:

```text
kafka:9092
```

Why two listeners: Windows host cannot resolve Docker-internal `kafka:9092`, while containers should not use `localhost:29092`. Docker Compose now advertises both.

Direct consumer:

```text
scripts/run_phase4_streaming_consumer.py
```

Responsibilities:

```text
ensure argus.raw-logs and argus.detections exist
consume JSON from argus.raw-logs
call StreamingSessionProcessor
produce JSON detections to argus.detections
allow Redis/Elasticsearch side effects through Phase3DetectionService
```

It now creates/waits for topics to avoid crashing after Docker resets.

Replay utility:

```text
scripts/replay_sessions_to_kafka.py
```

Supports:

```text
--sessions-parquet
--manifest
--vocab
--limit-sessions
--limit-events
--dry-run
```

Verification utility:

```text
scripts/verify_phase4_streaming.py
```

Checks:

```text
Kafka detections in argus.detections
Elasticsearch alert count in argus-alerts-*
```

Important issue: `--from-beginning` may read old smoke-test detections. The current next improvement is a "latest only" verifier that reads from recent offsets or filters by replay run ID.

## 20. Scripts Currently Used

Core data:

```text
scripts/build_sessions.py
scripts/build_vocab_and_tokenize.py
```

Training/eval:

```text
python -m src.training.pretrain
python -m src.training.finetune
scripts/train_mlm_smoke.py
scripts/score_sessions.py
scripts/calibrate_thresholds.py
scripts/compare_attack_scores.py
scripts/evaluate_attack_classifier.py
scripts/evaluate_mitre_classifier.py
```

Deployment/runtime:

```text
scripts/package_phase3_model_bundle.py
scripts/run_detection.py
scripts/register_phase3_model.py
scripts/replay_sessions_to_kafka.py
scripts/run_phase4_streaming_consumer.py
scripts/verify_phase4_streaming.py
```

Infra:

```text
scripts/create_kafka_topics.py
scripts/create_es_index.py
scripts/health_check.py
```

MITRE tools, currently paused:

```text
scripts/build_mitre_labels.py
scripts/validate_mitre_labels.py
src/training/mitre_dataset.py
scripts/evaluate_mitre_classifier.py
notebooks/kaggle_phase3_mitre_label_builder.ipynb
```

## 21. Kaggle Workflow

Heavy data/model execution happens in Kaggle. Local is used for packaging, inference, API, Docker, and streaming smoke tests.

Kaggle realities:

```text
Patched local repo is not automatically present in Kaggle.
If notebooks use REFRESH_REPO=True, GitHub must already contain the needed patch.
Use /kaggle/working for active writes.
Archive/download outputs before ending session.
Do not write huge live tokenization outputs directly to Google Drive.
```

Important tokenization resume behavior:

```text
Existing non-empty/readable day_*.parquet shards should be reused.
Readable shard validation should use pyarrow.parquet.ParquetFile(shard).
Manifest-plus-chunk tokenized outputs should be resumed/reused where possible.
Batch-level skip_rows is preferred over re-decoding skipped prefixes.
```

## 22. Optimization Goals

Current priorities:

```text
do not retrain unless necessary
preserve Kaggle artifacts
reuse verified 58-day tokenized/session data
keep max_seq_len=16 for feasibility
use calibrated threshold 0.9990, not 0.5
use bundle-driven inference
use Redis for shared UEBA state
use Elasticsearch for alert persistence
keep Kafka replay/consumer reliable after Docker resets
move Phase 4 streaming forward
```

Immediate optimization/fix target:

```text
Improve streaming verification so it reads latest detection messages only,
or tags each replay run with run_id and verifies only that run.
```

This is needed because `verify_phase4_streaming --from-beginning` can read old smoke-test detections with `threshold=0.0`, even when the current consumer is correctly using `threshold=0.999`.

## 23. Bottlenecks and Scalability Concerns

Major bottlenecks:

```text
100M+ sessions means no full in-memory workflow.
Tokenization is disk-heavy and slow.
max_seq_len=512 was infeasible on Kaggle.
max_seq_len=16 loses long-range behavior.
Current event token design is too compressed.
Attack set is tiny.
MITRE labels are unavailable in current LANL shards.
Docker consumer image is huge because PyTorch/ML dependencies are heavy.
Docker build cache can rapidly consume tens of GB.
Kafka topic/volume state disappears after Docker reset.
Elasticsearch alert counts can include old smoke-test alerts.
```

Scalability risks:

```text
Redis session state needs TTL/pressure monitoring for long streams.
Kafka consumer currently scores session batches one item at a time after flush.
Elasticsearch indexing is synchronous per alert through AlertStore.
No backpressure strategy yet for scoring slower than ingestion.
No production auth/rate limits on API.
No dashboard yet.
No production model governance beyond MLflow registration surface.
No strong held-out MITRE validation.
```

## 24. Known Bugs, Weaknesses, and Honest Risks

Brutally honest risks:

```text
Binary classifier is not a real MITRE ATT&CK classifier.
Same-day eval may be leakage-prone.
Attack windows loaded as 0 in one previous eval path.
Classifier probabilities are overconfident and thresholded near 1.0.
Thresholds near 0.999 are fragile.
Session IDs are missing in some parquet shards.
Current tokenization loses temporal and relational context.
Manifest-only tokenized files are useless without matching chunks.
Docker reset destroys Kafka topics and ES/Redis/Kafka volume state.
The local Docker image is heavy and can consume disk quickly.
Verification can confuse old and new Kafka messages unless offsets/run IDs are handled.
```

## 25. Important Commands

Create topics and ES templates:

```powershell
python scripts/create_kafka_topics.py
python scripts/create_es_index.py
```

Start core runtime:

```powershell
docker compose up -d redis kafka elasticsearch mlflow api consumer
```

Parquet replay:

```powershell
python -m scripts.replay_sessions_to_kafka `
  --sessions-parquet data/sessions/day_01.parquet `
  --vocab data/argus_phase3_model_bundle/vocab.json `
  --limit-sessions 1000
```

Verification:

```powershell
python -m scripts.verify_phase4_streaming --from-beginning --min-detections 1 --min-alerts 0
```

Smoke-test-only overrides:

```env
ARGUS_PHASE3_THRESHOLD=0.0
ARGUS_STREAM_SESSION_MAX_TOKENS=1
```

Real mode:

```env
ARGUS_PHASE3_THRESHOLD=
ARGUS_STREAM_SESSION_MAX_TOKENS=
```

## 26. Current Decisions Already Made

```text
Use Kaggle for large training/tokenization runs.
Use local Windows/Docker for Phase 4 runtime validation.
Use verified 58-day session/tokenized data where available.
Use manifest-plus-chunk tokenized format.
Use vocab_size=1233 and max_seq_len=16.
Use checkpoint_step_003501.pt as best MLM base.
Use binary classifier first.
Use operational threshold 0.9990.
Use Phase3DetectionService as the shared runtime abstraction.
Use Redis for shared UEBA risk/session state.
Use Elasticsearch for alert persistence.
Use MLflow registry for model governance.
Pause MITRE multi-class until real labels/dataset adapter exists.
Use direct Confluent Kafka consumer instead of Faust CLI for Docker reliability.
```

## 27. Future Plans

Near-term:

```text
1. Add latest-only or run_id-based streaming verifier.
2. Add replay run_id to Kafka events and propagate it into detections.
3. Add dashboard/Kibana saved objects or API alert query endpoints.
4. Make consumer batching/backpressure more production-like.
5. Add API auth/rate limiting before any exposed deployment.
6. Add deployment docs for Docker reset/rebuild/topic recreation.
```

Medium-term:

```text
1. Improve event token design beyond event_id_auth_type_logon_type.
2. Evaluate longer sequence lengths selectively, possibly not full 512.
3. Build real MITRE dataset adapter from external datasets.
4. Train/evaluate true multi-technique classifier.
5. Add held-out validation and leakage checks.
6. Add monitoring for Redis state, Kafka lag, ES indexing failures, and model latency.
```

Long-term vision:

```text
self-supervised log-language model
+ supervised attack classifier
+ anomaly scoring
+ UEBA risk over time
+ Kafka streaming ingestion
+ Redis online session state
+ Elasticsearch alert persistence
+ Kibana/SOC dashboard
+ MLflow model governance
+ eventually real MITRE multi-technique classification
```

## 28. Current Best Next Task

The best next implementation task is:

```text
Implement latest-only/run_id-aware streaming verification.
```

Why: the Phase 4 stack is live and processing, but verification can read stale smoke-test detections from Kafka/Elasticsearch. A robust verifier should tag replay events with a `replay_run_id`, propagate it through `StreamingSessionProcessor` and detection rows, store it in Elasticsearch, and verify only detections/alerts from that run. This will prevent false confidence and make repeated local/CI smoke tests deterministic.
