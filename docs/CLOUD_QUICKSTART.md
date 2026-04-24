# ARGUS Cloud Quickstart (Colab + Kaggle, 58-Day Resumable)

This is the canonical cloud flow for running LANL days `1-58` on free-tier notebook sessions with resume checkpoints.

## Canonical Run Order

1. Build session shards first (`day_01.parquet` ... `day_58.parquet`).
2. Build vocab/tokenized outputs second (train, val, test).
3. Persist outputs after each successful batch.

## Common Prerequisites

- You already have the full `auth.txt` dataset on each platform.
- Python 3 notebook runtime.
- ARGUS repo is cloneable: `https://github.com/NIghtIngale340/ARGUS`.

Use these variables in both platforms:

- `AUTH_PATH`: absolute path to `auth.txt`.
- `OUTPUT_ROOT`: persistent output root.
- `START_DAY=1`, `END_DAY=58`, `RESUME=true`.
- `DAY_BATCH_SIZE`: contiguous days per build command (larger is faster, smaller is safer).
- `BUCKET_COUNT`: memory-safety control for sessionization (`256` recommended).
- `REFRESH_GIT_CLONE=true`: default notebook behavior for Git-based runs so stale cloud repo copies are replaced before sessionization.

`build_sessions.py` now uses disk buckets per day to lower peak RAM usage. This is important for Kaggle/Colab full-data runs.
Both resumable notebooks validate the resolved `scripts/build_sessions.py --help` output before smoke or batch runs and fail fast if `--bucket-count` is missing.

Outputs are created under `OUTPUT_ROOT/data`:

- `data/sessions/day_XX.parquet`
- `data/vocab.json`
- `data/tokenized/sessions_train.pt`
- `data/tokenized/sessions_val.pt`
- `data/tokenized/sessions_test.pt`

## Option 1: Colab (Drive-backed persistence)

- Recommended persistence: `/content/drive/MyDrive/argus_outputs`
- Default repo clone source: `https://github.com/NIghtIngale340/ARGUS`
- Dataset source: `kagglehub.dataset_download("poornimakodithuwakku/lanl-dataset")`
- Notebook to run: `notebooks/colab_58day_resumable.ipynb`

`kagglehub.dataset_download(...)` downloads the Kaggle dataset into Colab local storage and returns the downloaded folder path.  
The notebook then searches that folder for `auth.txt` and sets `AUTH_PATH` automatically.
When `USE_GIT_CLONE=True`, the notebook refreshes `/content/ARGUS` by default before cloning so a stale repo copy does not bypass the newer CLI contract.

If running manually, use this command pattern per batch:

```bash
python scripts/build_sessions.py --input "$AUTH_PATH" --start-day "$BATCH_START" --end-day "$BATCH_END" --output-dir "$OUTPUT_ROOT/data/sessions" --parser-state "$OUTPUT_ROOT/data/drain3_state.bin" --bucket-count 256
```

Then tokenization by split (run from `OUTPUT_ROOT` so `data/...` paths resolve correctly):

```bash
python scripts/build_vocab_and_tokenize.py --sessions-glob "data/sessions/day_*.parquet" --split train --vocab-out data/vocab.json --tokenized-out data/tokenized/sessions_train.pt
python scripts/build_vocab_and_tokenize.py --sessions-glob "data/sessions/day_*.parquet" --split val --vocab-in data/vocab.json --tokenized-out data/tokenized/sessions_val.pt
python scripts/build_vocab_and_tokenize.py --sessions-glob "data/sessions/day_*.parquet" --split test --vocab-in data/vocab.json --tokenized-out data/tokenized/sessions_test.pt
```

## Option 2: Kaggle (`/kaggle/working` persistence)

- Recommended persistence: `/kaggle/working/argus_outputs`
- Default repo clone source: `https://github.com/NIghtIngale340/ARGUS`
- Dataset path behavior: auto-detect first `auth.txt` under `/kaggle/input/**`
- Notebook to run: `notebooks/kaggle_58day_resumable.ipynb`

When `USE_GIT_CLONE=True`, the notebook refreshes `/kaggle/working/ARGUS` by default before cloning so resumable outputs remain under `OUTPUT_ROOT` while the repo copy can be safely replaced.

After each successful day batch, save a Kaggle version:

1. Click `Save Version` in the top-right.
2. Choose a clear message, example: `processed day_01 to day_05`.
3. Continue from the next missing day in the next run.

## Resume Behavior

- Resume checks for existing non-empty shard files.
- If `data/sessions/day_17.parquet` exists and is non-empty, day 17 is skipped.
- Restarting with the same settings continues from missing shards.

## Validation Sequence

1. Smoke test on sample (`auth_sample_200k.txt`) for day 1.
2. Full file validation for day 1 and day 2.
3. Resume test: stop mid-run, restart, verify days already completed are skipped.
4. Final check: confirm all 58 day shards exist before tokenization.

## Completion Criteria

- `day_01.parquet` through `day_58.parquet` exist and are non-zero.
- `data/vocab.json` exists and is non-zero.
- `sessions_train.pt`, `sessions_val.pt`, `sessions_test.pt` exist and are non-zero.
