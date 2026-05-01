import argparse
from collections import Counter
import re
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.parsing.log_tokenizer import LogTokenizer
from src.parsing.vocab_builder import VocabBuilder, build_event_token

SPLIT_DAY_RANGES = {
    "train": (1, 40),
    "val": (41, 50),
    "test": (51, 58),
    "all": (1, 58),
}

TORCH_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}

TORCH_DTYPE_MAX_VALUES = {
    "uint8": 255,
    "int16": 32_767,
    "int32": 2_147_483_647,
    "int64": 9_223_372_036_854_775_807,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build vocabulary JSON and tokenized PyTorch tensors from session parquet files."
    )
    parser.add_argument(
        "--sessions-glob",
        type=str,
        default="data/sessions/day_*.parquet",
        help="Glob pattern for input parquet session files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=tuple(SPLIT_DAY_RANGES.keys()),
        help="Time-based split to process: train=days 1-40, val=41-50, test=51-58",
    )
    parser.add_argument(
        "--vocab-out",
        type=str,
        default="data/vocab.json",
        help="Output path for vocabulary JSON when building a new vocabulary",
    )
    parser.add_argument(
        "--vocab-in",
        type=str,
        default=None,
        help=(
            "Existing vocabulary JSON to reuse for tokenization. "
            "Use this for val/test so all splits share the train vocabulary."
        ),
    )
    parser.add_argument(
        "--tokenized-out",
        type=str,
        default="data/tokenized/sessions.pt",
        help="Output path for tokenized tensor artifact (.pt)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Minimum token frequency to include in vocabulary",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=2_000,
        help="Number of session rows to decode from parquet at a time",
    )
    parser.add_argument(
        "--tokenized-chunk-size",
        type=int,
        default=10_000,
        help="Number of tokenized sessions to keep in memory before flushing a chunk",
    )
    parser.add_argument(
        "--token-id-dtype",
        type=str,
        default="int32",
        choices=("int16", "int32", "int64"),
        help="Torch dtype used to store chunked input_ids. int32 is usually enough and halves disk use vs int64.",
    )
    parser.add_argument(
        "--attention-mask-dtype",
        type=str,
        default="bool",
        choices=("bool", "uint8", "int32", "int64"),
        help="Torch dtype used to store chunked attention_mask tensors.",
    )
    parser.add_argument(
        "--progress-interval-rows",
        type=int,
        default=250_000,
        help="Emit progress after this many rows per shard during vocab/tokenization passes.",
    )
    parser.add_argument(
        "--allow-insufficient-disk",
        action="store_true",
        help="Run even when the estimated tokenized artifact size is larger than available disk.",
    )
    parser.add_argument(
        "--resume-tokenized",
        action="store_true",
        help=(
            "Skip a complete tokenized manifest or continue an incomplete chunk directory "
            "instead of deleting existing tokenized chunks."
        ),
    )
    return parser.parse_args()


def _increment_stat(stats: Optional[Dict[str, int]], key: str, amount: int = 1) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _coerce_events(events: Any, stats: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    if events is None:
        return []

    if isinstance(events, (str, bytes)):
        _increment_stat(stats, "invalid_event_containers")
        return []

    if hasattr(events, "tolist") and not isinstance(events, list):
        events = events.tolist()

    if isinstance(events, tuple):
        events = list(events)

    if not isinstance(events, list):
        _increment_stat(stats, "invalid_event_containers")
        return []

    normalized: List[Dict[str, Any]] = []
    for event in events:
        if isinstance(event, Mapping):
            normalized.append(dict(event))
        else:
            _increment_stat(stats, "dropped_events")
    return normalized


def _count_session_events(sessions: Iterable[Mapping[str, Any]]) -> int:
    return sum(len(session.get("events", [])) for session in sessions)


def _count_non_empty_sessions(sessions: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for session in sessions if session.get("events"))


def _summarize_sessions(sessions: List[Dict[str, Any]]) -> None:
    total_events = _count_session_events(sessions)
    non_empty_sessions = _count_non_empty_sessions(sessions)
    print(f"Loaded {len(sessions):,} sessions ({non_empty_sessions:,} non-empty).")
    print(f"Total events across sessions: {total_events:,}")


def _ensure_sessions_have_events(sessions: List[Dict[str, Any]]) -> None:
    if not any(session.get("events") for session in sessions):
        raise RuntimeError("Sessions were loaded but every session has 0 events after coercion.")


def _validate_vocab_size(vocab: Dict[str, int], min_expected: int = 6) -> None:
    if len(vocab) < min_expected:
        raise RuntimeError(
            "Vocabulary contains only special tokens or too few learned tokens. "
            "Check event parsing and token construction."
        )


def _warn_if_event_token_space_looks_suspicious(
    sessions: Iterable[Mapping[str, Any]],
    vocab: Mapping[str, int],
    min_distinct_tokens: int = 50,
) -> None:
    total_events = 0
    missing_event_ids = 0
    distinct_observed_tokens = set()

    for session in sessions:
        for event in session.get("events", []):
            if not isinstance(event, Mapping):
                continue
            total_events += 1
            event_id = event.get("event_id")
            if event_id is None or str(event_id).strip() in {"", "NA", "UNK"}:
                missing_event_ids += 1
            distinct_observed_tokens.add(build_event_token(event))

    learned_token_count = max(0, len(vocab) - len(LogTokenizer.REQUIRED_SPECIAL_TOKENS))
    if total_events and missing_event_ids:
        print(
            "Warning: events with missing/placeholder event_id detected "
            f"({missing_event_ids:,}/{total_events:,}). "
            "If this is high, check Drain3 enrichment before tokenization."
        )

    if total_events >= min_distinct_tokens and learned_token_count < min_distinct_tokens:
        print(
            "Warning: learned vocabulary is small for this split "
            f"({learned_token_count:,} learned tokens; "
            f"{len(distinct_observed_tokens):,} distinct observed event tokens). "
            "If unexpected, check event_id/template_id generation and --min-freq."
        )


def _extract_day_from_shard_name(parquet_path: Path) -> int:
    match = re.search(r"day_(\d+)\.parquet$", parquet_path.name)
    if not match:
        raise ValueError(
            f"Unable to infer day index from shard name: {parquet_path.name}. "
            "Expected format: day_XX.parquet"
        )
    return int(match.group(1))


def _filter_paths_for_split(parquet_paths: Iterable[Path], split: str) -> List[Path]:
    if split == "all":
        return list(parquet_paths)

    start_day, end_day = SPLIT_DAY_RANGES[split]
    filtered: List[Path] = []
    for parquet_path in parquet_paths:
        day_idx = _extract_day_from_shard_name(parquet_path)
        if start_day <= day_idx <= end_day:
            filtered.append(parquet_path)
    return filtered


def iter_sessions(
    parquet_paths: Iterable[Path],
    parquet_batch_size: int = 2_000,
    progress_label: Optional[str] = None,
    progress_interval_rows: int = 250_000,
    skip_rows: int = 0,
) -> Iterable[Dict[str, Any]]:
    parquet_paths = list(parquet_paths)
    skip_remaining = max(0, skip_rows)
    coercion_stats: Dict[str, int] = {}
    for shard_index, parquet_path in enumerate(parquet_paths, start=1):
        try:
            parquet_file = pq.ParquetFile(parquet_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to read session parquet shard: {parquet_path}") from exc

        shard_rows = parquet_file.metadata.num_rows
        if progress_label:
            print(
                f"[{progress_label}] shard {shard_index}/{len(parquet_paths)}: "
                f"{parquet_path.name} ({shard_rows:,} rows)",
                flush=True,
            )

        rows_seen = 0
        next_progress_at = max(progress_interval_rows, 1)
        for batch in parquet_file.iter_batches(batch_size=parquet_batch_size):
            batch_rows = batch.num_rows
            if skip_remaining >= batch_rows:
                skip_remaining -= batch_rows
                rows_seen += batch_rows
                if progress_label and (rows_seen >= next_progress_at or rows_seen >= shard_rows):
                    print(
                        f"[{progress_label}] {parquet_path.name}: "
                        f"{rows_seen:,}/{shard_rows:,} rows",
                        flush=True,
                    )
                    while next_progress_at <= rows_seen:
                        next_progress_at += max(progress_interval_rows, 1)
                continue
            if skip_remaining:
                batch = batch.slice(skip_remaining)
                skip_remaining = 0
            for row in batch.to_pylist():
                normalized_row = dict(row)
                normalized_row["events"] = _coerce_events(
                    normalized_row.get("events", []),
                    stats=coercion_stats,
                )
                yield normalized_row
            rows_seen += batch_rows
            if progress_label and (rows_seen >= next_progress_at or rows_seen >= shard_rows):
                print(
                    f"[{progress_label}] {parquet_path.name}: "
                    f"{rows_seen:,}/{shard_rows:,} rows",
                    flush=True,
                )
                while next_progress_at <= rows_seen:
                    next_progress_at += max(progress_interval_rows, 1)

    invalid_containers = coercion_stats.get("invalid_event_containers", 0)
    dropped_events = coercion_stats.get("dropped_events", 0)
    if invalid_containers or dropped_events:
        print(
            "Warning: dropped malformed event data while loading sessions "
            f"(invalid event containers={invalid_containers:,}, dropped events={dropped_events:,})."
        )


def _summarize_session_stream(
    parquet_paths: Iterable[Path],
    parquet_batch_size: int,
) -> tuple[int, int, int]:
    session_count = 0
    non_empty_sessions = 0
    total_events = 0

    for session in iter_sessions(parquet_paths, parquet_batch_size=parquet_batch_size):
        session_count += 1
        event_count = len(session.get("events", []))
        total_events += event_count
        if event_count:
            non_empty_sessions += 1

    print(f"Loaded {session_count:,} sessions ({non_empty_sessions:,} non-empty).")
    print(f"Total events across sessions: {total_events:,}")
    return session_count, non_empty_sessions, total_events


def _ensure_session_stream_has_events(
    parquet_paths: Iterable[Path],
    parquet_batch_size: int,
) -> None:
    for session in iter_sessions(parquet_paths, parquet_batch_size=parquet_batch_size):
        if session.get("events"):
            return
    raise RuntimeError("Sessions were loaded but every session has 0 events after coercion.")


def _analyze_event_token_space(
    parquet_paths: Iterable[Path],
    parquet_batch_size: int,
) -> tuple[int, int, int]:
    total_events = 0
    missing_event_ids = 0
    distinct_event_tokens = set()

    for session in iter_sessions(parquet_paths, parquet_batch_size=parquet_batch_size):
        for event in session.get("events", []):
            if not isinstance(event, Mapping):
                continue
            total_events += 1
            event_id = event.get("event_id")
            if event_id is None or str(event_id).strip() in {"", "NA", "UNK"}:
                missing_event_ids += 1
            if len(distinct_event_tokens) < 50:
                distinct_event_tokens.add(build_event_token(event))

    return total_events, missing_event_ids, len(distinct_event_tokens)


def _build_vocab_from_session_shards(
    parquet_paths: Iterable[Path],
    min_freq: int,
    save_path: str,
    parquet_batch_size: int,
    progress_interval_rows: int,
) -> Dict[str, int]:
    vocab_builder = VocabBuilder(min_freq=min_freq)
    token_counts: Counter[str] = Counter()

    for session in iter_sessions(
        parquet_paths,
        parquet_batch_size=parquet_batch_size,
        progress_label="vocab",
        progress_interval_rows=progress_interval_rows,
    ):
        for event in session.get("events", []):
            if isinstance(event, Mapping):
                token_counts[build_event_token(event)] += 1

    return vocab_builder.build_vocab_from_counts(token_counts, save_path=save_path)


def _format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    return getattr(torch, dtype_name)


def _validate_token_id_dtype(vocab: Mapping[str, int], dtype_name: str) -> None:
    max_token_id = max(vocab.values()) if vocab else 0
    dtype_max = TORCH_DTYPE_MAX_VALUES[dtype_name]
    if max_token_id > dtype_max:
        raise RuntimeError(
            f"--token-id-dtype {dtype_name} cannot store max token id {max_token_id:,}. "
            "Use a wider dtype such as int32 or int64."
        )


def _read_parquet_metadata(parquet_paths: Iterable[Path]) -> tuple[int, Dict[Path, int]]:
    total_rows = 0
    rows_by_path: Dict[Path, int] = {}
    for parquet_path in parquet_paths:
        try:
            parquet_file = pq.ParquetFile(parquet_path)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid or unreadable parquet shard: {parquet_path}. "
                "Rebuild or replace this shard before tokenization."
            ) from exc

        rows = parquet_file.metadata.num_rows
        rows_by_path[parquet_path] = rows
        total_rows += rows
    return total_rows, rows_by_path


def _warn_if_split_days_missing(parquet_paths: Iterable[Path], split: str) -> None:
    if split == "all":
        return

    start_day, end_day = SPLIT_DAY_RANGES[split]
    observed_days = {_extract_day_from_shard_name(path) for path in parquet_paths}
    missing_days = [day for day in range(start_day, end_day + 1) if day not in observed_days]
    if missing_days:
        preview = ", ".join(f"{day:02d}" for day in missing_days[:10])
        suffix = "" if len(missing_days) <= 10 else f", ... (+{len(missing_days) - 10} more)"
        print(
            f"Warning: split '{split}' is missing expected day shard(s): {preview}{suffix}.",
            flush=True,
        )


def _preflight_tokenized_output(
    tokenized_out: str,
    session_count: int,
    max_len: int,
    token_id_dtype: str,
    attention_mask_dtype: str,
    allow_insufficient_disk: bool,
) -> None:
    bytes_per_token = TORCH_DTYPE_BYTES[token_id_dtype] + TORCH_DTYPE_BYTES[attention_mask_dtype]
    estimated_bytes = session_count * max_len * bytes_per_token
    output_parent = Path(tokenized_out).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent.resolve()).free

    print(
        "Tokenized output estimate: "
        f"{session_count:,} sessions x max_len={max_len:,} x {bytes_per_token} bytes/token "
        f"~= {_format_bytes(estimated_bytes)} "
        f"(free disk: {_format_bytes(free_bytes)}).",
        flush=True,
    )
    if estimated_bytes > free_bytes * 0.9 and not allow_insufficient_disk:
        raise RuntimeError(
            "Estimated tokenized output is larger than available disk. "
            "Reduce --max-len, rebuild sessions with fewer windows, increase storage, "
            "or pass --allow-insufficient-disk if you intentionally want to try anyway."
        )


def _load_torch_artifact(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tokenized_manifest_complete(tokenized_out: str, expected_session_count: int) -> bool:
    output = Path(tokenized_out)
    if not output.exists() or output.stat().st_size <= 0:
        return False

    try:
        artifact = _load_torch_artifact(output)
    except Exception:
        return False
    if not isinstance(artifact, Mapping):
        return False
    if artifact.get("format") != "tokenized_session_chunk_manifest_v1":
        return False
    if artifact.get("session_count") != expected_session_count:
        return False

    chunks = artifact.get("chunks")
    if not isinstance(chunks, list) or artifact.get("chunk_count") != len(chunks):
        return False
    for relative_chunk_path in chunks:
        if not isinstance(relative_chunk_path, str):
            return False
        chunk_path = output.parent / relative_chunk_path
        if not chunk_path.exists() or chunk_path.stat().st_size <= 0:
            return False
    return True


def _count_resumable_tokenized_sessions(tokenized_out: str) -> int:
    output = Path(tokenized_out)
    chunk_dir = output.parent / f"{output.stem}_chunks"
    if not chunk_dir.exists():
        return 0

    session_count = 0
    for expected_index, chunk_path in enumerate(sorted(chunk_dir.glob("chunk_*.pt"))):
        if chunk_path.name != f"chunk_{expected_index:05d}.pt":
            break
        try:
            chunk = _load_torch_artifact(chunk_path)
        except Exception:
            break
        if not isinstance(chunk, Mapping) or chunk.get("format") != "tokenized_session_chunk_v1":
            break

        session_ids = chunk.get("session_ids")
        input_ids = chunk.get("input_ids")
        attention_mask = chunk.get("attention_mask")
        if not isinstance(session_ids, list):
            break
        row_count = len(session_ids)
        if row_count <= 0:
            break
        if getattr(input_ids, "shape", [None])[0] != row_count:
            break
        if getattr(attention_mask, "shape", [None])[0] != row_count:
            break
        session_count += row_count

    return session_count


def main() -> None:
    args = parse_args()

    if args.min_freq <= 0:
        raise ValueError("--min-freq must be > 0")
    if args.max_len < 3:
        raise ValueError("--max-len must be >= 3")
    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet-batch-size must be > 0")
    if args.tokenized_chunk_size <= 0:
        raise ValueError("--tokenized-chunk-size must be > 0")
    if args.progress_interval_rows <= 0:
        raise ValueError("--progress-interval-rows must be > 0")

    all_parquet_paths = sorted(Path().glob(args.sessions_glob))
    if not all_parquet_paths:
        raise FileNotFoundError(f"No session parquet files found for pattern: {args.sessions_glob}")

    parquet_paths = _filter_paths_for_split(all_parquet_paths, split=args.split)
    if not parquet_paths:
        raise RuntimeError(f"No parquet shards matched split '{args.split}'.")

    split_start, split_end = SPLIT_DAY_RANGES[args.split]
    print(f"Found {len(parquet_paths)} parquet shard(s) for split '{args.split}' (days {split_start}-{split_end}).")
    _warn_if_split_days_missing(parquet_paths, split=args.split)
    session_count, _ = _read_parquet_metadata(parquet_paths)

    requested_vocab_path = Path(args.vocab_in) if args.vocab_in else Path(args.vocab_out)
    if (
        args.resume_tokenized
        and requested_vocab_path.exists()
        and _tokenized_manifest_complete(args.tokenized_out, session_count)
    ):
        print(
            f"[SKIP] split '{args.split}' already has a complete tokenized manifest: "
            f"{args.tokenized_out}",
            flush=True,
        )
        return

    resume_session_count = (
        _count_resumable_tokenized_sessions(args.tokenized_out)
        if args.resume_tokenized
        else 0
    )
    preflight_session_count = max(0, session_count - resume_session_count)
    if resume_session_count:
        print(
            f"[RESUME] split '{args.split}' has {resume_session_count:,} existing "
            f"tokenized session(s); {preflight_session_count:,} remaining.",
            flush=True,
        )

    _preflight_tokenized_output(
        tokenized_out=args.tokenized_out,
        session_count=preflight_session_count,
        max_len=args.max_len,
        token_id_dtype=args.token_id_dtype,
        attention_mask_dtype=args.attention_mask_dtype,
        allow_insufficient_disk=args.allow_insufficient_disk,
    )

    if args.vocab_in:
        vocab_path = Path(args.vocab_in)
        tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=args.max_len)
        vocab = tokenizer.vocab
        print(f"Loaded existing vocabulary from: {vocab_path}")
    else:
        vocab = _build_vocab_from_session_shards(
            parquet_paths,
            min_freq=args.min_freq,
            save_path=args.vocab_out,
            parquet_batch_size=args.parquet_batch_size,
            progress_interval_rows=args.progress_interval_rows,
        )
        vocab_path = Path(args.vocab_out)
        tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=args.max_len)
        print(f"Built vocabulary from split '{args.split}' sessions.")

    _validate_vocab_size(vocab)
    _validate_token_id_dtype(vocab, args.token_id_dtype)

    def report_chunk(chunk_count: int, processed_sessions: int, chunk_path: Path) -> None:
        print(
            f"[tokenize] wrote chunk {chunk_count:,} "
            f"({processed_sessions:,}/{session_count:,} sessions): {chunk_path}",
            flush=True,
        )

    save_stats = tokenizer.save_tokenized_sessions_pt_chunked_with_stats(
        sessions=iter_sessions(
            parquet_paths,
            parquet_batch_size=args.parquet_batch_size,
            progress_label="tokenize",
            progress_interval_rows=args.progress_interval_rows,
            skip_rows=resume_session_count,
        ),
        output_path=args.tokenized_out,
        chunk_size=args.tokenized_chunk_size,
        token_id_dtype=_torch_dtype(args.token_id_dtype),
        attention_mask_dtype=_torch_dtype(args.attention_mask_dtype),
        progress_callback=report_chunk,
        resume=args.resume_tokenized,
        resume_input_already_skipped=resume_session_count > 0,
    )
    completed_from_existing_chunks = (
        args.resume_tokenized
        and resume_session_count >= session_count
        and save_stats.session_count == session_count
        and save_stats.chunk_count > 0
    )
    if save_stats.total_events == 0 and not completed_from_existing_chunks:
        raise RuntimeError("Tokenization completed but found 0 events in the selected session shards.")

    print("\nArtifacts generated successfully:")
    print(f"- Vocabulary size: {len(vocab):,}")
    print(f"- Vocab path: {vocab_path}")
    print(f"- Tokenized path: {save_stats.path}")
    print(f"- Sessions tokenized: {save_stats.session_count:,}")
    print(f"- Chunk files: {save_stats.chunk_count:,}")
    print(f"- Unknown events: {save_stats.unknown_events:,}/{save_stats.total_events:,}")



if __name__ == "__main__":
    main()
