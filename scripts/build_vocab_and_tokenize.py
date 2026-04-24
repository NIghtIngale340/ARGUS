import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.parsing.log_tokenizer import LogTokenizer
from src.parsing.vocab_builder import VocabBuilder, build_event_token

SPLIT_DAY_RANGES = {
    "train": (1, 40),
    "val": (41, 50),
    "test": (51, 58),
    "all": (1, 58),
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


def load_sessions(parquet_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    coercion_stats: Dict[str, int] = {}

    for parquet_path in parquet_paths:
        dataframe = pd.read_parquet(parquet_path)
        for row in dataframe.to_dict(orient="records"):
            normalized_row = dict(row)
            normalized_row["events"] = _coerce_events(
                normalized_row.get("events", []),
                stats=coercion_stats,
            )
            sessions.append(normalized_row)

    invalid_containers = coercion_stats.get("invalid_event_containers", 0)
    dropped_events = coercion_stats.get("dropped_events", 0)
    if invalid_containers or dropped_events:
        print(
            "Warning: dropped malformed event data while loading sessions "
            f"(invalid event containers={invalid_containers:,}, dropped events={dropped_events:,})."
        )

    return sessions


def main() -> None:
    args = parse_args()

    if args.min_freq <= 0:
        raise ValueError("--min-freq must be > 0")
    if args.max_len < 3:
        raise ValueError("--max-len must be >= 3")

    all_parquet_paths = sorted(Path().glob(args.sessions_glob))
    if not all_parquet_paths:
        raise FileNotFoundError(f"No session parquet files found for pattern: {args.sessions_glob}")

    parquet_paths = _filter_paths_for_split(all_parquet_paths, split=args.split)
    if not parquet_paths:
        raise RuntimeError(
            f"No parquet shards matched split '{args.split}' ({SPLIT_DAY_RANGES[args.split][0]}-"
            f"{SPLIT_DAY_RANGES[args.split][1]})."
        )

    split_start, split_end = SPLIT_DAY_RANGES[args.split]
    print(
        f"Found {len(parquet_paths)} parquet shard(s) for split '{args.split}' "
        f"(days {split_start}-{split_end})."
    )
    sessions = load_sessions(parquet_paths)
    if not sessions:
        raise RuntimeError("Loaded 0 sessions from parquet files.")
    _summarize_sessions(sessions)
    _ensure_sessions_have_events(sessions)

    if args.vocab_in:
        vocab_path = Path(args.vocab_in)
        tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=args.max_len)
        vocab = tokenizer.vocab
        print(f"Loaded existing vocabulary from: {vocab_path}")
    else:
        vocab_builder = VocabBuilder(min_freq=args.min_freq)
        vocab = vocab_builder.build_vocab(sessions=sessions, save_path=args.vocab_out)
        vocab_path = Path(args.vocab_out)
        tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=args.max_len)
        print(f"Built vocabulary from split '{args.split}' sessions.")

    _validate_vocab_size(vocab)
    _warn_if_event_token_space_looks_suspicious(sessions, vocab)
    save_stats = tokenizer.save_tokenized_sessions_pt_with_stats(
        sessions=sessions,
        output_path=args.tokenized_out,
    )

    print("\nArtifacts generated successfully:")
    print(f"- Vocabulary size: {len(vocab):,}")
    print(f"- Vocab path: {vocab_path}")
    print(f"- Tokenized path: {save_stats.path}")
    print(f"- Unknown events: {save_stats.unknown_events:,}/{save_stats.total_events:,}")


if __name__ == "__main__":
    main()
