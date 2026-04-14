import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.parsing.log_tokenizer import LogTokenizer
from src.parsing.vocab_builder import VocabBuilder


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
        "--vocab-out",
        type=str,
        default="data/vocab.json",
        help="Output path for vocabulary JSON",
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


def _coerce_events(events: Any) -> List[Dict[str, Any]]:
    if events is None:
        return []

    if isinstance(events, (str, bytes)):
        return []

    if hasattr(events, "tolist") and not isinstance(events, list):
        events = events.tolist()

    if isinstance(events, tuple):
        events = list(events)

    if not isinstance(events, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for event in events:
        if isinstance(event, Mapping):
            normalized.append(dict(event))
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


def load_sessions(parquet_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []

    for parquet_path in parquet_paths:
        dataframe = pd.read_parquet(parquet_path)
        for row in dataframe.to_dict(orient="records"):
            normalized_row = dict(row)
            normalized_row["events"] = _coerce_events(normalized_row.get("events", []))
            sessions.append(normalized_row)

    return sessions


def main() -> None:
    args = parse_args()

    if args.min_freq <= 0:
        raise ValueError("--min-freq must be > 0")
    if args.max_len < 3:
        raise ValueError("--max-len must be >= 3")

    parquet_paths = sorted(Path().glob(args.sessions_glob))
    if not parquet_paths:
        raise FileNotFoundError(f"No session parquet files found for pattern: {args.sessions_glob}")

    print(f"Found {len(parquet_paths)} parquet shard(s).")
    sessions = load_sessions(parquet_paths)
    if not sessions:
        raise RuntimeError("Loaded 0 sessions from parquet files.")
    _summarize_sessions(sessions)
    _ensure_sessions_have_events(sessions)

    vocab_builder = VocabBuilder(min_freq=args.min_freq)
    vocab = vocab_builder.build_vocab(sessions=sessions, save_path=args.vocab_out)
    _validate_vocab_size(vocab)

    tokenizer = LogTokenizer(vocab_path=args.vocab_out, max_len=args.max_len)
    tokenized_path = tokenizer.save_tokenized_sessions_pt(sessions=sessions, output_path=args.tokenized_out)

    print("\nArtifacts generated successfully:")
    print(f"- Vocabulary size: {len(vocab):,}")
    print(f"- Vocab path: {Path(args.vocab_out)}")
    print(f"- Tokenized path: {tokenized_path}")


if __name__ == "__main__":
    main()
