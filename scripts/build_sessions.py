import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.parsing.drain3_parser import LogParser
from src.parsing.session_builder import SessionBuilder

LANL_AUTH_COLUMNS_9 = (
    "time",
    "src_user",
    "dst_user",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success",
)

LANL_AUTH_COLUMNS_7 = (
    "time",
    "src_user",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "success",
)

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass
class BuildStats:
    total_lines: int = 0
    parsed_lines: int = 0
    skipped_lines: int = 0
    processed_days: int = 0
    written_sessions: int = 0
    bucketed_events: int = 0


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description="Build session parquet shards from LANL auth logs.")
    arg_parser.add_argument("--start-day", type=int, default=1, help="First LANL day (1-indexed)")
    arg_parser.add_argument("--end-day", type=int, default=58, help="Last LANL day (inclusive)")
    arg_parser.add_argument("--input", type=str, default="data/raw/auth.txt", help="Path to auth.txt")
    arg_parser.add_argument("--output-dir", type=str, default="data/sessions", help="Output parquet directory")
    arg_parser.add_argument("--window-mins", type=int, default=30, help="Sliding window size in minutes")
    arg_parser.add_argument("--stride-mins", type=int, default=15, help="Window stride in minutes")
    arg_parser.add_argument("--min-events", type=int, default=3, help="Minimum events required per session")
    arg_parser.add_argument("--max-tokens", type=int, default=512, help="Maximum events kept per session")
    arg_parser.add_argument(
        "--parser-state",
        type=str,
        default="data/drain3_state.bin",
        help="Path to Drain3 parser state snapshot",
    )
    arg_parser.add_argument(
        "--bucket-count",
        type=int,
        default=256,
        help="Disk bucket count used for memory-safe day processing (higher reduces peak RAM).",
    )
    return arg_parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_day < 1:
        raise ValueError("--start-day must be >= 1")
    if args.end_day < args.start_day:
        raise ValueError("--end-day must be >= --start-day")
    if args.window_mins <= 0:
        raise ValueError("--window-mins must be > 0")
    if args.stride_mins <= 0:
        raise ValueError("--stride-mins must be > 0")
    if args.min_events <= 0:
        raise ValueError("--min-events must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.bucket_count <= 0:
        raise ValueError("--bucket-count must be > 0")


def get_day_index(timestamp_seconds: int) -> int:
    return (timestamp_seconds // SECONDS_PER_DAY) + 1


def parse_lanl_auth_line(line: str) -> Optional[Dict[str, object]]:
    raw = line.strip()
    if not raw:
        return None

    parts = raw.split(",")
    if len(parts) == len(LANL_AUTH_COLUMNS_9):
        record = dict(zip(LANL_AUTH_COLUMNS_9, parts))
    elif len(parts) == len(LANL_AUTH_COLUMNS_7):
        record = dict(zip(LANL_AUTH_COLUMNS_7, parts))
        record["dst_user"] = "UNKNOWN"
        record["auth_orientation"] = "UNKNOWN"
    else:
        return None

    try:
        timestamp = int(record["time"])
    except (TypeError, ValueError):
        return None

    success_raw = str(record.get("success", "")).strip().lower()
    record["time"] = timestamp
    record["user"] = record.get("src_user", "UNKNOWN")
    record["host"] = record.get("src_computer", "UNKNOWN")
    record["is_success"] = success_raw in {"success", "1", "true", "t", "yes"}
    return record


def build_template_message(event: Dict[str, object]) -> str:
    fields = (
        f"src_user={event.get('src_user', 'UNKNOWN')}",
        f"dst_user={event.get('dst_user', 'UNKNOWN')}",
        f"src_computer={event.get('src_computer', 'UNKNOWN')}",
        f"dst_computer={event.get('dst_computer', 'UNKNOWN')}",
        f"auth_type={event.get('auth_type', 'UNKNOWN')}",
        f"logon_type={event.get('logon_type', 'UNKNOWN')}",
        f"auth_orientation={event.get('auth_orientation', 'UNKNOWN')}",
        f"success={event.get('success', 'UNKNOWN')}",
    )
    return " ".join(fields)


def enrich_with_drain3(event: Dict[str, object], log_parser: LogParser) -> Dict[str, object]:
    enriched = dict(event)
    try:
        template_id, template_params = log_parser.parse(build_template_message(event))
        enriched["template_id"] = template_id
        enriched["event_id"] = str(template_id)
        enriched["template_params"] = template_params
    except Exception:
        # Keep processing even if template extraction fails for a malformed line.
        enriched["template_id"] = -1
        enriched["event_id"] = "UNK"
        enriched["template_params"] = []
    return enriched


def _bucket_key(event: Dict[str, object]) -> str:
    user = str(event.get("src_user", event.get("user", "UNKNOWN")))
    host = str(event.get("src_computer", event.get("host", "UNKNOWN")))
    return f"{user}|{host}"


def _bucket_index(event: Dict[str, object], bucket_count: int) -> int:
    digest = hashlib.blake2b(_bucket_key(event).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % bucket_count


def _close_bucket_handles(handles: Dict[int, TextIO]) -> None:
    for handle in handles.values():
        handle.close()
    handles.clear()


def _write_event_to_bucket(
    day_dir: Path,
    handles: Dict[int, TextIO],
    event: Dict[str, object],
    bucket_count: int,
) -> None:
    idx = _bucket_index(event, bucket_count)
    handle = handles.get(idx)
    if handle is None:
        bucket_path = day_dir / f"bucket_{idx:04d}.jsonl"
        handle = bucket_path.open("a", encoding="utf-8")
        handles[idx] = handle
    handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=True))
    handle.write("\n")


def _write_day_sessions_from_buckets(
    day: int,
    day_dir: Path,
    out_dir: Path,
    builder: SessionBuilder,
) -> tuple[int, int]:
    output_file = out_dir / f"day_{day:02d}.parquet"
    if output_file.exists():
        output_file.unlink()

    writer: Optional[pq.ParquetWriter] = None
    total_day_events = 0
    total_day_sessions = 0

    bucket_files = sorted(day_dir.glob("bucket_*.jsonl"))
    for bucket_file in tqdm(bucket_files, desc=f"Day {day:02d} buckets", leave=False):
        events: List[Dict[str, object]] = []
        with bucket_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                events.append(json.loads(line))

        if not events:
            continue

        total_day_events += len(events)
        sessions = builder.build_sessions(events)
        if not sessions:
            continue

        total_day_sessions += len(sessions)
        dataframe = pd.DataFrame([asdict(session) for session in sessions])
        table = pa.Table.from_pandas(dataframe, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(str(output_file), table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    return total_day_events, total_day_sessions


def main() -> None:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"Argument error: {exc}")
        raise SystemExit(2) from exc

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        raise SystemExit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_parser = LogParser(state_path=args.parser_state)
    if Path(args.parser_state).exists() and log_parser.load_state(args.parser_state):
        print(f"Loaded Drain3 state from {args.parser_state}")

    builder = SessionBuilder(
        window_mins=args.window_mins,
        stride_mins=args.stride_mins,
        min_events=args.min_events,
        max_tokens=args.max_tokens,
        timestamp_unit="seconds",
    )

    stats = BuildStats()
    print(
        f"Building sessions for LANL days {args.start_day} to {args.end_day} "
        f"(bucket_count={args.bucket_count})..."
    )

    temp_root = out_dir / ".tmp_session_buckets"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    current_day: Optional[int] = None
    current_day_dir: Optional[Path] = None
    bucket_handles: Dict[int, TextIO] = {}

    def flush_current_day() -> None:
        nonlocal current_day, current_day_dir
        if current_day is None or current_day_dir is None:
            return

        _close_bucket_handles(bucket_handles)
        day_events, day_sessions = _write_day_sessions_from_buckets(
            day=current_day,
            day_dir=current_day_dir,
            out_dir=out_dir,
            builder=builder,
        )
        stats.processed_days += 1
        stats.written_sessions += day_sessions
        print(f"Day {current_day:02d}: events={day_events:,}, sessions={day_sessions:,}")
        shutil.rmtree(current_day_dir, ignore_errors=True)
        current_day = None
        current_day_dir = None

    try:
        with input_path.open("r", encoding="utf-8", errors="replace") as infile:
            for line in tqdm(infile, desc="Reading auth.txt", unit="lines"):
                stats.total_lines += 1

                event = parse_lanl_auth_line(line)
                if event is None:
                    stats.skipped_lines += 1
                    continue

                stats.parsed_lines += 1
                day = get_day_index(int(event["time"]))

                if day < args.start_day:
                    continue
                if day > args.end_day:
                    break

                if current_day is None:
                    current_day = day
                    current_day_dir = temp_root / f"day_{current_day:02d}"
                    current_day_dir.mkdir(parents=True, exist_ok=True)

                if day != current_day:
                    flush_current_day()
                    current_day = day
                    current_day_dir = temp_root / f"day_{current_day:02d}"
                    current_day_dir.mkdir(parents=True, exist_ok=True)

                enriched = enrich_with_drain3(event, log_parser)
                _write_event_to_bucket(
                    day_dir=current_day_dir,
                    handles=bucket_handles,
                    event=enriched,
                    bucket_count=args.bucket_count,
                )
                stats.bucketed_events += 1

        flush_current_day()
    finally:
        _close_bucket_handles(bucket_handles)
        shutil.rmtree(temp_root, ignore_errors=True)

    log_parser.save_state(args.parser_state)
    print("\nBuild complete.")
    print(f"- Total lines read: {stats.total_lines:,}")
    print(f"- Parsed lines: {stats.parsed_lines:,}")
    print(f"- Skipped lines: {stats.skipped_lines:,}")
    print(f"- Bucketed events: {stats.bucketed_events:,}")
    print(f"- Days processed: {stats.processed_days:,}")
    print(f"- Sessions written: {stats.written_sessions:,}")

if __name__ == "__main__":
    main()
