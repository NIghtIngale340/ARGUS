import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.parsing.drain3_parser import LogParser
from src.parsing.session_builder import Session, SessionBuilder

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
        enriched["template_params"] = template_params
    except Exception:
        # Keep processing even if template extraction fails for a malformed line.
        enriched["template_id"] = -1
        enriched["template_params"] = []
    return enriched


def iter_events_grouped_by_day(
    input_path: Path,
    start_day: int,
    end_day: int,
    log_parser: LogParser,
    stats: BuildStats,
) -> Iterator[Tuple[int, List[Dict[str, object]]]]:
    current_day: Optional[int] = None
    current_events: List[Dict[str, object]] = []

    with input_path.open("r", encoding="utf-8", errors="replace") as infile:
        for line in tqdm(infile, desc="Reading auth.txt", unit="lines"):
            stats.total_lines += 1

            event = parse_lanl_auth_line(line)
            if event is None:
                stats.skipped_lines += 1
                continue

            stats.parsed_lines += 1
            day = get_day_index(int(event["time"]))

            if day < start_day:
                continue
            if day > end_day:
                break

            if current_day is None:
                current_day = day

            if day != current_day:
                yield current_day, current_events
                current_day = day
                current_events = []

            current_events.append(enrich_with_drain3(event, log_parser))

    if current_day is not None:
        yield current_day, current_events


def write_day_sessions(day: int, sessions: Sequence[Session], out_dir: Path) -> int:
    if not sessions:
        return 0

    session_dicts = [asdict(session) for session in sessions]
    output_file = out_dir / f"day_{day:02d}.parquet"
    pd.DataFrame(session_dicts).to_parquet(
        output_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    return len(session_dicts)


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
    print(f"Building sessions for LANL days {args.start_day} to {args.end_day}...")

    for day, day_events in iter_events_grouped_by_day(
        input_path=input_path,
        start_day=args.start_day,
        end_day=args.end_day,
        log_parser=log_parser,
        stats=stats,
    ):
        stats.processed_days += 1
        sessions = builder.build_sessions(day_events)
        written = write_day_sessions(day, sessions, out_dir)
        stats.written_sessions += written
        print(f"Day {day:02d}: events={len(day_events):,}, sessions={written:,}")

    log_parser.save_state(args.parser_state)
    print("\nBuild complete.")
    print(f"- Total lines read: {stats.total_lines:,}")
    print(f"- Parsed lines: {stats.parsed_lines:,}")
    print(f"- Skipped lines: {stats.skipped_lines:,}")
    print(f"- Days processed: {stats.processed_days:,}")
    print(f"- Sessions written: {stats.written_sessions:,}")

if __name__ == "__main__":
    main()
