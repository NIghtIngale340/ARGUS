"""Build MITRE ATT&CK session labels from attack windows and session Parquet.

Input attack windows describe when a known Atomic Red Team/CALDERA/lab attack
ran. This script scans ARGUS session shards and labels every session whose
time window overlaps an attack window, optionally constrained by user/host.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SESSION_COLUMNS = ("session_id", "user_id", "host_id", "start_ts", "end_ts")
VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AttackWindow:
    technique_id: str
    source_run_id: str
    start_ts: int
    end_ts: int
    split: str
    user_id: str | None = None
    host_id: str | None = None
    campaign_id: str | None = None
    attack_window_id: str | None = None


def normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    return {"validation": "val", "valid": "val", "dev": "val"}.get(split, split)


def normalize_technique(value: Any) -> str:
    return str(value).strip().upper()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                rows.append(row)
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as file_obj:
            return [dict(row) for row in csv.DictReader(file_obj)]
    raise ValueError("input file must be .jsonl or .csv")


def load_attack_windows(path: str | Path) -> list[AttackWindow]:
    rows = read_rows(Path(path))
    windows = []
    for row_index, row in enumerate(rows, start=1):
        missing = [
            field
            for field in ("technique_id", "source_run_id", "start_ts", "end_ts", "split")
            if row.get(field) is None or str(row.get(field)).strip() == ""
        ]
        if missing:
            raise ValueError(
                f"attack window row {row_index} missing field(s): {', '.join(missing)}"
            )
        start_ts = int(float(row["start_ts"]))
        end_ts = int(float(row["end_ts"]))
        if end_ts < start_ts:
            raise ValueError(f"attack window row {row_index} has end_ts < start_ts")
        split = normalize_split(row["split"])
        if split not in VALID_SPLITS:
            raise ValueError(f"attack window row {row_index} has invalid split: {split}")
        technique_id = normalize_technique(row["technique_id"])
        source_run_id = str(row["source_run_id"]).strip()
        windows.append(
            AttackWindow(
                technique_id=technique_id,
                source_run_id=source_run_id,
                start_ts=start_ts,
                end_ts=end_ts,
                split=split,
                user_id=_optional_str(row.get("user_id")),
                host_id=_optional_str(row.get("host_id")),
                campaign_id=_optional_str(row.get("campaign_id")),
                attack_window_id=_optional_str(row.get("attack_window_id"))
                or f"{technique_id}_{source_run_id}",
            )
        )
    if not windows:
        raise ValueError("attack window file is empty")
    return windows


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def collect_session_parquet_paths(sessions_path: str | Path) -> list[Path]:
    root = Path(sessions_path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"sessions path does not exist: {root}")
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet session shards found under: {root}")
    return paths


def session_overlaps_window(
    session: dict[str, Any],
    window: AttackWindow,
    *,
    require_user_host_match: bool = True,
) -> bool:
    start_ts = int(session["start_ts"])
    end_ts = int(session["end_ts"])
    if end_ts < window.start_ts or start_ts > window.end_ts:
        return False
    if require_user_host_match:
        if window.user_id is not None and str(session["user_id"]) != window.user_id:
            return False
        if window.host_id is not None and str(session["host_id"]) != window.host_id:
            return False
    return True


def label_session(
    session: dict[str, Any],
    windows: list[AttackWindow],
    *,
    conflict_policy: str = "error",
    require_user_host_match: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    matches = [
        window
        for window in windows
        if session_overlaps_window(
            session,
            window,
            require_user_host_match=require_user_host_match,
        )
    ]
    if not matches:
        return None, False
    if len(matches) > 1:
        distinct = {(match.technique_id, match.source_run_id) for match in matches}
        if len(distinct) > 1 and conflict_policy == "error":
            session_id = session["session_id"]
            details = ", ".join(f"{tech}:{run}" for tech, run in sorted(distinct))
            raise ValueError(f"session {session_id} overlaps multiple attack windows: {details}")
    window = matches[0]
    return build_label_row(session, window), True


def build_label_row(session: dict[str, Any], window: AttackWindow) -> dict[str, Any]:
    return {
        "session_id": str(session["session_id"]),
        "user_id": str(session["user_id"]),
        "host_id": str(session["host_id"]),
        "technique_id": window.technique_id,
        "split": window.split,
        "source_run_id": window.source_run_id,
        "campaign_id": window.campaign_id or window.source_run_id,
        "attack_window_id": window.attack_window_id or f"{window.technique_id}_{window.source_run_id}",
        "session_start_ts": int(session["start_ts"]),
        "session_end_ts": int(session["end_ts"]),
        "attack_start_ts": window.start_ts,
        "attack_end_ts": window.end_ts,
    }


def build_normal_label_row(
    session: dict[str, Any],
    *,
    split: str,
    source_run_id: str = "normal_control",
) -> dict[str, Any]:
    return {
        "session_id": str(session["session_id"]),
        "user_id": str(session["user_id"]),
        "host_id": str(session["host_id"]),
        "technique_id": "normal",
        "split": split,
        "source_run_id": source_run_id,
        "campaign_id": source_run_id,
        "attack_window_id": source_run_id,
        "session_start_ts": int(session["start_ts"]),
        "session_end_ts": int(session["end_ts"]),
    }


def iter_session_rows(
    parquet_paths: Iterable[Path],
    *,
    batch_size: int = 100_000,
) -> Iterable[dict[str, Any]]:
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema.names)
        missing = [column for column in SESSION_COLUMNS if column not in available]
        if missing:
            raise ValueError(
                f"session shard {parquet_path} missing column(s): {', '.join(missing)}"
            )
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(SESSION_COLUMNS),
        ):
            data = batch.to_pydict()
            row_count = len(data["session_id"])
            for index in range(row_count):
                yield {
                    "session_id": data["session_id"][index],
                    "user_id": data["user_id"][index],
                    "host_id": data["host_id"][index],
                    "start_ts": data["start_ts"][index],
                    "end_ts": data["end_ts"][index],
                }


def build_mitre_labels(
    *,
    sessions_path: str | Path,
    attack_windows_path: str | Path,
    out_path: str | Path,
    normal_per_split: int = 0,
    batch_size: int = 100_000,
    conflict_policy: str = "error",
    require_user_host_match: bool = True,
) -> dict[str, Any]:
    if normal_per_split < 0:
        raise ValueError("normal_per_split must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if conflict_policy not in {"error", "first"}:
        raise ValueError("conflict_policy must be 'error' or 'first'")

    windows = load_attack_windows(attack_windows_path)
    parquet_paths = collect_session_parquet_paths(sessions_path)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    attack_count = 0
    normal_counts: Counter = Counter()
    counts_by_technique: Counter = Counter()
    counts_by_split: Counter = Counter()
    seen_sessions: set[str] = set()
    scanned_sessions = 0

    with output.open("w", encoding="utf-8") as out_file:
        for session in iter_session_rows(parquet_paths, batch_size=batch_size):
            scanned_sessions += 1
            session_id = str(session["session_id"])
            label, matched_attack = label_session(
                session,
                windows,
                conflict_policy=conflict_policy,
                require_user_host_match=require_user_host_match,
            )
            if matched_attack:
                if session_id in seen_sessions:
                    continue
                seen_sessions.add(session_id)
                out_file.write(json.dumps(label, separators=(",", ":"), ensure_ascii=True) + "\n")
                attack_count += 1
                counts_by_technique[label["technique_id"]] += 1
                counts_by_split[label["split"]] += 1
                continue

            if normal_per_split > 0 and session_id not in seen_sessions:
                split = _next_normal_split(normal_counts, normal_per_split)
                if split is not None:
                    label = build_normal_label_row(session, split=split)
                    seen_sessions.add(session_id)
                    out_file.write(json.dumps(label, separators=(",", ":"), ensure_ascii=True) + "\n")
                    normal_counts[split] += 1
                    counts_by_technique["normal"] += 1
                    counts_by_split[split] += 1

    report = {
        "sessions_path": str(sessions_path),
        "attack_windows_path": str(attack_windows_path),
        "out_path": str(output),
        "scanned_sessions": scanned_sessions,
        "attack_windows": len(windows),
        "attack_label_count": attack_count,
        "normal_label_count": int(sum(normal_counts.values())),
        "counts_by_technique": dict(sorted(counts_by_technique.items())),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "normal_counts_by_split": dict(sorted(normal_counts.items())),
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _next_normal_split(counts: Counter, limit: int) -> str | None:
    for split in VALID_SPLITS:
        if counts[split] < limit:
            return split
    return None


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build MITRE session labels from attack windows")
    parser.add_argument(
        "--sessions",
        default="data/sessions",
        help="Session parquet directory or file, e.g. data/sessions",
    )
    parser.add_argument(
        "--attack-windows",
        required=True,
        help="JSONL/CSV attack run manifest with technique_id/start_ts/end_ts/split",
    )
    parser.add_argument(
        "--out",
        default="data/attack_labels/mitre_sessions.jsonl",
        help="Output MITRE label JSONL path",
    )
    parser.add_argument(
        "--normal-per-split",
        type=int,
        default=0,
        help="Optional number of non-overlapping normal sessions to add per split",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--conflict-policy",
        choices=("error", "first"),
        default="error",
        help="What to do when one session overlaps multiple distinct attack windows",
    )
    parser.add_argument(
        "--no-user-host-match",
        action="store_true",
        help="Ignore user_id/host_id filters in attack windows; match by time only",
    )
    return parser


def main(args: Namespace | None = None) -> dict[str, Any]:
    parsed = args or build_parser().parse_args()
    report = build_mitre_labels(
        sessions_path=parsed.sessions,
        attack_windows_path=parsed.attack_windows,
        out_path=parsed.out,
        normal_per_split=parsed.normal_per_split,
        batch_size=parsed.batch_size,
        conflict_policy=parsed.conflict_policy,
        require_user_host_match=not parsed.no_user_host_match,
    )
    print(f"Wrote MITRE labels: {report['out_path']}")
    print(f"Scanned sessions: {report['scanned_sessions']:,}")
    print(f"Attack labels: {report['attack_label_count']:,}")
    print(f"Normal labels: {report['normal_label_count']:,}")
    print(f"Counts by technique: {report['counts_by_technique']}")
    print(f"Summary: {Path(report['out_path']).with_suffix('.summary.json')}")
    return report


if __name__ == "__main__":
    main()
