from pathlib import Path

import pyarrow.parquet as pq

from scripts import build_sessions
from src.parsing.session_builder import SessionBuilder


def _write_bucket_file(bucket_path: Path, events: list[dict[str, object]]) -> None:
    with bucket_path.open("wb") as handle:
        for event in events:
            handle.write(build_sessions._json_dumps_bytes(event))
            handle.write(b"\n")


def _sample_bucket_events() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    bucket_a = [
        {
            "time": 0,
            "src_user": "alice",
            "src_computer": "pc1",
            "event_id": f"A{idx}",
            "template_id": idx,
            "auth_type": "Kerberos",
        }
        for idx in range(3)
    ]
    bucket_b = [
        {
            "time": 100 + idx,
            "src_user": "bob",
            "src_computer": "pc2",
            "event_id": f"B{idx}",
            "template_id": 10 + idx,
            "auth_type": "NTLM",
        }
        for idx in range(3)
    ]
    return bucket_a, bucket_b


def test_write_day_sessions_from_buckets_matches_for_single_and_multi_worker(tmp_path: Path) -> None:
    day_dir = tmp_path / "day_01"
    day_dir.mkdir()

    bucket_a, bucket_b = _sample_bucket_events()
    _write_bucket_file(day_dir / "bucket_0000.jsonl", bucket_a)
    _write_bucket_file(day_dir / "bucket_0001.jsonl", bucket_b)

    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=3, timestamp_unit="seconds")

    out_dir_single = tmp_path / "out_single"
    out_dir_multi = tmp_path / "out_multi"
    out_dir_single.mkdir()
    out_dir_multi.mkdir()

    single_events, single_sessions = build_sessions._write_day_sessions_from_buckets(
        day=1,
        day_dir=day_dir,
        out_dir=out_dir_single,
        builder=builder,
        bucket_workers=1,
    )
    multi_events, multi_sessions = build_sessions._write_day_sessions_from_buckets(
        day=1,
        day_dir=day_dir,
        out_dir=out_dir_multi,
        builder=builder,
        bucket_workers=2,
    )

    assert single_events == multi_events == 6
    assert single_sessions == multi_sessions == 2

    single_file = out_dir_single / "day_01.parquet"
    multi_file = out_dir_multi / "day_01.parquet"
    assert single_file.exists() and single_file.stat().st_size > 0
    assert multi_file.exists() and multi_file.stat().st_size > 0
    assert pq.read_table(single_file).num_rows == pq.read_table(multi_file).num_rows == 2


def test_write_day_sessions_from_buckets_falls_back_to_stdlib_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(build_sessions, "_orjson", None)

    day_dir = tmp_path / "day_02"
    out_dir = tmp_path / "out"
    day_dir.mkdir()
    out_dir.mkdir()

    bucket_a, _ = _sample_bucket_events()
    _write_bucket_file(day_dir / "bucket_0000.jsonl", bucket_a)

    builder = SessionBuilder(window_mins=30, stride_mins=15, min_events=3, timestamp_unit="seconds")
    day_events, day_sessions = build_sessions._write_day_sessions_from_buckets(
        day=2,
        day_dir=day_dir,
        out_dir=out_dir,
        builder=builder,
        bucket_workers=1,
    )

    output_file = out_dir / "day_02.parquet"
    assert day_events == 3
    assert day_sessions == 1
    assert output_file.exists() and output_file.stat().st_size > 0
    assert pq.read_table(output_file).num_rows == 1
