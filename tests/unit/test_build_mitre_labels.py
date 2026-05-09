"""Tests for building MITRE labels from attack windows."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_mitre_labels import (
    build_mitre_labels,
    label_session,
    load_attack_windows,
    main,
)


def _write_sessions(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "session_id": "normal_a",
                "user_id": "u0",
                "host_id": "h0",
                "start_ts": 0,
                "end_ts": 50,
                "events": [],
            },
            {
                "session_id": "t1078_hit",
                "user_id": "u1",
                "host_id": "h1",
                "start_ts": 100,
                "end_ts": 200,
                "events": [],
            },
            {
                "session_id": "wrong_host",
                "user_id": "u1",
                "host_id": "h9",
                "start_ts": 120,
                "end_ts": 180,
                "events": [],
            },
            {
                "session_id": "t1021_hit",
                "user_id": "u2",
                "host_id": "h2",
                "start_ts": 300,
                "end_ts": 360,
                "events": [],
            },
            {
                "session_id": "normal_b",
                "user_id": "u3",
                "host_id": "h3",
                "start_ts": 1_000,
                "end_ts": 1_100,
                "events": [],
            },
        ]
    ).to_parquet(path, index=False)
    return path


def _write_windows(path: Path) -> Path:
    rows = [
        {
            "technique_id": "T1078",
            "source_run_id": "atomic_t1078_001",
            "campaign_id": "campaign_a",
            "attack_window_id": "window_t1078",
            "user_id": "u1",
            "host_id": "h1",
            "start_ts": 90,
            "end_ts": 210,
            "split": "train",
        },
        {
            "technique_id": "T1021",
            "source_run_id": "atomic_t1021_001",
            "campaign_id": "campaign_b",
            "attack_window_id": "window_t1021",
            "user_id": "u2",
            "host_id": "h2",
            "start_ts": 250,
            "end_ts": 400,
            "split": "test",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_mitre_labels_matches_attack_windows_and_samples_normals(tmp_path: Path) -> None:
    sessions = _write_sessions(tmp_path / "sessions" / "day_01.parquet")
    windows = _write_windows(tmp_path / "attack_windows.jsonl")
    output = tmp_path / "labels" / "mitre_sessions.jsonl"

    report = build_mitre_labels(
        sessions_path=sessions.parent,
        attack_windows_path=windows,
        out_path=output,
        normal_per_split=1,
        batch_size=2,
    )

    rows = _read_jsonl(output)
    by_session = {row["session_id"]: row for row in rows}

    assert report["scanned_sessions"] == 5
    assert report["attack_label_count"] == 2
    assert report["normal_label_count"] == 3
    assert by_session["t1078_hit"]["technique_id"] == "T1078"
    assert by_session["t1078_hit"]["split"] == "train"
    assert by_session["t1021_hit"]["technique_id"] == "T1021"
    assert "wrong_host" in by_session
    assert by_session["wrong_host"]["technique_id"] == "normal"
    assert (output.with_suffix(".summary.json")).exists()


def test_load_attack_windows_validates_required_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"technique_id": "T1078"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing field"):
        load_attack_windows(bad)


def test_label_session_detects_conflicting_attack_windows(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.jsonl"
    rows = [
        {
            "technique_id": "T1078",
            "source_run_id": "run1",
            "start_ts": 10,
            "end_ts": 50,
            "split": "train",
        },
        {
            "technique_id": "T1021",
            "source_run_id": "run2",
            "start_ts": 20,
            "end_ts": 60,
            "split": "train",
        },
    ]
    windows_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    windows = load_attack_windows(windows_path)

    with pytest.raises(ValueError, match="multiple attack windows"):
        label_session(
            {
                "session_id": "s1",
                "user_id": "u1",
                "host_id": "h1",
                "start_ts": 25,
                "end_ts": 30,
            },
            windows,
        )


def test_no_user_host_match_allows_time_only_labeling(tmp_path: Path) -> None:
    sessions = _write_sessions(tmp_path / "sessions" / "day_01.parquet")
    windows = _write_windows(tmp_path / "attack_windows.jsonl")
    output = tmp_path / "labels.jsonl"

    build_mitre_labels(
        sessions_path=sessions,
        attack_windows_path=windows,
        out_path=output,
        normal_per_split=0,
        require_user_host_match=False,
    )

    rows = _read_jsonl(output)
    by_session = {row["session_id"]: row for row in rows}
    assert by_session["wrong_host"]["technique_id"] == "T1078"


def test_main_writes_labels(tmp_path: Path) -> None:
    sessions = _write_sessions(tmp_path / "sessions" / "day_01.parquet")
    windows = _write_windows(tmp_path / "attack_windows.jsonl")
    output = tmp_path / "mitre_sessions.jsonl"

    report = main(
        Namespace(
            sessions=str(sessions.parent),
            attack_windows=str(windows),
            out=str(output),
            normal_per_split=0,
            batch_size=2,
            conflict_policy="error",
            no_user_host_match=False,
        )
    )

    assert report["attack_label_count"] == 2
    assert output.exists()
