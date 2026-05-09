"""Tests for MITRE ATT&CK label validation."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.validate_mitre_labels import (
    load_label_rows,
    main,
    validate_mitre_labels,
)


def _row(
    session_id: str,
    technique_id: str,
    split: str,
    *,
    source_run_id: str | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "user_id": f"user_{session_id}",
        "host_id": f"host_{session_id}",
        "technique_id": technique_id,
        "split": split,
        "source_run_id": source_run_id or f"run_{session_id}",
        "campaign_id": f"campaign_{session_id}",
        "attack_window_id": f"window_{session_id}",
    }


def _valid_rows() -> list[dict]:
    rows = []
    for technique in ("T1078", "T1021"):
        for split in ("train", "val", "test"):
            rows.append(_row(f"{technique}_{split}_a", technique, split))
            rows.append(_row(f"{technique}_{split}_b", technique, split))
    return rows


def test_validate_mitre_labels_accepts_clean_multiclass_split() -> None:
    report = validate_mitre_labels(
        _valid_rows(),
        min_per_technique=6,
        required_techniques=("T1078", "T1021"),
    )

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["attack_counts_by_technique"] == {"T1021": 6, "T1078": 6}
    assert report["technique_split_coverage"]["T1078"] == ["test", "train", "val"]


def test_validate_mitre_labels_rejects_small_technique_counts() -> None:
    report = validate_mitre_labels(
        _valid_rows(),
        min_per_technique=7,
        required_techniques=("T1078", "T1021"),
    )

    assert report["valid"] is False
    assert any("T1078 has 6" in error for error in report["errors"])
    assert any("T1021 has 6" in error for error in report["errors"])


def test_validate_mitre_labels_rejects_missing_required_technique() -> None:
    report = validate_mitre_labels(
        _valid_rows(),
        min_per_technique=1,
        required_techniques=("T1078", "T1003"),
    )

    assert report["valid"] is False
    assert any("T1003 has 0" in error for error in report["errors"])


def test_validate_mitre_labels_rejects_session_split_leakage() -> None:
    rows = _valid_rows()
    rows.append(_row("leaky_session", "T1078", "train"))
    rows.append(_row("leaky_session", "T1078", "test"))

    report = validate_mitre_labels(
        rows,
        min_per_technique=1,
        required_techniques=("T1078",),
    )

    assert report["valid"] is False
    assert any("session_id leakage" in error for error in report["errors"])


def test_validate_mitre_labels_rejects_group_split_leakage() -> None:
    rows = _valid_rows()
    rows.append(_row("run_leak_train", "T1078", "train", source_run_id="atomic_run_1"))
    rows.append(_row("run_leak_test", "T1078", "test", source_run_id="atomic_run_1"))

    report = validate_mitre_labels(
        rows,
        min_per_technique=1,
        required_techniques=("T1078",),
    )

    assert report["valid"] is False
    assert any("source_run_id leakage" in error for error in report["errors"])
    assert report["group_leakage"]["source_run_id"]["atomic_run_1"] == ["test", "train"]


def test_validate_mitre_labels_rejects_missing_fields_and_bad_splits() -> None:
    rows = [_row("s1", "T1078", "train")]
    rows.append({"session_id": "bad", "technique_id": "T1078", "split": "holdout"})

    report = validate_mitre_labels(
        rows,
        min_per_technique=1,
        required_techniques=("T1078",),
    )

    assert report["valid"] is False
    assert any("missing required field" in error for error in report["errors"])
    assert any("invalid split" in error for error in report["errors"])


def test_load_label_rows_supports_jsonl_and_csv(tmp_path: Path) -> None:
    rows = _valid_rows()[:2]
    jsonl_path = tmp_path / "labels.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "session_id,user_id,host_id,technique_id,split,source_run_id,campaign_id,attack_window_id\n"
        "s1,u1,h1,T1078,train,r1,c1,w1\n",
        encoding="utf-8",
    )

    assert len(load_label_rows(jsonl_path)) == 2
    assert load_label_rows(csv_path)[0]["session_id"] == "s1"


def test_main_writes_report_for_valid_labels(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(json.dumps(row) for row in _valid_rows()),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    report = main(
        Namespace(
            labels=str(labels),
            out=str(out),
            min_per_technique=6,
            required_techniques=["T1078", "T1021"],
            group_fields=None,
        )
    )

    assert report["valid"] is True
    assert json.loads(out.read_text(encoding="utf-8"))["valid"] is True


def test_main_exits_nonzero_for_invalid_labels(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(_row("only_one", "T1078", "train")),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    with pytest.raises(SystemExit) as exc_info:
        main(
            Namespace(
                labels=str(labels),
                out=str(out),
                min_per_technique=2,
                required_techniques=["T1078"],
                group_fields=None,
            )
        )

    assert exc_info.value.code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["valid"] is False
