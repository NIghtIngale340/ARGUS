"""Validate MITRE ATT&CK session labels before Phase 3 multi-class training.

The validator checks that the labeled attack dataset is large enough per
technique and that train/val/test splits do not leak sessions or attack runs.
It is intentionally independent of model code so it can run before expensive
fine-tuning starts.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED_TECHNIQUES = ("T1078", "T1021", "T1110", "T1003", "T1059", "T1136")
DEFAULT_GROUP_FIELDS = ("source_run_id", "campaign_id", "attack_window_id")
DEFAULT_NORMAL_LABELS = ("normal", "benign", "none")
REQUIRED_FIELDS = ("session_id", "user_id", "host_id", "technique_id", "split")
VALID_SPLITS = ("train", "val", "test")


def _normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val", "dev": "val"}
    return aliases.get(split, split)


def _normalize_technique(value: Any) -> str:
    return str(value).strip().upper()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            row["_row_number"] = line_number
            rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = []
        for row_number, row in enumerate(csv.DictReader(file_obj), start=2):
            row["_row_number"] = row_number
            rows.append(dict(row))
        return rows


def load_label_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load MITRE labels from JSONL or CSV."""
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"MITRE label file does not exist: {label_path}")
    if label_path.suffix.lower() == ".jsonl":
        return _read_jsonl(label_path)
    if label_path.suffix.lower() == ".csv":
        return _read_csv(label_path)
    raise ValueError("labels file must be .jsonl or .csv")


def _collect_missing_required_fields(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        missing = [
            field
            for field in REQUIRED_FIELDS
            if row.get(field) is None or str(row.get(field)).strip() == ""
        ]
        if missing:
            row_number = row.get("_row_number", "?")
            errors.append(f"row {row_number}: missing required field(s): {', '.join(missing)}")
    return errors


def _build_split_index(
    rows: list[dict[str, Any]],
    *,
    normal_labels: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], Counter, Counter, Counter]:
    session_splits: dict[str, set[str]] = defaultdict(set)
    technique_splits: dict[str, set[str]] = defaultdict(set)
    counts_by_split: Counter = Counter()
    counts_by_technique: Counter = Counter()
    attack_counts_by_technique: Counter = Counter()

    for row in rows:
        split = _normalize_split(row.get("split", ""))
        technique = _normalize_technique(row.get("technique_id", ""))
        session_id = str(row.get("session_id", "")).strip()

        if split:
            counts_by_split[split] += 1
        if technique:
            counts_by_technique[technique] += 1
        if session_id and split:
            session_splits[session_id].add(split)

        if technique and technique.lower() not in normal_labels:
            attack_counts_by_technique[technique] += 1
            technique_splits[technique].add(split)

    return (
        session_splits,
        technique_splits,
        counts_by_split,
        counts_by_technique,
        attack_counts_by_technique,
    )


def _find_group_leakage(
    rows: list[dict[str, Any]],
    *,
    group_fields: Iterable[str],
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    leakage: dict[str, dict[str, list[str]]] = {}
    absent_fields = []

    for field in group_fields:
        group_splits: dict[str, set[str]] = defaultdict(set)
        present = False
        for row in rows:
            raw_group = row.get(field)
            if raw_group is None or str(raw_group).strip() == "":
                continue
            present = True
            split = _normalize_split(row.get("split", ""))
            if split:
                group_splits[str(raw_group).strip()].add(split)

        if not present:
            absent_fields.append(field)
            continue

        field_leaks = {
            group_id: sorted(splits)
            for group_id, splits in group_splits.items()
            if len(splits) > 1
        }
        if field_leaks:
            leakage[field] = field_leaks

    return leakage, absent_fields


def validate_mitre_labels(
    rows: list[dict[str, Any]],
    *,
    min_per_technique: int = 50,
    required_techniques: Iterable[str] = DEFAULT_REQUIRED_TECHNIQUES,
    required_splits: Iterable[str] = VALID_SPLITS,
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    normal_labels: Iterable[str] = DEFAULT_NORMAL_LABELS,
) -> dict[str, Any]:
    """Validate label rows and return a structured report."""
    if min_per_technique <= 0:
        raise ValueError("min_per_technique must be positive")

    required_technique_set = {_normalize_technique(value) for value in required_techniques}
    required_split_set = {_normalize_split(value) for value in required_splits}
    normal_label_set = {str(value).strip().lower() for value in normal_labels}
    errors = []
    warnings = []

    if not rows:
        errors.append("label file is empty")

    errors.extend(_collect_missing_required_fields(rows))

    (
        session_splits,
        technique_splits,
        counts_by_split,
        counts_by_technique,
        attack_counts_by_technique,
    ) = _build_split_index(rows, normal_labels=normal_label_set)

    invalid_splits = sorted(set(counts_by_split) - set(VALID_SPLITS))
    if invalid_splits:
        errors.append(f"invalid split value(s): {', '.join(invalid_splits)}")

    for split in sorted(required_split_set):
        if counts_by_split[split] == 0:
            errors.append(f"required split has no rows: {split}")

    leaking_sessions = {
        session_id: sorted(splits)
        for session_id, splits in session_splits.items()
        if len(splits) > 1
    }
    if leaking_sessions:
        preview = ", ".join(sorted(leaking_sessions)[:5])
        errors.append(
            "session_id leakage across splits: "
            f"{len(leaking_sessions)} session(s), examples: {preview}"
        )

    group_leakage, absent_group_fields = _find_group_leakage(rows, group_fields=group_fields)
    for field, leaks in group_leakage.items():
        preview = ", ".join(sorted(leaks)[:5])
        errors.append(
            f"{field} leakage across splits: {len(leaks)} group(s), examples: {preview}"
        )
    if absent_group_fields:
        warnings.append(
            "missing optional anti-leakage group field(s): "
            + ", ".join(absent_group_fields)
        )

    for technique in sorted(required_technique_set):
        count = attack_counts_by_technique[technique]
        if count < min_per_technique:
            errors.append(
                f"{technique} has {count} labeled attack session(s), "
                f"requires >= {min_per_technique}"
            )

    attack_techniques = sorted(attack_counts_by_technique)
    for technique in attack_techniques:
        splits = technique_splits[technique]
        missing_splits = sorted(required_split_set - splits)
        if missing_splits:
            errors.append(
                f"{technique} missing required split coverage: "
                + ", ".join(missing_splits)
            )

    duplicate_rows = len(rows) - len(
        {
            (
                str(row.get("session_id", "")).strip(),
                _normalize_technique(row.get("technique_id", "")),
                _normalize_split(row.get("split", "")),
            )
            for row in rows
        }
    )
    if duplicate_rows:
        warnings.append(f"duplicate session/technique/split row(s): {duplicate_rows}")

    report = {
        "valid": not errors,
        "row_count": len(rows),
        "min_per_technique": min_per_technique,
        "required_techniques": sorted(required_technique_set),
        "required_splits": sorted(required_split_set),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_technique": dict(sorted(counts_by_technique.items())),
        "attack_counts_by_technique": dict(sorted(attack_counts_by_technique.items())),
        "technique_split_coverage": {
            technique: sorted(splits)
            for technique, splits in sorted(technique_splits.items())
        },
        "session_leakage_count": len(leaking_sessions),
        "group_leakage": group_leakage,
        "warnings": warnings,
        "errors": errors,
    }
    return report


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Validate MITRE ATT&CK session labels")
    parser.add_argument(
        "--labels",
        default="data/attack_labels/mitre_sessions.jsonl",
        help="MITRE labels file (.jsonl or .csv)",
    )
    parser.add_argument(
        "--out",
        default="data/attack_labels/mitre_label_validation_report.json",
        help="Validation report JSON path",
    )
    parser.add_argument("--min-per-technique", type=int, default=50)
    parser.add_argument(
        "--required-technique",
        action="append",
        dest="required_techniques",
        help=(
            "Technique that must meet --min-per-technique. Repeat to override "
            "the default guide set."
        ),
    )
    parser.add_argument(
        "--group-field",
        action="append",
        dest="group_fields",
        help=(
            "Optional group field that must not cross train/val/test. Repeat to "
            "override source_run_id/campaign_id/attack_window_id."
        ),
    )
    return parser


def main(args: Namespace | None = None) -> dict[str, Any]:
    parsed = args or build_parser().parse_args()
    required_techniques = (
        parsed.required_techniques
        if parsed.required_techniques
        else DEFAULT_REQUIRED_TECHNIQUES
    )
    group_fields = parsed.group_fields if parsed.group_fields else DEFAULT_GROUP_FIELDS

    rows = load_label_rows(parsed.labels)
    report = validate_mitre_labels(
        rows,
        min_per_technique=parsed.min_per_technique,
        required_techniques=required_techniques,
        group_fields=group_fields,
    )

    out_path = Path(parsed.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["valid"]:
        print(
            "MITRE label validation passed: "
            f"{report['row_count']} rows, "
            f"{len(report['attack_counts_by_technique'])} attack technique(s)"
        )
    else:
        print("MITRE label validation failed:")
        for error in report["errors"]:
            print(f"  - {error}")
        print(f"Report written to: {out_path}")
        raise SystemExit(1)

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    print(f"Report written to: {out_path}")
    return report


if __name__ == "__main__":
    main()
