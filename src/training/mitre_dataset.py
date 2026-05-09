"""MITRE ATT&CK label-driven datasets for ARGUS classifier fine-tuning."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.parsing.log_tokenizer import (
    TOKENIZED_CHUNK_FORMAT,
    TOKENIZED_CHUNK_MANIFEST_FORMAT,
)


NORMAL_LABEL = "normal"


@dataclass(frozen=True)
class MITRELabel:
    session_id: str
    user_id: str
    host_id: str
    technique_id: str
    split: str


def normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    return {"validation": "val", "valid": "val", "dev": "val"}.get(split, split)


def normalize_technique(value: Any) -> str:
    technique = str(value).strip()
    if technique.lower() in {"normal", "benign", "none"}:
        return NORMAL_LABEL
    return technique.upper()


def load_mitre_labels(path: str | Path) -> list[MITRELabel]:
    """Load MITRE label rows from JSONL or CSV."""
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"MITRE labels file does not exist: {label_path}")

    if label_path.suffix.lower() == ".jsonl":
        raw_rows = []
        with label_path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                raw_rows.append(row)
    elif label_path.suffix.lower() == ".csv":
        with label_path.open("r", newline="", encoding="utf-8") as file_obj:
            raw_rows = list(csv.DictReader(file_obj))
    else:
        raise ValueError("MITRE labels file must be .jsonl or .csv")

    labels = []
    for row_index, row in enumerate(raw_rows, start=1):
        missing = [
            field
            for field in ("session_id", "user_id", "host_id", "technique_id", "split")
            if row.get(field) is None or str(row.get(field)).strip() == ""
        ]
        if missing:
            raise ValueError(
                f"MITRE label row {row_index} missing field(s): {', '.join(missing)}"
            )
        labels.append(
            MITRELabel(
                session_id=str(row["session_id"]).strip(),
                user_id=str(row["user_id"]).strip(),
                host_id=str(row["host_id"]).strip(),
                technique_id=normalize_technique(row["technique_id"]),
                split=normalize_split(row["split"]),
            )
        )
    return labels


def build_class_names(
    labels: Iterable[MITRELabel],
    *,
    include_normal: bool = True,
    class_names: Sequence[str] | None = None,
) -> list[str]:
    """Build stable class names with normal at index 0."""
    if class_names:
        normalized = [normalize_technique(value) for value in class_names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("class_names contains duplicate classes")
        return normalized

    techniques = sorted({label.technique_id for label in labels if label.technique_id != NORMAL_LABEL})
    return ([NORMAL_LABEL] if include_normal else []) + techniques


class MITREClassificationDataset(Dataset):
    """Map-style dataset joining tokenized session chunks to MITRE labels."""

    def __init__(
        self,
        manifest_paths: str | Path | Sequence[str | Path],
        labels_path: str | Path,
        *,
        split: str,
        class_names: Sequence[str] | None = None,
        limit_chunks: int | None = None,
        map_location: str = "cpu",
    ) -> None:
        if isinstance(manifest_paths, (str, Path)):
            self.manifest_paths = [Path(manifest_paths)]
        else:
            self.manifest_paths = [Path(path) for path in manifest_paths]
        if not self.manifest_paths:
            raise ValueError("manifest_paths must not be empty")
        if limit_chunks is not None and limit_chunks < 0:
            raise ValueError("limit_chunks must be >= 0 when provided")

        labels = load_mitre_labels(labels_path)
        active_split = normalize_split(split)
        split_labels = [label for label in labels if label.split == active_split]
        if not split_labels:
            raise ValueError(f"No MITRE labels found for split: {active_split}")

        self.class_names = build_class_names(labels, class_names=class_names)
        self.label_to_id = {label: index for index, label in enumerate(self.class_names)}
        self.id_to_label = {index: label for label, index in self.label_to_id.items()}
        self.samples: list[dict[str, Any]] = []
        self.map_location = map_location

        label_by_session: dict[str, MITRELabel] = {}
        for label in split_labels:
            if label.session_id in label_by_session:
                raise ValueError(f"Duplicate MITRE label for session_id: {label.session_id}")
            if label.technique_id not in self.label_to_id:
                raise ValueError(
                    f"Technique {label.technique_id} missing from class_names"
                )
            label_by_session[label.session_id] = label

        self._load_matching_sessions(label_by_session, limit_chunks=limit_chunks)
        if not self.samples:
            raise ValueError(
                f"No tokenized sessions matched MITRE labels for split: {active_split}"
            )

    @property
    def labels(self) -> list[int]:
        return [int(sample["label"]) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def _torch_load(self, path: Path) -> Any:
        try:
            return torch.load(path, map_location=self.map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.map_location)

    def _iter_manifest_chunks(self, manifest_path: Path, limit_chunks: int | None) -> Iterable[Path]:
        artifact = self._torch_load(manifest_path)
        if isinstance(artifact, Mapping) and artifact.get("format") == TOKENIZED_CHUNK_MANIFEST_FORMAT:
            chunks = artifact.get("chunks")
            if not isinstance(chunks, list):
                raise ValueError(f"Tokenized manifest chunks must be a list: {manifest_path}")
            for index, chunk_ref in enumerate(chunks):
                if limit_chunks is not None and index >= limit_chunks:
                    break
                chunk_path = Path(chunk_ref)
                yield chunk_path if chunk_path.is_absolute() else manifest_path.parent / chunk_path
            return

        if isinstance(artifact, list):
            yield manifest_path
            return

        raise ValueError(f"Unsupported tokenized manifest format: {manifest_path}")

    def _load_matching_sessions(
        self,
        label_by_session: dict[str, MITRELabel],
        *,
        limit_chunks: int | None,
    ) -> None:
        needed = set(label_by_session)
        found: set[str] = set()

        for manifest_path in self.manifest_paths:
            artifact = self._torch_load(manifest_path)
            if isinstance(artifact, list):
                self._load_from_legacy_list(artifact, label_by_session, found)
                continue

            if not isinstance(artifact, Mapping) or artifact.get("format") != TOKENIZED_CHUNK_MANIFEST_FORMAT:
                raise ValueError(f"Unsupported tokenized manifest format: {manifest_path}")
            chunks = artifact["chunks"]
            for chunk_index, chunk_ref in enumerate(chunks):
                if limit_chunks is not None and chunk_index >= limit_chunks:
                    break
                chunk_path = Path(chunk_ref)
                if not chunk_path.is_absolute():
                    chunk_path = manifest_path.parent / chunk_path
                chunk = self._torch_load(chunk_path)
                self._load_from_chunk(chunk, chunk_path, label_by_session, found)
                if found >= needed:
                    return

    def _load_from_legacy_list(
        self,
        items: list[Any],
        label_by_session: dict[str, MITRELabel],
        found: set[str],
    ) -> None:
        for row_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError("Legacy tokenized session item must be a mapping")
            session_id = str(item.get("session_id", row_index))
            label = label_by_session.get(session_id)
            if label is None:
                continue
            self._append_sample(
                label=label,
                input_ids=item["input_ids"],
                attention_mask=item["attention_mask"],
            )
            found.add(session_id)

    def _load_from_chunk(
        self,
        chunk: Any,
        chunk_path: Path,
        label_by_session: dict[str, MITRELabel],
        found: set[str],
    ) -> None:
        if not isinstance(chunk, Mapping) or chunk.get("format") != TOKENIZED_CHUNK_FORMAT:
            raise ValueError(f"Unexpected tokenized chunk format: {chunk_path}")
        session_ids = chunk.get("session_ids")
        input_ids = chunk.get("input_ids")
        attention_mask = chunk.get("attention_mask")
        if not isinstance(session_ids, list):
            raise ValueError(f"Tokenized chunk missing session_ids: {chunk_path}")
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise ValueError(f"Tokenized chunk missing tensors: {chunk_path}")
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != len(session_ids):
            raise ValueError(f"Tokenized chunk row counts do not match: {chunk_path}")

        for row_index, raw_session_id in enumerate(session_ids):
            session_id = str(raw_session_id)
            label = label_by_session.get(session_id)
            if label is None:
                continue
            self._append_sample(
                label=label,
                input_ids=input_ids[row_index],
                attention_mask=attention_mask[row_index],
            )
            found.add(session_id)

    def _append_sample(
        self,
        *,
        label: MITRELabel,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> None:
        self.samples.append(
            {
                "session_id": label.session_id,
                "user_id": label.user_id,
                "host_id": label.host_id,
                "technique_id": label.technique_id,
                "label": self.label_to_id[label.technique_id],
                "input_ids": input_ids.long(),
                "attention_mask": attention_mask.bool(),
            }
        )


def collate_mitre_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_id": [item["session_id"] for item in batch],
        "user_id": [item["user_id"] for item in batch],
        "host_id": [item["host_id"] for item in batch],
        "technique_id": [item["technique_id"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
