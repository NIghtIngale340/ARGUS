"""Reusable Phase 3 bundle inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch.utils.data import DataLoader

from src.inference.alert_engine import AlertEngine, ScoredSession
from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig
from src.parsing.log_tokenizer import LogTokenizer
from src.training.dataset import TokenizedManifestDataset


DETECTION_FIELDNAMES = [
    "session_id",
    "user_id",
    "host_id",
    "attack_probability",
    "prediction",
    "threshold",
    "alert_generated",
    "alert_class",
    "composite_severity",
]


@dataclass(frozen=True)
class Phase3BundlePaths:
    """Resolved artifact paths for a Phase 3 model bundle."""

    classifier: Path
    thresholds: Path
    vocab: Path
    metadata: Path | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested_device)


def resolve_bundle_paths(
    *,
    bundle_dir: str | Path | None = None,
    classifier: str | Path | None = None,
    thresholds: str | Path | None = None,
    vocab: str | Path | None = None,
    metadata: str | Path | None = None,
) -> Phase3BundlePaths:
    """Resolve either a packaged bundle directory or individual artifacts."""
    if bundle_dir:
        resolved_bundle = Path(bundle_dir)
        return Phase3BundlePaths(
            classifier=resolved_bundle / "best_classifier.pt",
            thresholds=resolved_bundle / "calibrated_thresholds.json",
            vocab=resolved_bundle / "vocab.json",
            metadata=resolved_bundle / "model_metadata.json",
        )

    missing = [
        label
        for label, value in {
            "--classifier": classifier,
            "--thresholds": thresholds,
            "--vocab": vocab,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Provide --bundle-dir or all individual artifact paths: "
            + ", ".join(missing)
        )

    return Phase3BundlePaths(
        classifier=Path(classifier),
        thresholds=Path(thresholds),
        vocab=Path(vocab),
        metadata=Path(metadata) if metadata else None,
    )


def require_existing(paths: Phase3BundlePaths) -> None:
    for label, path in {
        "classifier": paths.classifier,
        "thresholds": paths.thresholds,
        "vocab": paths.vocab,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} artifact does not exist: {path}")
    if paths.metadata is not None and not paths.metadata.exists():
        return


def load_threshold(thresholds_path: Path, override: float | None) -> float:
    if override is not None:
        if not 0.0 <= override <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return override

    thresholds = load_json(thresholds_path)
    threshold = thresholds.get("classifier_attack_threshold")
    if threshold is None:
        threshold = thresholds.get("operating_threshold_metrics", {}).get("threshold")
    if threshold is None:
        raise ValueError(
            "Could not find classifier threshold in calibrated_thresholds.json"
        )
    return float(threshold)


def load_classifier(classifier_path: Path, device: torch.device) -> ARGUSClassifier:
    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, ArgusBertConfig):
        config = ArgusBertConfig()

    model = ARGUSClassifier(
        config=config,
        num_classes=int(checkpoint.get("num_classes", 2)),
        freeze_layers=0,
    )
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Classifier checkpoint missing model_state_dict: {classifier_path}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def coerce_events(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        if hasattr(events, "tolist"):
            events = events.tolist()
        elif isinstance(events, tuple):
            events = list(events)
        else:
            return []
    return [dict(event) for event in events if isinstance(event, Mapping)]


def batch_items(
    items: Iterable[dict[str, Any]],
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_manifest_batches(
    manifest_path: Path,
    batch_size: int,
    *,
    num_workers: int,
) -> Iterator[list[dict[str, Any]]]:
    dataset = TokenizedManifestDataset(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    for batch in loader:
        session_ids = batch.get("session_id")
        rows = []
        for row_index in range(batch["input_ids"].shape[0]):
            if isinstance(session_ids, (list, tuple)):
                session_id = session_ids[row_index]
            elif isinstance(session_ids, torch.Tensor):
                session_id = session_ids[row_index].item()
            else:
                session_id = row_index
            rows.append(
                {
                    "session_id": session_id,
                    "user_id": "",
                    "host_id": "",
                    "input_ids": batch["input_ids"][row_index],
                    "attention_mask": batch["attention_mask"][row_index],
                }
            )
        yield rows


class Phase3DetectionService:
    """Load a Phase 3 bundle once and score raw or tokenized sessions."""

    def __init__(
        self,
        paths: Phase3BundlePaths,
        *,
        threshold: float | None = None,
        device: str = "auto",
        max_seq_len: int | None = None,
        anomaly_ceiling: float = 15.0,
        dedup_window_secs: float = 0.0,
        technique_id: str = "T1078",
    ) -> None:
        require_existing(paths)
        self.paths = paths
        self.metadata = (
            load_json(paths.metadata)
            if paths.metadata is not None and paths.metadata.exists()
            else {}
        )
        self.threshold = load_threshold(paths.thresholds, threshold)
        self.device = select_device(device)
        self.max_seq_len = int(max_seq_len or self.metadata.get("max_seq_len") or 16)
        self.anomaly_ceiling = anomaly_ceiling
        self.technique_id = technique_id
        self.model = load_classifier(paths.classifier, self.device)
        self.tokenizer = LogTokenizer(paths.vocab, max_len=self.max_seq_len)
        self.alert_engine = AlertEngine(
            anomaly_ceiling=anomaly_ceiling,
            dedup_window_secs=dedup_window_secs,
        )

    @classmethod
    def from_bundle_dir(
        cls,
        bundle_dir: str | Path,
        **kwargs: Any,
    ) -> "Phase3DetectionService":
        return cls(resolve_bundle_paths(bundle_dir=bundle_dir), **kwargs)

    def encode_session(
        self,
        session: Mapping[str, Any],
        *,
        row_index: int,
    ) -> dict[str, Any]:
        mutable_session = dict(session)
        mutable_session["events"] = coerce_events(mutable_session.get("events", []))
        input_ids, attention_mask = self.tokenizer.tokenize_with_attention_mask(
            mutable_session
        )
        return {
            "session_id": mutable_session.get("session_id", row_index),
            "user_id": mutable_session.get("user_id", ""),
            "host_id": mutable_session.get("host_id", ""),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }

    def iter_jsonl_sessions(self, path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file_obj:
            for row_index, line in enumerate(file_obj):
                stripped = line.strip()
                if not stripped:
                    continue
                session = json.loads(stripped)
                if not isinstance(session, dict):
                    raise ValueError(
                        f"JSONL row must be an object at line {row_index + 1}"
                    )
                yield self.encode_session(session, row_index=row_index)

    def score_sessions(
        self,
        sessions: Iterable[Mapping[str, Any]],
        *,
        threshold: float | None = None,
        technique_id: str | None = None,
    ) -> list[dict[str, Any]]:
        items = [
            self.encode_session(session, row_index=row_index)
            for row_index, session in enumerate(sessions)
        ]
        if not items:
            return []
        return self.score_items(
            items,
            threshold=threshold,
            technique_id=technique_id,
        )

    def score_items(
        self,
        batch: list[dict[str, Any]],
        *,
        threshold: float | None = None,
        technique_id: str | None = None,
    ) -> list[dict[str, Any]]:
        active_threshold = self.threshold if threshold is None else threshold
        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        input_ids = torch.stack([item["input_ids"] for item in batch]).to(self.device)
        attention_mask = torch.stack(
            [item["attention_mask"] for item in batch]
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            attack_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

        rows = []
        active_technique = technique_id or self.technique_id
        for item, attack_probability in zip(batch, attack_probs):
            prediction = "attack" if attack_probability >= active_threshold else "normal"
            confidence = attack_probability if prediction == "attack" else 0.95
            scored = ScoredSession(
                session_id=str(item["session_id"]),
                user_id=str(item.get("user_id", "")),
                host_id=str(item.get("host_id", "")),
                anomaly_score=float(attack_probability) * self.anomaly_ceiling,
                classification=prediction,
                classification_confidence=float(confidence),
                technique_id=active_technique if prediction == "attack" else None,
            )
            alert = self.alert_engine.process_session(scored)
            rows.append(
                {
                    "session_id": scored.session_id,
                    "user_id": scored.user_id,
                    "host_id": scored.host_id,
                    "attack_probability": float(attack_probability),
                    "prediction": prediction,
                    "threshold": active_threshold,
                    "alert_generated": alert is not None,
                    "alert_class": alert.alert_class if alert else "",
                    "composite_severity": alert.composite_severity if alert else "",
                }
            )
        return rows


def write_detection_csv(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=DETECTION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            total += 1
    return total
