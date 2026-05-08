"""Run ARGUS Phase 3 detection from a packaged model bundle.

Inputs can be either:
- raw session JSONL with an ``events`` list per row, or
- an existing tokenized manifest produced by the ARGUS tokenizer.

Example:
    python scripts/run_detection.py \
        --bundle-dir /kaggle/working/argus_phase3_model_bundle \
        --sessions-jsonl /kaggle/working/new_sessions.jsonl \
        --out /kaggle/working/argus_detection_results.csv
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.alert_engine import AlertEngine, ScoredSession
from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig
from src.parsing.log_tokenizer import LogTokenizer
from src.training.dataset import TokenizedManifestDataset


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


def resolve_bundle_paths(args: Namespace) -> dict[str, Path]:
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
        return {
            "classifier": bundle_dir / "best_classifier.pt",
            "thresholds": bundle_dir / "calibrated_thresholds.json",
            "vocab": bundle_dir / "vocab.json",
            "metadata": bundle_dir / "model_metadata.json",
        }

    missing = [
        name for name, value in {
            "--classifier": args.classifier,
            "--thresholds": args.thresholds,
            "--vocab": args.vocab,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Provide --bundle-dir or all individual artifact paths: "
            + ", ".join(missing)
        )
    return {
        "classifier": Path(args.classifier),
        "thresholds": Path(args.thresholds),
        "vocab": Path(args.vocab),
        "metadata": Path(args.metadata) if args.metadata else Path(),
    }


def require_existing(paths: Mapping[str, Path]) -> None:
    for label, path in paths.items():
        if label == "metadata" and not str(path):
            continue
        if label == "metadata" and not path.exists():
            continue
        if not path.exists():
            raise FileNotFoundError(f"{label} artifact does not exist: {path}")


def load_threshold(thresholds_path: Path, override: float | None) -> float:
    if override is not None:
        if not 0.0 <= override <= 1.0:
            raise ValueError("--threshold must be in [0, 1]")
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


def iter_jsonl_sessions(
    path: Path,
    tokenizer: LogTokenizer,
) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        for row_index, line in enumerate(file_obj):
            stripped = line.strip()
            if not stripped:
                continue
            session = json.loads(stripped)
            if not isinstance(session, dict):
                raise ValueError(f"JSONL row must be an object at line {row_index + 1}")
            session["events"] = coerce_events(session.get("events", []))
            input_ids, attention_mask = tokenizer.tokenize_with_attention_mask(session)
            yield {
                "session_id": session.get("session_id", row_index),
                "user_id": session.get("user_id", ""),
                "host_id": session.get("host_id", ""),
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            }


def batch_items(items: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
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
            rows.append({
                "session_id": session_id,
                "user_id": "",
                "host_id": "",
                "input_ids": batch["input_ids"][row_index],
                "attention_mask": batch["attention_mask"][row_index],
            })
        yield rows


def score_batch(
    model: ARGUSClassifier,
    batch: list[dict[str, Any]],
    *,
    device: torch.device,
    threshold: float,
    alert_engine: AlertEngine,
    anomaly_ceiling: float,
    technique_id: str,
) -> list[dict[str, Any]]:
    input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
    attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        attack_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

    rows = []
    for item, attack_probability in zip(batch, attack_probs):
        prediction = "attack" if attack_probability >= threshold else "normal"
        confidence = attack_probability if prediction == "attack" else 0.95
        scored = ScoredSession(
            session_id=str(item["session_id"]),
            user_id=str(item.get("user_id", "")),
            host_id=str(item.get("host_id", "")),
            anomaly_score=float(attack_probability) * anomaly_ceiling,
            classification=prediction,
            classification_confidence=float(confidence),
            technique_id=technique_id if prediction == "attack" else None,
        )
        alert = alert_engine.process_session(scored)
        rows.append({
            "session_id": scored.session_id,
            "user_id": scored.user_id,
            "host_id": scored.host_id,
            "attack_probability": float(attack_probability),
            "prediction": prediction,
            "threshold": threshold,
            "alert_generated": alert is not None,
            "alert_class": alert.alert_class if alert else "",
            "composite_severity": alert.composite_severity if alert else "",
        })
    return rows


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run ARGUS Phase 3 detection")
    parser.add_argument("--bundle-dir", help="Directory from package_phase3_model_bundle.py")
    parser.add_argument("--classifier", help="Path to best_classifier.pt")
    parser.add_argument("--thresholds", help="Path to calibrated_thresholds.json")
    parser.add_argument("--vocab", help="Path to vocab.json")
    parser.add_argument("--metadata", help="Path to model_metadata.json")

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sessions-jsonl", help="Raw session JSONL input")
    inputs.add_argument("--manifest", help="Tokenized session manifest input")

    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--threshold", type=float, help="Override classifier threshold")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--technique-id", default="T1078")
    parser.add_argument("--anomaly-ceiling", type=float, default=15.0)
    parser.add_argument("--dedup-window-secs", type=float, default=0.0)
    return parser


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()
    if parsed.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if parsed.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    paths = resolve_bundle_paths(parsed)
    require_existing(paths)
    threshold = load_threshold(paths["thresholds"], parsed.threshold)
    device = select_device(parsed.device)

    model = load_classifier(paths["classifier"], device)
    tokenizer = LogTokenizer(paths["vocab"], max_len=parsed.max_seq_len)
    alert_engine = AlertEngine(
        anomaly_ceiling=parsed.anomaly_ceiling,
        dedup_window_secs=parsed.dedup_window_secs,
    )

    if parsed.sessions_jsonl:
        batches = batch_items(
            iter_jsonl_sessions(Path(parsed.sessions_jsonl), tokenizer),
            parsed.batch_size,
        )
    else:
        batches = iter_manifest_batches(
            Path(parsed.manifest),
            parsed.batch_size,
            num_workers=parsed.num_workers,
        )

    output_path = Path(parsed.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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

    total = 0
    alerts = 0
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for batch in batches:
            rows = score_batch(
                model,
                batch,
                device=device,
                threshold=threshold,
                alert_engine=alert_engine,
                anomaly_ceiling=parsed.anomaly_ceiling,
                technique_id=parsed.technique_id,
            )
            for row in rows:
                writer.writerow(row)
                total += 1
                alerts += int(row["alert_generated"])

    print(f"Wrote {total:,} detection row(s) to {output_path}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Alerts generated: {alerts:,}")


if __name__ == "__main__":
    main()
