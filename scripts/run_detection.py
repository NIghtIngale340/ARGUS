"""Run ARGUS Phase 3 detection from a packaged model bundle.

Inputs can be either:
- raw session JSONL with an ``events`` list per row, or
- an existing tokenized manifest produced by the ARGUS tokenizer.

Example:
    python -m scripts.run_detection \
        --bundle-dir /kaggle/working/argus_phase3_model_bundle \
        --sessions-jsonl /kaggle/working/new_sessions.jsonl \
        --out /kaggle/working/argus_detection_results.csv
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.phase3_detection import (
    DETECTION_FIELDNAMES,
    Phase3DetectionService,
    batch_items,
    iter_manifest_batches,
    resolve_bundle_paths,
)


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
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--technique-id", default="T1078")
    parser.add_argument("--anomaly-ceiling", type=float, default=15.0)
    parser.add_argument("--dedup-window-secs", type=float, default=0.0)
    parser.add_argument("--redis-url", help="Use Redis-backed UEBA risk state")
    parser.add_argument("--redis-key-prefix", default="argus:ueba")
    return parser


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()
    if parsed.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if parsed.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    paths = resolve_bundle_paths(
        bundle_dir=parsed.bundle_dir,
        classifier=parsed.classifier,
        thresholds=parsed.thresholds,
        vocab=parsed.vocab,
        metadata=parsed.metadata,
    )
    service = Phase3DetectionService(
        paths,
        threshold=parsed.threshold,
        device=parsed.device,
        max_seq_len=parsed.max_seq_len,
        anomaly_ceiling=parsed.anomaly_ceiling,
        dedup_window_secs=parsed.dedup_window_secs,
        technique_id=parsed.technique_id,
        redis_url=getattr(parsed, "redis_url", None),
        redis_key_prefix=getattr(parsed, "redis_key_prefix", "argus:ueba"),
    )

    if parsed.sessions_jsonl:
        batches = batch_items(
            service.iter_jsonl_sessions(Path(parsed.sessions_jsonl)),
            parsed.batch_size,
        )
    else:
        batches = iter_manifest_batches(
            Path(parsed.manifest),
            parsed.batch_size,
            num_workers=parsed.num_workers,
        )

    output_path = Path(parsed.out)
    total = 0
    alerts = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from csv import DictWriter

    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = DictWriter(file_obj, fieldnames=DETECTION_FIELDNAMES)
        writer.writeheader()
        for batch in batches:
            rows = service.score_items(batch)
            for row in rows:
                writer.writerow(row)
                total += 1
                alerts += int(row["alert_generated"])

    print(f"Wrote {total:,} detection row(s) to {output_path}")
    print(f"Threshold: {service.threshold:.6f}")
    print(f"Alerts generated: {alerts:,}")


if __name__ == "__main__":
    main()
