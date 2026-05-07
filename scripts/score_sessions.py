"""Score tokenized sessions with pretrained ARGUS-BERT MLM loss."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.training.dataset import TokenizedManifestDataset
from src.inference.anomaly_scorer import MLMAnomalyScorer


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Score ARGUS tokenized sessions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10_000)
    return parser


def build_dataloader(args: Namespace, *, device: torch.device) -> DataLoader:
    """Build DataLoader for streaming tokenized sessions."""
    dataset = TokenizedManifestDataset(
        args.manifest,
        limit_chunks=args.limit_chunks,
        limit_sessions=args.limit_sessions,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()
    if parsed.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if parsed.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if parsed.log_every <= 0:
        raise ValueError("--log-every must be > 0")

    scorer = MLMAnomalyScorer(
        parsed.checkpoint,
        device=parsed.device,
    )
    dataloader = build_dataloader(parsed, device=scorer.device)

    output_path = Path(parsed.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_index = 0
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["row_index", "session_id", "anomaly_score"])

        for batch in dataloader:
            scores = scorer.score_batch(batch).detach().cpu().tolist()
            session_ids = batch.get("session_id")
            for batch_index, score in enumerate(scores):
                if isinstance(session_ids, (list, tuple)):
                    session_id = session_ids[batch_index]
                elif isinstance(session_ids, torch.Tensor):
                    session_id = session_ids[batch_index].item()
                else:
                    session_id = row_index
                writer.writerow([row_index, session_id, float(score)])
                row_index += 1
                if row_index % parsed.log_every == 0:
                    print(f"Scored {row_index:,} session(s)...", flush=True)

    print(f"Wrote {row_index:,} score(s) to {output_path}")


if __name__ == "__main__":
    main()
