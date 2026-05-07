"""ARGUS-BERT masked-language-model pretraining entrypoint."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import random
import time
from typing import Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.models.argus_bert import ArgusBertForMaskedLM
from src.training.dataset import TokenizedManifestDataset
from src.training.mlm_collator import ArgusMLMCollator


def build_parser() -> ArgumentParser:
    """Create the pretraining CLI parser."""

    parser = ArgumentParser(description="ARGUS-BERT MLM pretraining")

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--vocab-path")
    parser.add_argument("--config", default="configs/argus_bert_phase2.yaml")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--output-dir", default="/kaggle/working/argus_mlm_checkpoints")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-pin-memory", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


def build_dataset(
    manifest_path: str | Path,
    args: Namespace,
) -> TokenizedManifestDataset:
    return TokenizedManifestDataset(
        manifest_path,
        limit_chunks=args.limit_chunks,
        limit_sessions=args.limit_sessions,
    )


def build_dataloader(
    dataset: TokenizedManifestDataset,
    args: Namespace,
    *,
    device: torch.device,
) -> DataLoader:
    pin_memory = device.type == "cuda" and not args.no_pin_memory
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=ArgusMLMCollator(),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: str | Path,
    *,
    model: ArgusBertForMaskedLM,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    args: Namespace,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"checkpoint_step_{global_step:06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        },
        checkpoint_path,
    )
    latest_path = output_path / "latest.pt"
    torch.save({"checkpoint": str(checkpoint_path)}, latest_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: ArgusBertForMaskedLM,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


@torch.no_grad()
def evaluate(
    model: ArgusBertForMaskedLM,
    dataloader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(dataloader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        batch = move_batch_to_device(batch, device)
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def train(args: Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")

    set_seed(args.seed)
    device = select_device(args.device)

    train_dataset = build_dataset(args.train_manifest, args)
    train_dataloader = build_dataloader(train_dataset, args, device=device)

    val_dataloader = None
    if args.val_manifest:
        val_dataset = build_dataset(args.val_manifest, args)
        val_dataloader = build_dataloader(val_dataset, args, device=device)

    model = ArgusBertForMaskedLM().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps)

    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        start_epoch, global_step = load_checkpoint(
            args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resumed checkpoint {args.resume_checkpoint} at step {global_step}.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_updates = 0
    started_at = time.time()

    print(
        "Training setup: "
        f"device={device}, batch_size={args.batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps}, "
        f"num_workers={args.num_workers}, max_steps={args.max_steps}"
    )

    stop_training = False
    for epoch in range(start_epoch, args.max_epochs):
        for batch_index, batch in enumerate(train_dataloader, start=1):
            batch = move_batch_to_device(batch, device)
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {global_step}: {loss}")

            (loss / args.gradient_accumulation_steps).backward()

            if batch_index % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += float(loss.detach().cpu())
            running_updates += 1

            if global_step % args.log_every == 0 or global_step == 1:
                elapsed = max(time.time() - started_at, 1e-6)
                avg_loss = running_loss / max(running_updates, 1)
                print(
                    f"step={global_step:,} epoch={epoch + 1} "
                    f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.6g} "
                    f"steps_per_sec={global_step / elapsed:.3f}",
                    flush=True,
                )
                running_loss = 0.0
                running_updates = 0

            if val_dataloader is not None and args.eval_every > 0 and global_step % args.eval_every == 0:
                val_loss = evaluate(
                    model,
                    val_dataloader,
                    device=device,
                    max_batches=args.max_val_batches,
                )
                print(f"eval step={global_step:,} val_loss={val_loss:.4f}", flush=True)

            if args.save_every > 0 and global_step % args.save_every == 0:
                checkpoint_path = save_checkpoint(
                    args.output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                )
                print(f"saved checkpoint: {checkpoint_path}", flush=True)

            if global_step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break

    final_path = save_checkpoint(
        args.output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch if "epoch" in locals() else 0,
        global_step=global_step,
        args=args,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


def main(args: Namespace | None = None) -> None:
    parsed = args or build_parser().parse_args()
    train(parsed)


if __name__ == "__main__":
    main()
