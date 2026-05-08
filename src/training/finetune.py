"""ARGUS Phase 3.2 — Fine-tune pre-trained BERT for attack classification.

Loads the pre-trained MLM checkpoint, attaches a classification head,
and trains on labeled attack sessions using weighted cross-entropy
to handle the severe class imbalance (500K normal vs 212 attack).
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.models.attack_classifier import ARGUSClassifier
from src.models.config import ArgusBertConfig


class AttackClassificationDataset(Dataset):
    """Binary classification dataset: 0 = normal, 1 = attack.

    Loads tokenized session chunks from manifests produced by the
    ARGUS tokenization pipeline.
    """

    def __init__(
        self,
        normal_manifest_path: str,
        attack_manifest_path: str,
        max_normal: int = 50_000,
        limit_chunks: Optional[int] = None,
    ) -> None:
        self.input_ids: list[Tensor] = []
        self.attention_masks: list[Tensor] = []
        self.labels: list[int] = []

        n_normal = self._load_manifest(
            normal_manifest_path, label=0,
            max_sessions=max_normal, limit_chunks=limit_chunks,
        )

        n_attack = self._load_manifest(
            attack_manifest_path, label=1,
            max_sessions=None, limit_chunks=None,
        )

        print(f"Dataset: {n_normal:,} normal + {n_attack:,} attack = {len(self):,} total")

    def _load_manifest(
        self,
        manifest_path: str,
        label: int,
        max_sessions: Optional[int] = None,
        limit_chunks: Optional[int] = None,
    ) -> int:
        """Load sessions from a chunked manifest file."""
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)

        if isinstance(manifest, dict) and "chunks" in manifest:
            chunk_paths = manifest["chunks"]
            base_dir = Path(manifest_path).parent
        else:
            for item in manifest:
                self.input_ids.append(item["input_ids"])
                self.attention_masks.append(item["attention_mask"])
                self.labels.append(label)
            return len(manifest)

        count = 0
        chunks_to_load = chunk_paths[:limit_chunks] if limit_chunks else chunk_paths
        for chunk_rel in chunks_to_load:
            chunk_path = base_dir / chunk_rel
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            ids_tensor = chunk["input_ids"]
            mask_tensor = chunk["attention_mask"]

            for i in range(ids_tensor.shape[0]):
                if max_sessions and count >= max_sessions:
                    return count
                self.input_ids.append(ids_tensor[i])
                self.attention_masks.append(mask_tensor[i])
                self.labels.append(label)
                count += 1

        return count

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch items into tensors."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


@dataclass
class FinetuneConfig:
    """Fine-tuning hyperparameters."""
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 20
    batch_size: int = 64
    max_normal_sessions: int = 50_000
    freeze_layers: int = 4
    classifier_dropout: float = 0.1
    log_every: int = 50
    eval_every: int = 200
    patience: int = 5
    num_workers: int = 2


def compute_class_weights(labels: list[int], num_classes: int = 2) -> Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run evaluation and return metrics dict."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def train(
    cfg: FinetuneConfig,
    model: ARGUSClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Full fine-tuning loop with early stopping."""
    model.to(device)

    all_labels = [item["label"] for item in train_loader.dataset]
    class_weights = compute_class_weights(all_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1 = 0.0
    patience_counter = 0
    global_step = 0
    history = []

    trainable, total = model.count_trainable_params()
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    print(f"Class weights: normal={class_weights[0]:.3f}, attack={class_weights[1]:.3f}")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg = epoch_loss / epoch_steps
                print(f"  step {global_step}: loss={avg:.4f}")

        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{cfg.epochs} ({elapsed:.1f}s): "
            f"train_loss={epoch_loss/max(epoch_steps,1):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f} "
            f"(TP={val_metrics['tp']} FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']} TN={val_metrics['tn']})"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(epoch_steps, 1), 6),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in val_metrics.items()},
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = output_dir / "best_classifier.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "num_classes": model.num_classes,
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  ✓ New best F1={best_f1:.4f}, saved → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping after {cfg.patience} epochs without improvement.")
                break

    history_path = output_dir / "finetune_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Training history → {history_path}")

    return {"best_f1": best_f1, "epochs_trained": epoch + 1, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ARGUS-BERT for attack classification")
    parser.add_argument("--normal-manifest", required=True, help="Path to normal sessions manifest .pt")
    parser.add_argument("--attack-manifest", required=True, help="Path to attack sessions manifest .pt")
    parser.add_argument("--checkpoint", required=True, help="Path to pre-trained MLM checkpoint .pt")
    parser.add_argument("--out", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-normal", type=int, default=50_000)
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AttackClassificationDataset(
        normal_manifest_path=args.normal_manifest,
        attack_manifest_path=args.attack_manifest,
        max_normal=args.max_normal,
        limit_chunks=args.limit_chunks,
    )

    labels = dataset.labels
    attack_indices = [i for i, l in enumerate(labels) if l == 1]
    normal_indices = [i for i, l in enumerate(labels) if l == 0]

    n_val_attack = max(1, int(len(attack_indices) * args.val_split))
    n_val_normal = max(1, int(len(normal_indices) * args.val_split))

    rng = np.random.RandomState(42)
    rng.shuffle(attack_indices)
    rng.shuffle(normal_indices)

    val_indices = attack_indices[:n_val_attack] + normal_indices[:n_val_normal]
    train_indices = attack_indices[n_val_attack:] + normal_indices[n_val_normal:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_labels = [labels[i] for i in train_indices]
    sample_weights = []
    attack_weight = len(train_labels) / max(sum(train_labels), 1)
    normal_weight = len(train_labels) / max(len(train_labels) - sum(train_labels), 1)
    for l in train_labels:
        sample_weights.append(attack_weight if l == 1 else normal_weight)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = ARGUSClassifier(
        num_classes=2,
        freeze_layers=args.freeze_layers,
        classifier_dropout=0.1,
    )
    model.load_pretrained_bert(args.checkpoint)
    print("Loaded pre-trained BERT weights.")

    cfg = FinetuneConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_normal_sessions=args.max_normal,
        freeze_layers=args.freeze_layers,
        log_every=args.batch_size,
    )

    output_dir = Path(args.out)
    train_result = train(cfg, model, train_loader, val_loader, device, output_dir)

    summary_path = output_dir / "finetune_summary.json"
    summary_path.write_text(json.dumps(train_result, indent=2))
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
