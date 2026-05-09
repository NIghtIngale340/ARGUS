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
from src.training.mitre_dataset import MITREClassificationDataset, collate_mitre_batch


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

    num_classes = int(getattr(model, "num_classes", max(all_labels + all_preds) + 1 if n else 2))

    if num_classes == 2:
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
            "macro_f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    per_class = []
    for class_id in range(num_classes):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == class_id and l == class_id)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == class_id and l != class_id)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != class_id and l == class_id)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        support = sum(1 for label in all_labels if label == class_id)
        per_class.append({
            "class_id": class_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(n, 1)
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_f1": macro_f1,
        "per_class": per_class,
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

    all_labels = [int(item["label"]) for item in train_loader.dataset]
    class_weights = compute_class_weights(all_labels, num_classes=model.num_classes).to(device)
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
    weights_text = ", ".join(
        f"class_{index}={float(weight):.3f}"
        for index, weight in enumerate(class_weights.detach().cpu())
    )
    print(f"Class weights: {weights_text}")

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
            f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f}"
        )
        if model.num_classes == 2:
            print(
                f"  Confusion: TP={val_metrics['tp']} FP={val_metrics['fp']} "
                f"FN={val_metrics['fn']} TN={val_metrics['tn']}"
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
                "class_names": getattr(model, "class_names", None),
                "label_to_id": getattr(model, "label_to_id", None),
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
    parser.add_argument("--normal-manifest", help="Path to normal sessions manifest .pt")
    parser.add_argument("--attack-manifest", help="Path to attack sessions manifest .pt")
    parser.add_argument("--mitre-manifest", help="Tokenized manifest for MITRE label-driven training")
    parser.add_argument("--mitre-labels", help="MITRE labels JSONL/CSV from validate_mitre_labels.py")
    parser.add_argument("--mitre-train-split", default="train")
    parser.add_argument("--mitre-val-split", default="val")
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="Explicit class name order. Repeat, e.g. normal, T1078, T1021.",
    )
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

    mitre_mode = bool(args.mitre_labels or args.mitre_manifest)
    class_names = args.class_names
    num_classes = 2

    if mitre_mode:
        if not args.mitre_labels or not args.mitre_manifest:
            raise ValueError("--mitre-labels and --mitre-manifest must be provided together")
        train_dataset = MITREClassificationDataset(
            args.mitre_manifest,
            args.mitre_labels,
            split=args.mitre_train_split,
            class_names=class_names,
            limit_chunks=args.limit_chunks,
        )
        val_dataset = MITREClassificationDataset(
            args.mitre_manifest,
            args.mitre_labels,
            split=args.mitre_val_split,
            class_names=train_dataset.class_names,
            limit_chunks=args.limit_chunks,
        )
        labels = train_dataset.labels
        num_classes = len(train_dataset.class_names)
        collate = collate_mitre_batch
        print(f"MITRE classes: {train_dataset.class_names}")
    else:
        if not args.normal_manifest or not args.attack_manifest:
            raise ValueError("--normal-manifest and --attack-manifest are required for binary mode")
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
        labels = [dataset.labels[i] for i in train_indices]
        collate = collate_fn

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_labels = [int(item["label"]) for item in train_dataset]
    sample_weights = []
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_sample_weights = len(train_labels) / (num_classes * class_counts)
    for label in train_labels:
        sample_weights.append(float(class_sample_weights[label]))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = ARGUSClassifier(
        num_classes=num_classes,
        freeze_layers=args.freeze_layers,
        classifier_dropout=0.1,
    )
    if mitre_mode:
        model.class_names = train_dataset.class_names
        model.label_to_id = train_dataset.label_to_id
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
