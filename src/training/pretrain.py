"""Phase 2 lesson worksheet: pretraining entrypoint.

Goal:
Wire the dataset, collator, model, and DataLoader together for a tiny MLM
smoke run before any full Kaggle training job.

This file is the command-line entrypoint. The other files define pieces. This
file decides how those pieces are created and connected.
"""

# =============================================================================
# Step 1: Import CLI tools.
# =============================================================================
# Hint:
# ArgumentParser builds command-line flags. Namespace stores parsed values.

from argparse import ArgumentParser, Namespace


# =============================================================================
# Step 2: Import PyTorch DataLoader.
# =============================================================================
# Hint:
# DataLoader reads from the dataset and uses the collator to create batches.

from torch.utils.data import DataLoader


# =============================================================================
# Step 3: Import Phase 2 project pieces.
# =============================================================================
# Hint:
# These are the dataset, collator, and model files you are learning.

from src.models.argus_bert import ArgusBertForMaskedLM
from src.training.dataset import TokenizedManifestDataset
from src.training.mlm_collator import ArgusMLMCollator


# =============================================================================
# Step 4: Build the command-line parser.
# =============================================================================
# Hint:
# The smoke script will call this parser when you run it from terminal/Kaggle.

def build_parser() -> ArgumentParser:
    """Create the Phase 2 pretraining CLI parser."""

    parser = ArgumentParser(
        description="ARGUS-BERT MLM pretraining scaffold"
    )

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--vocab-path")
    parser.add_argument("--config", default="configs/argus_bert_phase2.yaml")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--batch-size", type=int, default=8)

    return parser

# =============================================================================
# Step 5: Create the dataset from CLI values.
# =============================================================================
# Hint:
# We start with train only. Validation can be wired after the smoke path works.

def build_train_dataset(args: Namespace) -> TokenizedManifestDataset:
    """Create the train dataset from parsed CLI args."""

    return TokenizedManifestDataset(
        args.train_manifest,
        limit_chunks=args.limit_chunks,
        limit_sessions=args.limit_sessions,
    )


# =============================================================================
# Step 6: Build the tiny smoke-run objects.
# =============================================================================
# Hint:
# This does not train yet. It proves the dataset and collator can create a batch
# and the model can run one forward pass.

def main(args: Namespace | None = None) -> None:
    """Wire dataset, collator, model, and a tiny smoke batch."""

    parsed = args or build_parser().parse_args()

    dataset = build_train_dataset(parsed)
    collator = ArgusMLMCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=parsed.batch_size,
        collate_fn=collator,
    )

    model = ArgusBertForMaskedLM()

    first_batch = next(iter(dataloader))
    output = model(
        input_ids=first_batch["input_ids"],
        attention_mask=first_batch["attention_mask"],
        labels=first_batch["labels"],
    )

    print("Smoke run loss:", output.loss)


# =============================================================================
# Step 7: Make the file runnable as a script.
# =============================================================================
# Hint:
# This block runs main() only when this file is executed directly.

if __name__ == "__main__":
    main()
