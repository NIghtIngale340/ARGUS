"""Entrypoint for ARGUS-BERT MLM smoke runs."""

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from src.models.argus_bert import ArgusBertForMaskedLM
from src.training.dataset import TokenizedManifestDataset
from src.training.mlm_collator import ArgusMLMCollator
def build_parser() -> ArgumentParser:
    """Create the pretraining CLI parser."""

    parser = ArgumentParser(description="ARGUS-BERT MLM pretraining scaffold")

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--vocab-path")
    parser.add_argument("--config", default="configs/argus_bert_phase2.yaml")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--batch-size", type=int, default=8)

    return parser
def build_train_dataset(args: Namespace) -> TokenizedManifestDataset:
    """Create the train dataset from parsed CLI args."""

    return TokenizedManifestDataset(
        args.train_manifest,
        limit_chunks=args.limit_chunks,
        limit_sessions=args.limit_sessions,
    )
def main(args: Namespace | None = None) -> None:
    """Run a single-batch smoke pass."""

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
if __name__ == "__main__":
    main()
