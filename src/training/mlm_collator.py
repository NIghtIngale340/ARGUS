"""Masked language modeling collator for ARGUS tokens."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class MLMMaskingConfig:
    """Masking defaults for the verified ARGUS tokenized artifact."""

    vocab_size: int = 1233
    mask_token_id: int = 2
    pad_token_id: int = 3
    mask_probability: float = 0.15
    mask_token_probability: float = 0.8
    random_token_probability: float = 0.1
    unchanged_probability: float = 0.1
    ignore_index: int = -100


class ArgusMLMCollator:
    """Create MLM inputs and labels for ARGUS-BERT."""

    def __init__(self, config: MLMMaskingConfig | None = None) -> None:
        self.config = config or MLMMaskingConfig()

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not examples:
            raise ValueError("ArgusMLMCollator requires at least one example")

        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack(
            [example["attention_mask"] for example in examples]
        )


        labels = input_ids.clone()


        non_padding = input_ids != self.config.pad_token_id
        active_tokens = attention_mask.bool()
        maskable_positions = non_padding & active_tokens


        random_scores = torch.rand(input_ids.shape, device=input_ids.device)
        selected_for_mlm = random_scores < self.config.mask_probability
        selected_for_mlm = selected_for_mlm & maskable_positions


        labels[~selected_for_mlm] = self.config.ignore_index


        masked_input_ids = input_ids.clone()


        replacement_scores = torch.rand(input_ids.shape, device=input_ids.device)
        replace_with_mask = (
            replacement_scores < self.config.mask_token_probability
        ) & selected_for_mlm
        masked_input_ids[replace_with_mask] = self.config.mask_token_id


        random_token_scores = torch.rand(input_ids.shape, device=input_ids.device)
        replace_with_random = (
            random_token_scores < 0.5
        ) & selected_for_mlm & ~replace_with_mask
        random_tokens = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        masked_input_ids[replace_with_random] = random_tokens[replace_with_random]

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
