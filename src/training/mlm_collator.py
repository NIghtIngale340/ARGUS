"""Phase 2 lesson worksheet: Masked Log Modeling collator.

Goal:
Build the batch-preparation object that turns tokenized log sessions into
masked-language-model training batches.

What this file should eventually do:
1. Receive examples from TokenizedManifestDataset.
2. Stack them into one batch.
3. Create labels for MLM training.
4. Replace some token ids with [MASK], random ids, or keep them unchanged.

Important:
The collator is where the model's learning task is created. The dataset only
reads tokens. The collator decides which tokens the model must predict.
"""

# =============================================================================
# Step 1: Import basic Python typing and dataclass tools.
# =============================================================================
# Hint:
# - dataclass creates a small config object.
# - Sequence means an ordered group like a list or tuple.
# - Mapping means dictionary-like data.

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# =============================================================================
# Step 2: Import PyTorch.
# =============================================================================
# Hint:
# The collator needs torch because it stacks tensors and creates random masks.

import torch


# =============================================================================
# Step 3: Define the masking configuration.
# =============================================================================
# Hint:
# These defaults match the verified Phase 2 artifact and the YAML config.

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


# =============================================================================
# Step 4: Create the collator class.
# =============================================================================
# Hint:
# A collator is a callable object. PyTorch DataLoader can call it to turn
# multiple examples into one batch.

class ArgusMLMCollator:
    """Create MLM inputs and labels for ARGUS-BERT."""

    # =========================================================================
    # Step 5: Store the config.
    # =========================================================================
    # Hint:
    # If no config is passed, create the default MLMMaskingConfig.

    def __init__(self, config: MLMMaskingConfig | None = None) -> None:
        self.config = config or MLMMaskingConfig()

    # =========================================================================
    # Step 6: Start the callable batch function.
    # =========================================================================
    # Hint:
    # examples is the list of rows yielded by TokenizedManifestDataset.

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Return a batch with masked input_ids, attention_mask, and labels."""

        if not examples:
            raise ValueError("ArgusMLMCollator requires at least one example")

        # =====================================================================
        # Step 7: Stack individual examples into batch tensors.
        # =====================================================================
        # Hint:
        # Each example has input_ids shape [seq_len]. torch.stack turns many
        # [seq_len] tensors into one [batch_size, seq_len] tensor.

        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack(
            [example["attention_mask"] for example in examples]
        )

        # =====================================================================
        # Step 8: Create labels from the original input ids.
        # =====================================================================
        # Hint:
        # labels must keep the original token ids for masked positions.
        # Later, unmasked positions are changed to ignore_index.

        labels = input_ids.clone()

        # =====================================================================
        # Step 9: Decide which positions are allowed to be masked.
        # =====================================================================
        # Hint:
        # Padding tokens should never be masked.

        non_padding = input_ids != self.config.pad_token_id
        active_tokens = attention_mask.bool()
        maskable_positions = non_padding & active_tokens

        # =====================================================================
        # Step 10: Randomly choose the positions that become MLM targets.
        # =====================================================================
        # Hint:
        # torch.rand creates random floats between 0 and 1.
        # A token becomes selected when the random value is below 0.15.

        random_scores = torch.rand(input_ids.shape, device=input_ids.device)
        selected_for_mlm = random_scores < self.config.mask_probability
        selected_for_mlm = selected_for_mlm & maskable_positions

        # =====================================================================
        # Step 11: Ignore labels for positions that are not selected.
        # =====================================================================
        # Hint:
        # PyTorch losses usually ignore labels equal to -100.

        labels[~selected_for_mlm] = self.config.ignore_index

        # =====================================================================
        # Step 12: Create the masked version of input_ids.
        # =====================================================================
        # Hint:
        # We clone because input_ids is the original batch. masked_input_ids is
        # the modified version that goes into the model.

        masked_input_ids = input_ids.clone()

        # =====================================================================
        # Step 13: Apply the 80 percent [MASK] replacement rule.
        # =====================================================================
        # Hint:
        # Of the selected positions, 80 percent become mask_token_id.

        replacement_scores = torch.rand(input_ids.shape, device=input_ids.device)
        replace_with_mask = (
            replacement_scores < self.config.mask_token_probability
        ) & selected_for_mlm
        masked_input_ids[replace_with_mask] = self.config.mask_token_id

        # =====================================================================
        # Step 14: Apply the 10 percent random-token replacement rule.
        # =====================================================================
        # Hint:
        # Random token ids must be between 0 and vocab_size - 1.

        random_token_scores = torch.rand(input_ids.shape, device=input_ids.device)
        replace_with_random = (
            random_token_scores < 0.5
        ) & selected_for_mlm & ~replace_with_mask
        random_token = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
        )
        masked_input_ids[replace_with_random] = random_token[replace_with_random]

        # =====================================================================
        # Step 15: Return the batch dictionary.
        # =====================================================================
        # Hint:
        # The unchanged 10 percent needs no code because masked_input_ids already
        # contains the original token ids at those selected positions.

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
