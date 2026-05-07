"""Tests for ARGUS MLM masking."""

import pytest

from src.training.mlm_collator import ArgusMLMCollator, MLMMaskingConfig


def test_mlm_collator_returns_masked_inputs_attention_mask_and_labels() -> None:
    torch = pytest.importorskip("torch")
    collator = ArgusMLMCollator(
        MLMMaskingConfig(
            mask_probability=1.0,
            mask_token_probability=1.0,
            random_token_probability=0.0,
            unchanged_probability=0.0,
        )
    )
    examples = [
        {
            "input_ids": torch.tensor([0, 10, 1, 3], dtype=torch.int16),
            "attention_mask": torch.tensor([True, True, True, False]),
        },
        {
            "input_ids": torch.tensor([0, 11, 1, 3], dtype=torch.int16),
            "attention_mask": torch.tensor([True, True, True, False]),
        },
    ]

    batch = collator(examples)

    assert batch["input_ids"].tolist() == [[0, 2, 1, 3], [0, 2, 1, 3]]
    assert batch["attention_mask"].tolist() == [
        [True, True, True, False],
        [True, True, True, False],
    ]
    assert batch["labels"].tolist() == [
        [-100, 10, -100, -100],
        [-100, 11, -100, -100],
    ]


def test_mlm_collator_random_tokens_keep_compact_input_dtype() -> None:
    torch = pytest.importorskip("torch")
    collator = ArgusMLMCollator(
        MLMMaskingConfig(
            mask_probability=1.0,
            mask_token_probability=0.0,
            random_token_probability=1.0,
            unchanged_probability=0.0,
        )
    )
    examples = [
        {
            "input_ids": torch.tensor([0, 10, 1, 3], dtype=torch.int16),
            "attention_mask": torch.tensor([True, True, True, False]),
        }
    ]

    batch = collator(examples)

    assert batch["input_ids"].dtype == torch.int16
    assert batch["labels"].tolist() == [[-100, 10, -100, -100]]


def test_mlm_collator_forces_one_target_for_tiny_batches() -> None:
    torch = pytest.importorskip("torch")
    collator = ArgusMLMCollator(MLMMaskingConfig(mask_probability=0.0))
    examples = [
        {
            "input_ids": torch.tensor([0, 10, 1, 3], dtype=torch.int16),
            "attention_mask": torch.tensor([True, True, True, False]),
        }
    ]

    batch = collator(examples)

    assert (batch["labels"] != -100).sum().item() == 1
