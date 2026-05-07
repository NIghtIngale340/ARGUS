"""Tests for ARGUS MLM masking."""

import pytest

from src.training.mlm_collator import ArgusMLMCollator


pytestmark = pytest.mark.skip(reason="Phase 2 MLM collator scaffold only")


def test_mlm_collator_returns_masked_inputs_attention_mask_and_labels() -> None:
    collator = ArgusMLMCollator()

    assert collator is not None


def test_mlm_collator_never_masks_padding_tokens() -> None:
    collator = ArgusMLMCollator()

    assert collator is not None
