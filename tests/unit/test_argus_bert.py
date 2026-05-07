"""Tests for the ARGUS-BERT MLM model."""

import pytest

from src.models.argus_bert import ArgusBertForMaskedLM
from src.models.config import ArgusBertConfig


def test_argus_bert_config_matches_verified_artifact() -> None:
    config = ArgusBertConfig()

    assert config.vocab_size == 1233
    assert config.max_seq_len == 16
    assert config.num_hidden_layers == 6
    assert config.num_attention_heads == 8


def test_argus_bert_forward_returns_mlm_logits_for_compact_token_ids() -> None:
    torch = pytest.importorskip("torch")
    model = ArgusBertForMaskedLM()

    input_ids = torch.tensor([[0, 10, 1, 3], [0, 11, 1, 3]], dtype=torch.int16)
    attention_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    labels = torch.tensor(
        [[-100, 10, -100, -100], [-100, 11, -100, -100]],
        dtype=torch.int16,
    )

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert tuple(output.logits.shape) == (2, 4, 1233)
    assert output.loss is not None
