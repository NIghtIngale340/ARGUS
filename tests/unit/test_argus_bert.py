"""TODO tests for the ARGUS-BERT MLM model scaffold."""

import pytest

from src.models.argus_bert import ArgusBertForMaskedLM
from src.models.config import ArgusBertConfig


pytestmark = pytest.mark.skip(reason="Phase 2 model scaffold only")


def test_argus_bert_config_matches_verified_artifact() -> None:
    config = ArgusBertConfig()

    assert config.vocab_size == 1233
    assert config.max_seq_len == 16


def test_argus_bert_forward_returns_mlm_logits() -> None:
    model = ArgusBertForMaskedLM()

    assert model.config.vocab_size == 1233
    # TODO: assert logits shape is [batch, 16, 1233] once forward is implemented.
