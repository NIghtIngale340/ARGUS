"""ARGUS-BERT wrapper for masked-language modeling."""

from typing import Any

from torch import nn
from transformers import BertConfig, BertForMaskedLM

from src.models.config import ArgusBertConfig

class ArgusBertForMaskedLM(nn.Module):
    """ARGUS-BERT model for masked log modeling."""

    def __init__(self, config: ArgusBertConfig | None = None) -> None:
        super().__init__()

        self.config = config or ArgusBertConfig()
        self.config.validate()

        self.hf_config = BertConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            max_position_embeddings=self.config.max_seq_len,
            pad_token_id=self.config.pad_token_id,
        )

        self.model = BertForMaskedLM(self.hf_config)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        labels: Any | None = None,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
