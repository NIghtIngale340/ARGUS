"""Phase 2 lesson worksheet: ARGUS-BERT model.

Goal:
Wrap a BERT masked-language-model so it can train on ARGUS log tokens.

Important:
We do not need to invent transformer math from scratch here. The project
already depends on Hugging Face transformers, so the practical first model can
use BertForMaskedLM with ARGUS-specific config values.
"""

# =============================================================================
# Step 1: Import model-related tools.
# =============================================================================
# Hint:
# - Any is used for a flexible return type while learning.
# - nn.Module is the base class for PyTorch models.
# - BertConfig describes the BERT shape.
# - BertForMaskedLM is BERT with a token-prediction head.

from typing import Any

from torch import nn
from transformers import BertConfig, BertForMaskedLM

from src.models.config import ArgusBertConfig


# =============================================================================
# Step 2: Declare the model wrapper class.
# =============================================================================
# Hint:
# This model inherits from nn.Module so PyTorch can train it.

class ArgusBertForMaskedLM(nn.Module):
    """ARGUS-BERT model for masked log modeling."""

    # =========================================================================
    # Step 3: Initialize config and validate it.
    # =========================================================================
    # Hint:
    # If no config is passed, create the default ArgusBertConfig.

    def __init__(self, config: ArgusBertConfig | None = None) -> None:
        super().__init__()

        self.config = config or ArgusBertConfig()
        self.config.validate()

        # =====================================================================
        # Step 4: Convert ArgusBertConfig into Hugging Face BertConfig.
        # =====================================================================
        # Hint:
        # max_position_embeddings should match max_seq_len for this artifact.

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

        # =====================================================================
        # Step 5: Create the real Hugging Face model.
        # =====================================================================
        # Hint:
        # This object contains embeddings, transformer layers, and the MLM head.

        self.model = BertForMaskedLM(self.hf_config)

    # =========================================================================
    # Step 6: Forward inputs through the model.
    # =========================================================================
    # Hint:
    # labels are optional. If labels are passed, Hugging Face returns a loss too.

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        labels: Any | None = None,
    ) -> Any:
        """Run MLM forward pass."""

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
