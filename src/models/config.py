"""Phase 2 lesson worksheet: ARGUS-BERT configuration.

Goal:
Create one Python config object that describes the model shape.

Why this file exists:
The YAML config stores settings for runs. This Python config stores the values
the model code needs when constructing ARGUS-BERT.
"""

# =============================================================================
# Step 1: Import dataclass.
# =============================================================================
# Hint:
# dataclass is useful for simple classes that mainly store values.

from dataclasses import dataclass


# =============================================================================
# Step 2: Define the config class.
# =============================================================================
# Hint:
# frozen=True means code should not mutate the config after it is created.

@dataclass(frozen=True)
class ArgusBertConfig:
    """Current config defaults must match the verified Kaggle artifact."""

    # =========================================================================
    # Step 3: Fill in the token and sequence settings.
    # =========================================================================
    # Hint:
    # These values match the tokenized Phase 2 artifact.

    vocab_size: int = 1233
    max_seq_len: int = 16
    pad_token_id: int = 3
    mask_token_id: int = 2

    # =========================================================================
    # Step 4: Fill in the transformer size settings.
    # =========================================================================
    # Hint:
    # hidden_size must be divisible by num_attention_heads.

    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024

    # =========================================================================
    # Step 5: Fill in dropout settings.
    # =========================================================================
    # Hint:
    # Dropout helps reduce overfitting during training.

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # =========================================================================
    # Step 6: Validate the config before model construction.
    # =========================================================================
    # Hint:
    # Validation catches impossible model settings early.

    def validate(self) -> None:
        """Validate values before model construction."""

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
