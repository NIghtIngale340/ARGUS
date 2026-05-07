"""ARGUS-BERT configuration defaults."""

from dataclasses import dataclass

@dataclass(frozen=True)
class ArgusBertConfig:
    """Defaults for ARGUS-BERT model shape."""
    vocab_size: int = 1233
    max_seq_len: int = 16
    pad_token_id: int = 3
    mask_token_id: int = 2

    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
