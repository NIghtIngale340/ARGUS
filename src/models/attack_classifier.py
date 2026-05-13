"""ARGUS attack classifier with a classification head on pre-trained BERT."""

from typing import Optional

import torch
from torch import nn, Tensor
from transformers import BertModel, BertConfig

from src.models.config import ArgusBertConfig


class ARGUSClassifier(nn.Module):
    """Pre-trained BERT + classification head on [CLS] token.

    Architecture:
        session tokens -> BERT encoder (bottom layers frozen)
        -> [CLS] hidden state (256-dim)
        -> Dropout -> Linear(256, num_classes)
        -> logits
    """

    def __init__(
        self,
        config: ArgusBertConfig | None = None,
        num_classes: int = 2,
        freeze_layers: int = 4,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = config or ArgusBertConfig()
        self.config.validate()
        self.num_classes = num_classes

        hf_config = BertConfig(
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

        self.bert = BertModel(hf_config)
        self._freeze_layers(freeze_layers)

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(
            self.config.hidden_size,
            self.num_classes,
        )

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze embedding + bottom N encoder layers.

        Keeps upper layers trainable so they can specialize for classification
        while preserving the general "log grammar" learned during pre-training.
        """
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        num_to_freeze = min(num_layers, self.config.num_hidden_layers)
        for layer_idx in range(num_to_freeze):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass: tokens to class logits.

        Args:
            input_ids: (B, T) token ID tensor.
            attention_mask: (B, T) mask tensor (1 = real token, 0 = padding).

        Returns:
            logits: (B, num_classes) classification logits.
        """
        input_ids = input_ids.long()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def load_pretrained_bert(self, checkpoint_path: str) -> None:
        """Load pre-trained BERT weights from an MLM checkpoint.

        The MLM checkpoint saves ``ArgusBertForMaskedLM.state_dict()`` under
        the ``"model"`` key. We strip the leading ``model.bert.`` prefix and
        ignore any ``model.cls.`` entries (MLM head).

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))

        TARGET_PREFIX = "model.bert."
        bert_state = {}
        skipped_mlm_keys = 0

        for key, value in state_dict.items():
            if key.startswith(TARGET_PREFIX):
                new_key = key[len(TARGET_PREFIX):]
                bert_state[new_key] = value
            elif key.startswith("model.cls."):
                skipped_mlm_keys += 1
            elif not key.startswith("model."):
                bert_state[key] = value

        if not bert_state:
            raise RuntimeError(
                f"Could not extract BERT weights from checkpoint. "
                f"Keys sample: {list(state_dict.keys())[:5]}"
            )

        missing, unexpected = self.bert.load_state_dict(bert_state, strict=False)
        loaded = len(bert_state) - len(unexpected)
        print(
            f"Loaded {loaded} BERT weight tensors from pre-trained checkpoint "
            f"(skipped {skipped_mlm_keys} MLM-head keys)."
        )
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}")
        if missing:
            print(f"  Missing keys (random init): {missing[:5]}")

    def count_trainable_params(self) -> tuple[int, int]:
        """Return (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
