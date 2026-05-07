"""MLM-loss anomaly scorer for session anomaly detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from src.models.argus_bert import ArgusBertForMaskedLM


@dataclass(frozen=True)
class AnomalyScoringConfig:
    """Settings for deterministic MLM-loss scoring."""

    cls_token_id: int = 0
    sep_token_id: int = 1
    mask_token_id: int = 2
    pad_token_id: int = 3
    ignore_index: int = -100


class MLMAnomalyScorer:
    """Score sessions using deterministic masked-token reconstruction loss."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        config: AnomalyScoringConfig | None = None,
    ) -> None:
        self.config = config or AnomalyScoringConfig()
        self.device = self._select_device(device)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _select_device(self, requested_device: str) -> torch.device:
        """Select CPU or CUDA based on availability."""
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def _load_model(self, checkpoint_path: str | Path) -> ArgusBertForMaskedLM:
        """Load pretrained checkpoint into model."""
        model = ArgusBertForMaskedLM()
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        return model

    def prepare_scoring_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device).bool()

        labels = input_ids.clone()
        maskable = (
            attention_mask
            & (input_ids != self.config.cls_token_id)
            & (input_ids != self.config.sep_token_id)
            & (input_ids != self.config.pad_token_id)
        )

        labels[~maskable] = self.config.ignore_index
        masked_input_ids = input_ids.clone()
        masked_input_ids[maskable] = self.config.mask_token_id

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_mask": maskable,
        }

    @torch.inference_mode()
    def score_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute anomaly score per session using masked-token prediction loss."""
        scoring_batch = self.prepare_scoring_batch(batch)
        output = self.model(
            input_ids=scoring_batch["input_ids"],
            attention_mask=scoring_batch["attention_mask"],
        )

        logits = output.logits
        labels = scoring_batch["labels"].long()
        target_mask = scoring_batch["target_mask"]

        token_losses = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction="none",
            ignore_index=self.config.ignore_index,
        ).view(labels.shape)

        loss_sum = (token_losses * target_mask.float()).sum(dim=1)
        token_count = target_mask.sum(dim=1).clamp_min(1)
        return loss_sum / token_count
