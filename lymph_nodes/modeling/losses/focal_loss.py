# import torch
from torch import Tensor, nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.bce = nn.BCELoss(reduction="none")

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        bce_loss = self.bce(preds, targets)

        # Compute focal weight
        p_t = preds * targets + (1 - preds) * (1 - targets)  # p_t = p if y=1, else 1-p
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        return (focal_weight * bce_loss).mean()
