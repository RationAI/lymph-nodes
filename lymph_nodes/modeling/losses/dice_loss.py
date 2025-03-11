import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice_score  # Dice Loss (1 - Dice Coefficient)
