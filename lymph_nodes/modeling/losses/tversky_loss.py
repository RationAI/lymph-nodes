import torch
from torch import nn


class TverskyLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        true_pos = (preds * targets).sum(dim=(1, 2, 3))
        false_neg = ((1 - preds) * targets).sum(dim=(1, 2, 3))
        false_pos = (preds * (1 - targets)).sum(dim=(1, 2, 3))

        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        return 1 - tversky_index.mean()
