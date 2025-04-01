from torch import Tensor, nn

from lymph_nodes.modeling.losses.focal_loss import FocalLoss
from lymph_nodes.modeling.losses.tversky_loss import TverskyLoss


class TverskyFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        tversky_weight: float = 0.5,
        focal_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss()
        self.focal = FocalLoss(alpha, gamma)
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.tversky_weight * self.tversky(
            preds, targets
        ) + self.focal_weight * self.focal(preds, targets)
