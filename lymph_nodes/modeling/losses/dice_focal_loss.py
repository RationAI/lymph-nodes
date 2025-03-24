from torch import Tensor, nn

from lymph_nodes.modeling.losses.dice_loss import DiceLoss
from lymph_nodes.modeling.losses.focal_loss import FocalLoss
from lymph_nodes.modeling.losses.tversky_loss import TverskyLoss


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha, gamma)
        self.tversky = TverskyLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return (
            self.dice(preds, targets)
            + self.focal(preds, targets)
            + self.tversky(preds, targets)
        )
