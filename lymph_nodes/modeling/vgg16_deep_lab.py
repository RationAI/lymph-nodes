import torch
import torch.nn as nn
import torchvision.models as models

from lymph_nodes.modeling.deep_lab import DeepLab
from lymph_nodes.modeling.encoder import VGG16


class VGG16DeepLab(DeepLab):
    def __init__(
        self,
        weights: models.VGG16_BN_Weights | None = models.VGG16_BN_Weights.IMAGENET1K_V1,
        cls_head: nn.Module | None = None,
    ) -> None:
        super().__init__(
            VGG16(weights=weights),
            features=(256, 512, 512),
            cls_head=cls_head,
        )

    def _get_skips(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skips = self.backbone(x)

        return skips[2], skips[4], skips[-1]
