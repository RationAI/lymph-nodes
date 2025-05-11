import torch
import torch.nn as nn
import torchvision.models as models

from lymph_nodes.modeling.decoder import DeepLabDecoder
from lymph_nodes.modeling.deep_lab import DeepLab
from lymph_nodes.modeling.encoder import ConvNeXt
from lymph_nodes.typing import Outputs


class ConvNeXtDeepLab(DeepLab):
    def __init__(
        self,
        weights: models.ConvNeXt_Base_Weights
        | None = models.ConvNeXt_Base_Weights.DEFAULT,
        cls_head: nn.Module | None = None,
    ) -> None:
        super().__init__(
            ConvNeXt(weights=weights),
            DeepLabDecoder(f1=128, f2=512),
            features=1024,
            cls_head=cls_head,
        )

    def forward(self, x: torch.Tensor) -> Outputs:
        skips = self.backbone(x)

        x1, x2 = skips[0], skips[-1]
        x_aspp = self.aspp(x2)

        mask_pred = self.decoder.interpolate(self.decoder(x_aspp, x1), x.shape[2:])

        return Outputs(
            labels=self.cls_head(x_aspp) if self.cls_head else None,
            masks=mask_pred.sigmoid().squeeze(1),
        )
