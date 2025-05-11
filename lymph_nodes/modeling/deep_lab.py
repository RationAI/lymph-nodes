import torch
import torch.nn as nn

from lymph_nodes.modeling.blocks import ASPP
from lymph_nodes.modeling.decoder import DeepLabDecoder
from lymph_nodes.typing import Outputs


class DeepLab(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: DeepLabDecoder,
        features: int,
        cls_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = encoder
        self.decoder = decoder
        self.aspp = ASPP(features, 512)
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> Outputs:
        skips = self.backbone(x)

        x1, x2 = skips[2], skips[-1]
        x_aspp = self.aspp(x2)

        mask_pred = self.decoder.interpolate(self.decoder(x_aspp, x1), x.shape[2:])

        return Outputs(
            labels=self.cls_head(x_aspp) if self.cls_head else None,
            masks=mask_pred.sigmoid().squeeze(1),
        )
