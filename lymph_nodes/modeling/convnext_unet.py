import torch
import torch.nn as nn
import torchvision.models as models

from lymph_nodes.modeling.blocks import BasicConvBlock, BasicUpBlock
from lymph_nodes.modeling.encoder import ConvNeXt
from lymph_nodes.modeling.unet import UNet
from lymph_nodes.typing import Outputs


class ConvNeXtUNet(UNet):
    def __init__(
        self,
        decoder: nn.Module,
        weights: models.ConvNeXt_Base_Weights
        | None = models.ConvNeXt_Base_Weights.DEFAULT,
        cls_head: nn.Module | None = None,
    ) -> None:
        super().__init__(ConvNeXt(weights=weights), decoder, cls_head)

        # It has  to be named bacbone because of FineTuner
        self.pre_convolution = BasicConvBlock(
            in_channels=3,
            out_channels=64,
        )

        self.last_upsampling = BasicUpBlock(
            in_channels=128,
            out_channels=64,
            layers=BasicConvBlock(64 * 2, 64),
            kernel_size=4,
            stride=4,
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Outputs:
        x_1 = self.pre_convolution(x)
        skips = self.backbone(x)

        return Outputs(
            labels=self.cls_head(skips[-1]) if self.cls_head else None,
            masks=self.final_conv(
                self.last_upsampling(self.decoder([*skips[::-1]]), x_1)
            )
            .sigmoid()
            .squeeze(1),
        )
