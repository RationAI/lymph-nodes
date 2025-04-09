import torch
import torch.nn as nn
import torchvision.models as models

from lymph_nodes.modeling.blocks.basic_conv_block import BasicConvBlock
from lymph_nodes.modeling.encoder import ConvNeXt
from lymph_nodes.modeling.unet import UNet


class ConvNeXtUNet(UNet):
    def __init__(
        self,
        decored: nn.Module,
        weights: models.ConvNeXt_Base_Weights
        | None = models.ConvNeXt_Base_Weights.DEFAULT,
    ) -> None:
        super().__init__(ConvNeXt(weights=weights), decored)
        # It has  to be named bacbone because of FineTuner
        self.pre_convolution = BasicConvBlock(
            in_channels=3,
            out_channels=128,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.pre_convolution(x)
        x = self.backbone(x)
        x = self.decoder([*x[::-1], x_1])
        return x.sigmoid()
