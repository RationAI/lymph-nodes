import torch
import torch.nn as nn
import torchvision.models as models

from lymph_nodes.modeling.blocks import BasicConvBlock, BasicUpBlock
from lymph_nodes.modeling.encoder import ConvNeXt
from lymph_nodes.modeling.unet import UNet


class ConvNeXtUNet(UNet):
    def __init__(
        self,
        decoder: nn.Module,
        weights: models.ConvNeXt_Base_Weights
        | None = models.ConvNeXt_Base_Weights.DEFAULT,
    ) -> None:
        super().__init__(ConvNeXt(weights=weights), decoder)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.pre_convolution(x)
        x = self.backbone(x)
        x = self.decoder([*x[::-1]])
        x = self.last_upsampling(x, x_1)
        return self.final_conv(x).sigmoid()
