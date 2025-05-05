import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(
        self,
        weights: models.VGG16_BN_Weights | None = models.VGG16_BN_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        vgg16 = models.vgg16_bn(weights=weights)
        features = list(vgg16.features.children())

        # VGG16 with batch norm has these blocks:
        self.layers = nn.ModuleList(
            [
                nn.Sequential(*features[0:6]),  # 64 H
                nn.Sequential(*features[6:13]),  # 128 H/2
                nn.Sequential(*features[13:23]),  # 256 H/4
                nn.Sequential(*features[23:33]),  # 512 H/8
                nn.Sequential(*features[33:43]),  # 512 H/16
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_ret = []

        for layer in self.layers:
            x = layer(x)
            x_ret.append(x)

        return x_ret
