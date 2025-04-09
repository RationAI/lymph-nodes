import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXt(nn.Module):
    def __init__(
        self,
        weights: models.ConvNeXt_Base_Weights
        | None = models.ConvNeXt_Base_Weights.DEFAULT,
    ) -> None:
        super().__init__()

        # Load pre-trained ConvNeXt backbone
        self.backbone = models.convnext_base(weights=weights)

        # Extract encoder stages
        self.features = list(self.backbone.features.children())
        self.layers = nn.ModuleList(
            [
                nn.Sequential(*self.features[:2]),  # 128 H/4
                nn.Sequential(*self.features[2:4]),  # 256 H/8
                nn.Sequential(*self.features[4:6]),  # 512 H/16
                nn.Sequential(*self.features[6:]),  # 1024 H/32
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_ret = [x]

        for layer in self.layers:
            x = layer(x)
            x_ret.append(x)

        return x_ret
