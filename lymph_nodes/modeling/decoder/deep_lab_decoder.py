import torch
import torch.nn as nn
import torch.nn.functional as F

from lymph_nodes.modeling.blocks import BasicConvBlock


class DeepLabDecoder(nn.Module):
    def __init__(
        self, f1: int, f2: int, num_classes: int = 1, inner_f: int = 48
    ) -> None:
        super().__init__()

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(f1, inner_f, 1), nn.BatchNorm2d(inner_f), nn.ReLU()
        )

        self.decoder = BasicConvBlock(f2 + inner_f, 256)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def interpolate(self, x: torch.Tensor, size: list[int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x_aspp: torch.Tensor, x1: torch.tensor) -> torch.Tensor:
        x_aspp_up = self.interpolate(x_aspp, x1.shape[2:])
        x_low = self.low_level_proj(x1)

        return self.classifier(self.decoder(torch.cat([x_aspp_up, x_low], dim=1)))
