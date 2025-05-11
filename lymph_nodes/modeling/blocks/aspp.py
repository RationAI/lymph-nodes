import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.atrous_block1 = nn.Conv2d(
            in_channels, out_channels, 1, padding=0, dilation=1
        )
        self.atrous_block6 = nn.Conv2d(
            in_channels, out_channels, 3, padding=6, dilation=6
        )
        self.atrous_block12 = nn.Conv2d(
            in_channels, out_channels, 3, padding=12, dilation=12
        )
        self.atrous_block18 = nn.Conv2d(
            in_channels, out_channels, 3, padding=18, dilation=18
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1), nn.ReLU()
        )

        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        out1 = self.atrous_block1(x)
        out2 = self.atrous_block6(x)
        out3 = self.atrous_block12(x)
        out4 = self.atrous_block18(x)

        out5 = F.interpolate(
            self.global_avg_pool(x), size=size, mode="bilinear", align_corners=False
        )

        return self.merge(torch.cat([out1, out2, out3, out4, out5], dim=1))
