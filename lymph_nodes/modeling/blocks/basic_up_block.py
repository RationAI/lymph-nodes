import torch
import torch.nn as nn


class BasicUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: nn.Module,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.layers = layers

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([self.up(x_1), x_2], dim=1))
