from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn


if TYPE_CHECKING:
    from torch import Tensor


class ContextHead(nn.Module):
    """Per-patch logits from each patch token and its neighbours': a small convolutional head.

    The linear probe (a 1x1 convolution, configs/model/foundation_segmentation.yaml)
    classifies every 14 px patch (~7 um at 0.5 um/px, less than a cell) from its own
    token alone. Here a 1x1 convolution reduces the tokens to ``hidden`` channels,
    ``layers`` convolutions of ``kernel_size`` mix each patch with its neighbours (two
    3x3 layers see 5 x 5 patches, ~35 um: a cell and what surrounds it) and a 1x1
    convolution gives ``out_channels`` logits. The tile's edges are zero-padded.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        kernel_size: int = 3,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        ]
        for _ in range(layers):
            blocks += [
                nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
            ]
        blocks.append(nn.Conv2d(hidden, out_channels, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)
