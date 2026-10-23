from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn


if TYPE_CHECKING:
    from torch import Tensor


def _context_blocks(
    in_channels: int, hidden: int, kernel_size: int, layers: int, dropout: float
) -> list[nn.Module]:
    """ContextHead's layers up to its logits: tokens reduced, then mixed with neighbours'."""
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
    return blocks


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
        blocks = _context_blocks(in_channels, hidden, kernel_size, layers, dropout)
        blocks.append(nn.Conv2d(hidden, out_channels, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class UpsamplingHead(nn.Module):
    """ContextHead with a learned ``scale`` x upsampling: logits on a finer grid than the patches.

    The patch tokens (14 px, ~7 um at 0.5 um/px) go through ContextHead's layers; a
    transposed convolution then splits every patch into ``scale`` x ``scale`` cells (7 px,
    ~3.4 um at 2x) and a convolution refines them into ``out_channels`` logits:
    (B, out, g * scale, g * scale). A tumour cell of ~10 um straddles several patches,
    each seeing part of it; the finer cells let one be centred on it. Trained against
    target grids of that size (tilings with ``grid`` = 32 for 224 px tiles).

    With ``image_channels`` > 0 the tile's own pixels join at the fine grid: their optical
    densities through two 3x3 convolutions, averaged over each cell, concatenated before
    the refinement — the detail of a stained cell's outline that the tokens blur. The
    model then calls it with the tiles too (``uses_image``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        kernel_size: int = 3,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        scale: int = 2,
        image_channels: int = 0,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            *_context_blocks(in_channels, hidden, kernel_size, layers, dropout)
        )
        fine = hidden // 2
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden, fine, scale, stride=scale), nn.GELU()
        )
        self.uses_image = image_channels > 0
        self.stem = (
            nn.Sequential(
                nn.Conv2d(3, image_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(image_channels, image_channels, 3, padding=1),
                nn.GELU(),
            )
            if self.uses_image
            else None
        )
        self.refine = nn.Sequential(
            nn.Conv2d(fine + image_channels, fine, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(fine, out_channels, 1),
        )

    def forward(self, features: Tensor, tiles: Tensor | None = None) -> Tensor:
        x = self.up(self.trunk(features))
        if self.stem is not None:
            if tiles is None:
                raise ValueError("UpsamplingHead with image_channels needs the tiles")
            # uint8 (B, H, W, 3) -> optical density (B, 3, H, W): stain amounts, not light.
            density = -torch.log((tiles.permute(0, 3, 1, 2).float() + 1) / 256)
            pixels = F.adaptive_avg_pool2d(self.stem(density), x.shape[-2:])
            x = torch.cat([x, pixels], dim=1)
        return self.refine(x)
