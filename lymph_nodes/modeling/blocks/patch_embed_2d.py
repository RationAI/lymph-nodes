from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        patch_size: int | tuple[int, int] = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Callable[..., torch.nn.Module] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Args:
        patch_size: Patch token size. Default: 4.
        in_channels: Number of input image channels. Default: 3.
        embed_dim: Number of linear projection output channels. Default: 96.
        norm_layer: Normalization layer. Default: None.
        """  # noqa: D205
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
