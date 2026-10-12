from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn


if TYPE_CHECKING:
    from torch import Tensor


class FrozenFoundationModel(nn.Module):
    """Patch tokens of a frozen foundation model, as one feature map per tile.

    ``name`` is a display name of preprocessing.models.FOUNDATION_MODELS (GigaPath,
    UNI2, Virchow2: timm ViTs with 14 px patches). Takes uint8 RGB tiles (B, H, W, 3),
    normalizes them as the model expects, and returns the normalized patch tokens of
    its last ``n_layers`` blocks concatenated along the channels: (B, n_layers * D,
    H / 14, W / 14) — 16 x 16 at 224 px. Several blocks rather than only the last one:
    dense tasks gain from the intermediate layers (the DINOv2 "multi-scale" probe).

    Never trained: no gradients, always in eval mode, and left out of checkpoints
    (see SegmentationMetaArch) — the weights come from the hub every time.
    """

    mean: Tensor
    std: Tensor

    def __init__(self, name: str, n_layers: int = 4) -> None:
        super().__init__()
        from timm.data import resolve_model_data_config  # type: ignore[attr-defined]

        from preprocessing.models import FOUNDATION_MODELS

        self.model = FOUNDATION_MODELS[name]().eval().requires_grad_(False)
        self.n_layers = n_layers
        self.out_channels = n_layers * cast("int", self.model.embed_dim)

        config = resolve_model_data_config(self.model)
        self.register_buffer(
            "mean",
            torch.tensor(config["mean"]).view(1, 3, 1, 1) * 255,
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(config["std"]).view(1, 3, 1, 1) * 255, persistent=False
        )

    def train(self, mode: bool = True) -> FrozenFoundationModel:
        # Lightning puts the whole LightningModule into train mode for fitting; the
        # backbone stays in eval mode (no dropout / stochastic depth) regardless.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, tiles: Tensor) -> Tensor:
        images = (tiles.permute(0, 3, 1, 2).float() - self.mean) / self.std
        # forward_intermediates is timm ViT API, not declared on nn.Module.
        layers = self.model.forward_intermediates(  # type: ignore[operator]
            images,
            indices=self.n_layers,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return torch.cat(layers, dim=1)
