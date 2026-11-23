from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch import Tensor


class LoRALinear(nn.Module):
    """A frozen ``base`` linear layer plus a trainable low-rank update (LoRA).

    ``base(x) + alpha / rank * x A^T B^T``, with B zero at the start, so training starts
    from the frozen model's outputs. Only ``lora_a`` and ``lora_b`` are trained.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        update = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return self.base(x) + self.scaling * update


class FrozenFoundationModel(nn.Module):
    """Patch tokens of a frozen foundation model, as one feature map per tile.

    ``name`` is a display name of preprocessing.models.FOUNDATION_MODELS (timm ViTs:
    UNI2 and Virchow2 with 14 px patches, GigaPath with 16 px). Takes uint8 RGB tiles
    (B, H, W, 3), normalizes them as the model expects, and returns the normalized
    patch tokens of its last ``n_layers`` blocks concatenated along the channels:
    (B, n_layers * D, H / patch, W / patch) — 16 x 16 at 224 px for 14 px patches,
    14 x 14 for 16 px. Several blocks rather than only the last one: dense tasks gain
    from the intermediate layers (the DINOv2 "multi-scale" probe). ``layers``, if given,
    names the blocks instead (0-based, e.g. [7, 15, 23, 31] for Virchow2's 32): blocks
    spread through the network keep more of what the last ones abstract away — the
    stain intensity, which pretraining with colour augmentation teaches them to ignore.

    Never trained: no gradients, always in eval mode, and left out of checkpoints
    (see SegmentationMetaArch) — the weights come from the hub every time. Except, for
    the idea research-notes/ideas/lymph-nodes-foundation-magnification-lora.md
    (2026-11-23), two options, both off by default:

    - ``input_scale``: the tiles resized by this factor (bicubic) before the model, its
      position embeddings resampled to match (timm ``set_input_size``; ``tile_size``
      px tiles), so a tumour cell spans more patch tokens: 2 gives 32 x 32 tokens at
      224 px tiles — the target grid of the upsampled tilings itself, so the head's
      own upsampling goes (UpsamplingHead ``scale`` 1).
    - ``lora_rank`` > 0: LoRA (LoRALinear) on the attention's qkv and output
      projections of the last ``lora_blocks`` blocks (0: all). Only those weights train
      (``trainable_parameters``) and are checkpointed; the rest stays frozen.
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        name: str,
        n_layers: int = 4,
        layers: Sequence[int] | None = None,
        input_scale: float = 1.0,
        tile_size: int = 224,
        lora_rank: int = 0,
        lora_blocks: int = 0,
        lora_alpha: float | None = None,
    ) -> None:
        super().__init__()
        from timm.data import resolve_model_data_config  # type: ignore[attr-defined]

        from preprocessing.models import FOUNDATION_MODELS

        self.model = FOUNDATION_MODELS[name]().eval().requires_grad_(False)
        self.input_scale = float(input_scale)
        if self.input_scale != 1.0:
            size = round(tile_size * self.input_scale)
            self.model.set_input_size(img_size=(size, size))  # type: ignore[operator]
        self.trainable = lora_rank > 0
        if self.trainable:
            blocks = cast("nn.Sequential", self.model.blocks)
            for block in list(blocks)[-lora_blocks:] if lora_blocks else blocks:
                attention = cast("nn.Module", block.attn)
                for part in ("qkv", "proj"):
                    setattr(
                        attention,
                        part,
                        LoRALinear(
                            getattr(attention, part),
                            lora_rank,
                            float(lora_alpha or lora_rank),
                        ),
                    )
        self.layers: int | list[int] = (
            n_layers if layers is None else [int(i) for i in layers]
        )
        count = n_layers if layers is None else len(self.layers)  # type: ignore[arg-type]
        self.out_channels = count * cast("int", self.model.embed_dim)

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

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """The LoRA weights (none without ``lora_rank``): what the optimizer steps."""
        return (p for n, p in self.named_parameters() if ".lora_" in f".{n}")

    def forward(self, tiles: Tensor) -> Tensor:
        # Gradients only with LoRA, and only where they are on anyway (training).
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            images = (tiles.permute(0, 3, 1, 2).float() - self.mean) / self.std
            if self.input_scale != 1.0:
                images = F.interpolate(
                    images,
                    scale_factor=self.input_scale,
                    mode="bicubic",
                    align_corners=False,
                )
            # forward_intermediates is timm ViT API, not declared on nn.Module.
            layers = self.model.forward_intermediates(  # type: ignore[operator]
                images,
                indices=self.layers,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        return torch.cat(layers, dim=1)
