from collections.abc import Callable
from functools import partial

import torch
from timm.models.layers import DropPath
from torch import nn


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from lymph_nodes.modeling.blocks import SS2D


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.drop_path(self.self_attention(self.ln_1(input)))
