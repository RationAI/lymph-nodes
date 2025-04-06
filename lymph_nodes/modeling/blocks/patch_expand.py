from collections.abc import Callable
from torch import nn
from einops import rearrange
import torch


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(
        self,
        dim: int,
        dim_scale: int = 2,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        return x
