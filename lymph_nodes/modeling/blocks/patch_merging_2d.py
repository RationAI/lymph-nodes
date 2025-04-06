from collections.abc import Callable
from torch import nn
import torch


class PatchMerging2D(nn.Module):
    """Patch Merging Layer."""

    def __init__(
        self, dim: int, norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm
    ) -> None:
        """Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(
                f"Warning, x.shape {x.shape} is not match even ===========", flush=True
            )
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x1 = x1[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x2 = x2[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x3 = x3[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
