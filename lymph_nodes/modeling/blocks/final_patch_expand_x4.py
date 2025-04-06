from collections.abc import Callable
from einops import rearrange
from torch import nn
import torch


class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(
        self,
        dim: int,
        dim_scale: int = 4,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // (self.dim_scale**2),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x
