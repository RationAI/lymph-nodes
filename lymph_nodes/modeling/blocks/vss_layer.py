from collections.abc import Callable
import math
from torch import nn

import torch
import torch.utils.checkpoint as checkpoint


from lymph_nodes.modeling.blocks import VSSBlock


class VSSLayer(nn.Module):
    """A basic layer for one stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        d_state: int = 16,
    ) -> None:
        """Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        """
        super().__init__()

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

        if True:  # is this really applied? Yes, but been overriden later in VSSM!

            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
