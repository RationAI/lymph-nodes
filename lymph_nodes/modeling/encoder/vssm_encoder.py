from collections.abc import Callable
import math
from torch import nn
import torch
from typing import Any
from timm.models.layers import trunc_normal_

from lymph_nodes.modeling.blocks import PatchEmbed2D, VSSLayer, PatchMerging2D


class VSSMEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_channels: int = 3,
        depths: list[int] = [2, 2, 9, 2],
        dims: list[int] = [96, 192, 384, 768],
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(
                    PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer)
                )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x):
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret
