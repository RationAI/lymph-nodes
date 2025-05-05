import timm
import torch
import torch.nn as nn


class GigaPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        tile_encoder = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        blocks = tile_encoder.blocks

        self.embed = nn.Sequential(
            tile_encoder.patch_embed,
            tile_encoder.pos_drop,
            tile_encoder.patch_drop,
            tile_encoder.norm_pre,
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(*blocks[0:3]),  # 2
                nn.Sequential(*blocks[3:6]),  # 5
                nn.Sequential(*blocks[6:19]),  # 18
                nn.Sequential(*blocks[19:37]),  # 36
                nn.Sequential(*blocks[37:]),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.embed(x)

        x_ret = []

        for layer in self.layers:
            x = layer(x)
            x_ret.append(x)

        return x_ret
