import timm
import torch
import torch.nn as nn


class GigaPath(nn.Module):
    def __init__(self, cls_head: nn.Module) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        tile_encoder = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        self.cls_head = cls_head

        self.backbone = nn.Sequential(
            tile_encoder.patch_embed,
            tile_encoder.pos_drop,
            tile_encoder.patch_drop,
            tile_encoder.norm_pre,
            tile_encoder.blocks,
            tile_encoder.norm,
            tile_encoder.fc_norm,
            tile_encoder.head_drop,
            tile_encoder.head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(x).permute(0, 2, 1).reshape(-1, 1536, 14, 14)

        return self.cls_head(latent)
