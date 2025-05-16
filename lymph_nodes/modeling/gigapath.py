import timm
import torch
import torch.nn as nn


class GigaPath(nn.Module):
    def __init__(self, cls_head: nn.Module) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(x).permute(0, 2, 1).reshape(-1, 1536, 14, 14)

        return self.cls_head(latent)
