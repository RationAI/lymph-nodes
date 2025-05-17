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
        embed = self.backbone(x)

        return self.cls_head(embed)
