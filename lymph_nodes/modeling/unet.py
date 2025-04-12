import torch
import torch.nn as nn

from lymph_nodes.typing import Outputs


class UNet(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, cls_head: nn.Module | None = None
    ) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = encoder
        self.decoder = decoder
        self.cls_head = cls_head

    def forward(self, x: torch.Tensor) -> Outputs:
        skips = self.backbone(x)

        return Outputs(
            labels=self.cls_head(skips[-1]) if self.cls_head else None,
            masks=self.decoder(skips[::-1]).sigmoid().squeeze(1),
        )
