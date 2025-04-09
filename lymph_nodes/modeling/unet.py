import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, encoder: nn.Module, decored: nn.Module) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = encoder
        self.decoder = decored

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.decoder(x[::-1])
        return x.sigmoid()
