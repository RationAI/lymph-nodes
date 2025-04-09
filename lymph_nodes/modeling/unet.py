import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, encoder: nn.Module, decored: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decored

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x.sigmoid()
