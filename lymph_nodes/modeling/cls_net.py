import torch
import torch.nn as nn


class ClsNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        bottleneck: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        if isinstance(x, list):
            x = x[-1]

        print(x.shape, flush=True)

        if self.bottleneck is not None:
            x = self.bottleneck(x)

        print(x.shape, flush=True)

        return self.head(x)
