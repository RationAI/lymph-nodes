import torch
import torch.nn as nn

from lymph_nodes.modeling.blocks import BinaryClassifier


class ClassifacationNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        bottleneck: nn.Module,
        head: nn.Module = BinaryClassifier(),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.bottleneck(self.backbone(x)))
