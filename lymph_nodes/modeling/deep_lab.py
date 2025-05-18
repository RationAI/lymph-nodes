from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from lymph_nodes.modeling.blocks import ASPP
from lymph_nodes.modeling.decoder import DeepLabDecoder
from lymph_nodes.typing import Outputs


class DeepLab(nn.Module, ABC):
    def __init__(
        self,
        encoder: nn.Module,
        features: tuple[int, int, int],
        cls_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = encoder
        self.decoder = DeepLabDecoder(features[0], 512)
        self.aspp = ASPP(features[1], features[-1], 512)
        self.cls_head = cls_head

    @abstractmethod
    def _get_skips(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def forward(self, x: torch.Tensor) -> Outputs:
        h, m, l = self._get_skips(x)

        x_aspp = self.aspp(m, l)

        mask_pred = self.decoder.interpolate(self.decoder(x_aspp, h), x.shape[2:])

        return Outputs(
            labels=self.cls_head(x_aspp) if self.cls_head else None,
            masks=mask_pred.sigmoid().squeeze(1),
        )
