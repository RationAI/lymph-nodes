import torch
import torch.nn as nn

from lymph_nodes.modeling.decoder import DeepLabDecoder
from lymph_nodes.modeling.deep_lab import DeepLab
from lymph_nodes.modeling.encoder import GigaPath
from lymph_nodes.typing import Outputs


class GigaPathDeepLab(DeepLab):
    def __init__(self, cls_head: nn.Module | None = None) -> None:
        super().__init__(
            GigaPath(), DeepLabDecoder(f1=128, f2=512), features=512, cls_head=cls_head
        )

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1536, 2 ** (i + 5), kernel_size=(1, 1)),
                    nn.ConvTranspose2d(
                        2 ** (i + 5),
                        2 ** (i + 5),
                        kernel_size=(224 // 14 // (2**i), 224 // 14 // (2**i)),
                        stride=(224 // 14 // (2**i), 224 // 14 // (2**i)),
                    ),
                )
                for i in [2, 4]
            ]
        )

    def forward(self, x: torch.Tensor) -> Outputs:
        embeds = self.backbone(x)
        embeds = [embeds[2], embeds[4]]

        x1, x2 = [
            conv_block(skip.permute(0, 2, 1).reshape(skip.shape[0], 1536, 14, 14))
            for conv_block, skip in zip(self.conv_blocks, embeds, strict=True)
        ]

        x_aspp = self.aspp(x2)
        mask_pred = self.decoder.interpolate(self.decoder(x_aspp, x1), x.shape[2:])

        return Outputs(
            labels=self.cls_head(x2) if self.cls_head else None,
            masks=mask_pred.sigmoid().squeeze(1),
        )
