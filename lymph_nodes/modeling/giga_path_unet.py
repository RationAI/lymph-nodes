import torch
import torch.nn as nn

from lymph_nodes.modeling.decoder import SimpleDecoder
from lymph_nodes.modeling.encoder import GigaPath
from lymph_nodes.typing import Outputs


class GigaPathUnet(nn.Module):
    def __init__(self, cls_head: nn.Module | None = None) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = GigaPath()
        self.decoder = SimpleDecoder(nun_features=[512, 256, 128, 64, 32])
        self.cls_head = cls_head

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
                for i in range(5)
            ]
        )

    def forward(self, x: torch.Tensor) -> Outputs:
        skips = self.backbone(x)

        skips = [
            conv_block(skip.permute(0, 2, 1).reshape(skip.shape[0], 1536, 14, 14))
            for conv_block, skip in zip(self.conv_blocks, skips, strict=True)
        ]

        return Outputs(
            labels=self.cls_head(skips[-1]) if self.cls_head else None,
            masks=self.decoder(skips[::-1]).sigmoid().squeeze(1),
        )
