import torch
import torch.nn as nn

from lymph_nodes.modeling.deep_lab import DeepLab
from lymph_nodes.modeling.encoder import GigaPath


class GigaPathDeepLab(DeepLab):
    def __init__(self, cls_head: nn.Module | None = None) -> None:
        super().__init__(GigaPath(), features=(128, 512, 1536), cls_head=cls_head)

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

    def _get_skips(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeds = self.backbone(x)[::2]

        h, m = [
            conv_block(skip.permute(0, 2, 1).reshape(skip.shape[0], 1536, 14, 14))
            for conv_block, skip in zip(self.conv_blocks, embeds[:-1], strict=True)
        ]

        l = embeds[-1].permute(0, 2, 1).reshape(embeds[-1].shape[0], 1536, 14, 14)

        return h, m, l
