import torch
import torch.nn as nn

from lymph_nodes.modeling.blocks import BasicConvBlock, BasicUpBlock


class SimpleDecoder(nn.Module):
    def __init__(self, nun_features: list[int], num_classes: int = 1) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                BasicUpBlock(
                    nun_features[i],
                    nun_features[i + 1],
                    BasicConvBlock(nun_features[i + 1] * 2, nun_features[i + 1]),
                )
                for i in range(len(nun_features) - 1)
            ]
        )

        self.final_conv = nn.Conv2d(nun_features[-1], num_classes, kernel_size=1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x_k = x[0]
        for i, layer in enumerate(self.layers):
            x_k = layer(x_k, x[i + 1])

        return self.final_conv(x_k)
