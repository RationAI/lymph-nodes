from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        features: list[int] = [4096, 4096],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        features = [7 * 7 * 512, *features]

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=(1, 1)),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=-3, end_dim=-1),  # (B, C)
            *[
                nn.Sequential(
                    nn.Linear(features[i], features[i + 1]),
                    nn.ReLU(True),
                    nn.Dropout(p=dropout),
                )
                for i in range(len(features) - 1)
            ],
            nn.Linear(features[-1], num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x.sigmoid()
