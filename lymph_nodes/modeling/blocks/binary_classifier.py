from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    def __init__(
        self, features: list[int] = [512, 512, 1], dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(features[i], features[i + 1]),
                    nn.ReLU(True),
                    nn.Dropout(p=dropout),
                )
                for i in range(len(features) - 2)
            ],
            nn.Linear(features[-2], features[-1]),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=-3, end_dim=-1)  # (B, C)
        x = self.head(x)
        return x.sigmoid()
