import torchvision
from torch import nn


def resnet50(weights: str | None = None) -> nn.Module:
    resnet = torchvision.models.resnet50(weights=weights)
    return nn.Sequential(
        *(list(resnet.children())[:-2]),
        nn.Conv2d(2048, 512, kernel_size=1, stride=1),
    )
