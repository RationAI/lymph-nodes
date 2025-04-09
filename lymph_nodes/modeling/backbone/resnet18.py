import torchvision.models as model
from torch import nn


def resnet18(
    weights: model.ResNet18_Weights | None = model.ResNet18_Weights.IMAGENET1K_V1,
) -> nn.Module:
    resnet = model.resnet18(weights=weights)
    return nn.Sequential(
        *(list(resnet.children())[:-2]),
    )
