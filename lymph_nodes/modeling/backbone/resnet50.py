import torchvision.models as model
from torch import nn


def resnet50(
    weights: model.ResNet50_Weights | None = model.ResNet50_Weights.IMAGENET1K_V1,
) -> nn.Module:
    resnet = model.resnet50(weights=weights)
    return nn.Sequential(*(list(resnet.children())[:-2]))
