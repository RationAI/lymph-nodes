# Copyright (c) The RationAI team
import mlflow
import torch
from torch import nn
from torchvision.models import resnet50


class ResNet50Features:
    """ResNet50Features is a structure containing the feature extraction part of the model resnet50."""

    def __new__(cls, weights=None, model_uri=None) -> torch.nn.Module:
        if weights is not None and model_uri is not None:
            raise ValueError("Only one of weights or model_uri can be specified")
        if model_uri:
            donor = mlflow.pytorch.load_model(model_uri)
        else:
            donor = resnet50(weights=weights)
        return nn.Sequential(
            *list(donor.children())[:-2]
        )  # Exclude the last fully connected layer and global average pool


class ResnetBinaryClassifier(torch.nn.Sequential):
    """ResnetBinaryClassifier is a structure containing the binary classifier part of the resnet architecture."""

    def __init__(self, dropout_probability: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_module("dropout", torch.nn.Dropout(p=dropout_probability))
        self.add_module("dense", torch.nn.Linear(2048, 1))

        torch.nn.init.xavier_uniform_(self.dense.weight)
        torch.nn.init.zeros_(self.dense.bias)
