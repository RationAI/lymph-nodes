# Copyright (c) The RationAI team
import mlflow
import torch
from torchvision.models import alexnet


class AlexNetFeatures:
    """AlexNetFeatures is a structure containing the feature extraction part of the model alexnet."""

    def __new__(cls, weights=None, model_uri=None) -> torch.nn.Module:
        if weights is not None and model_uri is not None:
            raise ValueError("Only one of weights or model_uri can be specified")
        if model_uri:
            donor = mlflow.pytorch.load_model(model_uri)
        else:
            donor = alexnet(weights=weights)
        return donor.features


class AlexBinaryClassifier(torch.nn.Sequential):
    """AlexBinaryClassifier is a structure containing the binary classifier part of the alexnet architecture."""

    def __init__(self, dropout_probability: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_module("dropout", torch.nn.Dropout(p=dropout_probability))
        # AlexNet's conv5 output is 256 feature maps of size 6x6, which are then pooled to 1x1
        self.add_module(
            "dense", torch.nn.Linear(256, 1)
        )  # Adjusted the input features to match AlexNet

        torch.nn.init.xavier_uniform_(self.dense.weight)
        torch.nn.init.zeros_(self.dense.bias)
