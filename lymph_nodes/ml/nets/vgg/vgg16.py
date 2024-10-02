# Copyright (c) The RationAI team
import mlflow
import torch
from torchvision.models import vgg16


"""
The following code modify vgg16 architecture.  
"""


class VGG16Features:
    """VGG16Features is a structure containing the feature extraction part of the model vgg16."""

    def __new__(cls, weights=None, model_uri=None) -> torch.nn.Module:
        if weights is not None and model_uri is not None:
            raise ValueError("Only one of weights or model_uri can be specified")
        if model_uri:
            donor = mlflow.pytorch.load_model(model_uri)
        else:
            donor = vgg16(weights=weights)
        return donor.features
