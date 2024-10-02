# Copyright (c) The RationAI team
import torch


class VGGBinaryClassifier(torch.nn.Sequential):
    """VGGBinaryClassifier is a structure containing the binary classifier part of the vgg architecture."""

    def __init__(self, dropout_probability: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_module("dropout", torch.nn.Dropout(p=dropout_probability))
        self.add_module("dense", torch.nn.Linear(512, 1))

        torch.nn.init.xavier_uniform_(self.dense.weight)
        torch.nn.init.zeros_(self.dense.bias)
