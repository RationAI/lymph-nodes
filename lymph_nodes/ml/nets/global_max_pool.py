# Copyright (c) The RationAI team
import torch.nn.functional as F
from torch import nn


class GMaxPool(nn.Module):
    """GMaxPool.

    GMaxPool applies global max pooling fucntion to result
    feature map from the feature extraction part of the CNN model.
    """

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=1)
        return x.flatten(start_dim=-3, end_dim=-1)
