from typing import Any
from torch import nn
import torch
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)


from lymph_nodes.modeling.decoder import UNetResDecoder
from lymph_nodes.modeling.encoder import VSSMEncoder
from lymph_nodes.modeling.utils import InitWeights_He


class SwinUMamba(nn.Module):
    def __init__(self, vss_args: dict[str, Any], decoder_args: dict[str, Any]) -> None:
        super().__init__()
        self.vssm_encoder = VSSMEncoder(**vss_args)
        self.decoder = UNetResDecoder(**decoder_args)

        self.apply(InitWeights_He(1e-2))
        self.apply(init_last_bn_before_add_to_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.vssm_encoder(x)
        out = self.decoder(skips)
        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True
