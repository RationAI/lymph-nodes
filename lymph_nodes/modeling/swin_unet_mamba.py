from typing import Any

import numpy as np
import torch
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)
from torch import nn

from lymph_nodes.modeling.decoder import UNetResDecoder
from lymph_nodes.modeling.encoder import VSSMEncoder
from lymph_nodes.modeling.utils import InitWeights_He
from lymph_nodes.typing import Outputs


torch.serialization.add_safe_globals(
    [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]
)


class SwinUNetMamba(nn.Module):
    def __init__(
        self,
        vss_args: dict[str, Any],
        decoder_args: dict[str, Any],
        pretrained: str | None = None,
        cls_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # It has  to be named bacbone because of FineTuner
        self.backbone = VSSMEncoder(**vss_args)
        self.decoder = UNetResDecoder(**decoder_args)
        self.cls_head = cls_head

        self.apply(InitWeights_He(1e-2))
        self.apply(init_last_bn_before_add_to_0)

        if pretrained is not None:
            self.load_pretrained_ckpt(
                num_input_channels=vss_args["in_channels"],
                ckpt_path=pretrained,
            )

    def forward(self, x: torch.Tensor) -> Outputs:
        skips = self.backbone(x)

        return Outputs(
            labels=self.cls_head(skips[-1]) if self.cls_head else None,
            masks=self.decoder(skips).sigmoid().squeeze(1),
        )

    def load_pretrained_ckpt(self, num_input_channels: int, ckpt_path: str) -> None:
        print(f"Loading weights from: {ckpt_path}")
        skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias"]

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_dict = self.state_dict()

        for k, v in ckpt["network_weights"].items():
            p, *k = k.split(".")
            k = ".".join(k)

            if p == "decoder":
                # print(f"Passing weights: {k}", flush=True)
                continue

            if k in skip_params:
                # print(f"Skipping weights: {k}", flush=True)
                continue

            kr = f"vssm_encoder.{k}"

            if (
                "patch_embed" in k
                and "weight" in k
                and "norm" not in k
                and ckpt["network_weights"][kr].shape[1] != num_input_channels
            ):
                # print(f"Passing weights: {k}", flush=True)
                continue

            # if "downsample" in kr:
            #     print("donwsample", flush=True)
            #     i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            #     kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            #     assert kr in model_dict.keys()

            if kr in model_dict:
                assert v.shape == model_dict[kr].shape, (
                    f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
                )
                model_dict["backbone." + ".".join(kr.split(".")[1:])] = v
            # else:
            # print(f"Passing weights: {k}", flush=True)

        self.load_state_dict(model_dict)
