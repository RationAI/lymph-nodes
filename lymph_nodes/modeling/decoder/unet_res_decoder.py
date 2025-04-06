import math
from typing import List, Tuple, Union
from torch import nn
import torch

from lymph_nodes.modeling.blocks import PatchExpand, VSSLayer, FinalPatchExpand_X4


class UNetResDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        deep_supervision,
        features_per_stage: Union[Tuple[int, ...], List[int]],
        drop_path_rate: float = 0.2,
        d_state: int = 16,
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()

        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        dpr = [
            x.item()
            for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)
        ]
        depths = [2, 2, 2, 2]

        # we start with the bottleneck and work out way up
        stages = []
        expand_layers = []
        seg_layers = []
        concat_back_dim = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            expand_layers.append(
                PatchExpand(
                    dim=input_features_below,
                    dim_scale=2,
                    norm_layer=nn.LayerNorm,
                )
            )
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(
                VSSLayer(
                    dim=input_features_skip,
                    depth=2,
                    attn_drop=0.0,
                    drop_path=dpr[sum(depths[: s - 1]) : sum(depths[:s])],
                    d_state=math.ceil(2 * input_features_skip / 6)
                    if d_state is None
                    else d_state,
                    norm_layer=nn.LayerNorm,
                    downsample=None,
                    use_checkpoint=False,
                )
            )
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1))
            concat_back_dim.append(
                nn.Linear(2 * input_features_skip, input_features_skip)
            )

        # for final prediction
        expand_layers.append(
            FinalPatchExpand_X4(
                dim=encoder_output_channels[0],
                dim_scale=4,
                norm_layer=nn.LayerNorm,
            )
        )
        stages.append(nn.Identity())
        seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1))

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

    def forward(
        self, skips: list[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
            x = self.stages[s](x).permute(0, 3, 1, 2)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
