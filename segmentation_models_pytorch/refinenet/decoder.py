from collections.abc import Sequence

import torch.nn as nn
import torch.nn.functional as F

from ..base.model import Model
from .blocks import RefineNetBlock, RCU, LWCRP


class RefineNetDecoder(Model):
    """Decoder based on RefineNet architecture described in
    'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation',
    https://arxiv.org/abs/1611.06612
    """

    def __init__(self,
                 encoder_channels,
                 num_refine_channels=256,
                 final_channels=10,
                 output_upsampling_factor=None,
                 upsampling_config=None):
        super(RefineNetDecoder, self).__init__()
        if not isinstance(encoder_channels, Sequence):
            raise TypeError("encoder_channels should be a Sequence")

        if upsampling_config is None:
            upsampling_config = {'mode': 'bilinear', 'align_corners': True}

        self.upsampling_config = upsampling_config

        refine_channels = [int(num_refine_channels * (1 + c / num_refine_channels // 8)) for c in encoder_channels]
        # for num_refine_channels=256 and encoder_channels=reversed([256, 512, 1024, 2048])
        # e.g refine_channels = reversed([256, 256, 256, 512])

        adaptation_blocks = []
        refinenet_blocks = []
        prev_c, prev_r = None, None
        for c, r in (zip(encoder_channels, refine_channels)):
            c_list = [c, ]
            r_list = [r, ]
            if prev_c is not None:
                c_list.append(prev_c)

            if prev_c is not None:
                r_list.append(prev_r)

            adaptation_blocks.append(
                nn.Conv2d(c, r, kernel_size=3, stride=1, padding=1, bias=False),
            )
            refinenet_blocks.append(
                RefineNetBlock(r_list, out_channels=r)
            )
            prev_c = c
            prev_r = r

        self.adaptation_blocks = nn.ModuleList(adaptation_blocks)
        self.refinenet_blocks = nn.ModuleList(refinenet_blocks)
        self.classifier = nn.Sequential(
            RCU(num_refine_channels),
            RCU(num_refine_channels),
            nn.Conv2d(num_refine_channels, final_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.output_upsampling_factor = output_upsampling_factor

    def refine(self, x):
        output = None
        for i, adapt, block in zip(x, self.adaptation_blocks, self.refinenet_blocks):
            i = adapt(i)
            inputs = [i, ]
            if output is not None:
                inputs.append(output)
            output = block(inputs)
        return output

    def forward(self, x):
        if not (isinstance(x, Sequence) and len(x) == len(self.refinenet_blocks)):
            raise TypeError("Input x should be a sequence of len {}, but given: {} (len={})"
                            .format(len(self.refinenet_blocks), type(x), len(x) if isinstance(x, Sequence) else None))
        output = self.refine(x)
        output = self.classifier(output)

        if self.output_upsampling_factor is not None:
            output = F.interpolate(output, scale_factor=self.output_upsampling_factor, **self.upsampling_config)
        return output


class LightWeightRefineNetDecoder(Model):
    """Decoder based on Light-Weight RefineNet architecture described in
    'Light-Weight RefineNet for Real-Time Semantic Segmentation',
    https://arxiv.org/abs/1810.03272
    """

    num_inputs = 6

    def __init__(self,
                 encoder_channels,
                 num_refine_channels=256,
                 final_channels=10,
                 output_upsampling_factor=None,
                 upsampling_config=None):
        super(LightWeightRefineNetDecoder, self).__init__()
        if not isinstance(encoder_channels, Sequence):
            raise TypeError("encoder_channels should be a Sequence")

        if len(encoder_channels) != self.num_inputs:
            raise TypeError("encoder_channels should contain 6 entries")

        if upsampling_config is None:
            upsampling_config = {'mode': 'bilinear', 'align_corners': True}

        self.upsampling_config = upsampling_config

        adaptation_blocks = []
        conv_params = dict(kernel_size=1, stride=1, padding=0, bias=False)
        for c in encoder_channels:
            adaptation_blocks.append(
                nn.Conv2d(c, num_refine_channels, **conv_params),
            )

        crp_conv_blocks = []
        for _ in range(3):
            crp_conv_blocks.append(
                nn.Sequential(
                    LWCRP(num_refine_channels),
                    nn.Conv2d(num_refine_channels, num_refine_channels, **conv_params),
                )
            )
        crp_conv_blocks.append(LWCRP(num_refine_channels))

        self.adaptation_blocks = nn.ModuleList(adaptation_blocks)
        self.crp_conv_blocks = nn.ModuleList(crp_conv_blocks)
        self.classifier = nn.Conv2d(num_refine_channels, final_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_upsampling_factor = output_upsampling_factor

    def refine(self, x):
        y = [None] * len(x)
        for i, adapt in enumerate(self.adaptation_blocks):
            y[i] = adapt(x[i])

        # sum up 4 top outputs
        z = [y[0] + y[1],
             y[2] + y[3],
             y[4],
             y[5]]

        output = None
        for i, block in zip(z, self.crp_conv_blocks):
            if output is not None:
                output = F.interpolate(output, size=(i.shape[2], i.shape[3]), **self.upsampling_config)
                i = output + i
            output = block(i)
        return output

    def forward(self, x):
        if not (isinstance(x, Sequence) and len(x) == len(self.adaptation_blocks)):
            raise TypeError("Input x should be a sequence of len {}, but given: {} (len={})"
                            .format(len(self.adaptation_blocks), type(x), len(x) if isinstance(x, Sequence) else None))
        output = self.refine(x)
        output = self.classifier(output)

        if self.output_upsampling_factor is not None:
            output = F.interpolate(output, scale_factor=self.output_upsampling_factor,
                                   **self.upsampling_config)
        return output


class LightWeightRefineNetDecoderV2(LightWeightRefineNetDecoder):
    """Another decoder based on Light-Weight RefineNet architecture described in
    'Light-Weight RefineNet for Real-Time Semantic Segmentation',
    https://arxiv.org/abs/1810.03272

    This decoder works on 5 encoder entries.
    """

    num_inputs = 5

    def refine(self, x):
        y = [None] * len(x)
        for i, adapt in enumerate(self.adaptation_blocks):
            y[i] = adapt(x[i])

        output = None
        for i, block in zip(y, self.crp_conv_blocks):
            if output is not None:
                output = F.interpolate(output, size=(i.shape[2], i.shape[3]), **self.upsampling_config)
                i = output + i
            output = block(i)
        return output
