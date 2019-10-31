from collections.abc import Sequence

import torch.nn as nn
import torch.nn.functional as F


class RCU(nn.Module):
    """
    Paper's Residual Convolution Unit
    """

    def __init__(self, channels):
        super(RCU, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv_unit(x)
        return out + x


class MRF(nn.Module):
    """
    Paper's Multi-Resolution Fusion module
    """

    def __init__(self, encoder_channels, out_channels, upsampling_config=None):
        super(MRF, self).__init__()

        if not isinstance(encoder_channels, Sequence):
            raise TypeError("encoder_channels should be a Sequence")

        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels should have at least 2 values")

        if upsampling_config is None:
            upsampling_config = {'mode': 'nearest', 'align_corners': None}

        self.upsampling_config = upsampling_config

        self.convs = nn.ModuleList([
            nn.Conv2d(n, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            for n in encoder_channels
        ])

    def forward(self, xs):

        if len(xs) != len(self.convs):
            raise ValueError("MRF should have {} inputs, but given {}".format(len(self.convs), len(xs)))

        out_size = [
            max([x.shape[2] for x in xs]),  # h
            max([x.shape[3] for x in xs]),  # w
        ]
        output = self.convs[0](xs[0])
        output = F.interpolate(output, size=out_size, **self.upsampling_config)
        for op, x in zip(self.convs[1:], xs[1:]):
            output += F.interpolate(op(x), size=out_size, **self.upsampling_config)

        return output


class CRP(nn.Module):
    """
    Improved Chain Residual Pooling from
    https://github.com/guosheng/refinenet#network-architecture-and-implementation
    """

    def __init__(self, channels):
        super(CRP, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv_pools = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            )
            for _ in range(4)
        ])

    def forward(self, x):
        output = self.relu(x)
        x = output
        for op in self.conv_pools:
            x = op(x)
            output = output + x

        return output


class LWCRP(CRP):
    """
    Light-weight Chain Residual Pooling from "Light-Weight RefineNet forReal-Time
    Semantic Segmentation"
    """

    def __init__(self, channels):
        super(LWCRP, self).__init__(channels)
        self.relu = nn.ReLU6(inplace=True)
        self.conv_pools = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
            for _ in range(4)
        ])


class RefineNetBlock(nn.Module):

    def __init__(self, encoder_channels, out_channels, upsampling_config=None):
        super(RefineNetBlock, self).__init__()
        if not isinstance(encoder_channels, Sequence):
            raise TypeError("encoder_channels should be a Sequence")

        if len(encoder_channels) == 1 and (out_channels != encoder_channels[0]):
            raise ValueError("out_channels={} should be equal to encoder_channels[0]={}"
                             .format(out_channels, encoder_channels[0]))

        self.rcu_blocks = nn.ModuleList([
            nn.Sequential(
                RCU(c),
                RCU(c)
            )
            for c in encoder_channels
        ])
        self.mrf = MRF(encoder_channels,
                       out_channels=out_channels,
                       upsampling_config=upsampling_config) if len(encoder_channels) > 1 else lambda x: x[0]
        self.crp = CRP(out_channels)
        self.output_conv = RCU(out_channels)

    def forward(self, xs):

        if len(xs) != len(self.rcu_blocks):
            raise ValueError("RefineNetBlock should have {} inputs, but given {}"
                             .format(len(self.rcu_blocks), len(xs)))

        mrf_input = [rcu(x) for rcu, x in zip(self.rcu_blocks, xs)]
        output = self.mrf(mrf_input)
        output = self.crp(output)
        output = self.output_conv(output)
        return output
