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

    def __init__(self, encoder_channels, out_channels):
        super(MRF, self).__init__()

        if not isinstance(encoder_channels, Sequence):
            raise TypeError("encoder_channels should be a Sequence")

        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels should have at least 2 values")

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
        output = F.interpolate(output, size=out_size, mode='nearest')
        for op, x in zip(self.convs[1:], xs[1:]):
            output += F.interpolate(op(x), size=out_size, mode='nearest')

        return output


class CRP(nn.Module):

    def __init__(self, channels):
        super(CRP, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            )
            for _ in range(4)
        ])

    def forward(self, x):
        output = self.relu(x)
        x = output
        for op in self.pool_convs:
            x = op(x)
            output = output + x

        return output


class RefineNetBlock(nn.Module):

    def __init__(self, encoder_channels, out_channels):
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
        self.mrf = MRF(encoder_channels, out_channels=out_channels) if len(encoder_channels) > 1 else lambda x: x[0]
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
