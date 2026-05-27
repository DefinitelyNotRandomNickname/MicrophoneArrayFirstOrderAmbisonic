import torch.nn as nn
import torch.nn.functional as F

from models.layers.activations import get_activation
from models.layers.norms import get_norm_2d


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation, norm_2d, repeats):
        super().__init__()

        padding = kernel_size // 2

        layers = []

        in_channels = in_ch
        for _ in range(repeats):
            layers += [
                nn.Conv2d(in_channels, out_ch, kernel_size=kernel_size, padding=padding),
                get_norm_2d(norm_2d, out_ch),
                get_activation(activation),
            ]
            in_channels = out_ch

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SamePadDepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
        )

    def forward(self, x):
        total_padding = self.kernel_size - 1
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        x = F.pad(x, (left_padding, right_padding))
        return self.conv(x)
