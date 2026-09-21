import torch.nn as nn
import torch.nn.functional as F

from models.layers.activations import get_activation
from models.layers.norms import get_norm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation, norm_2d, repeats):
        super().__init__()

        padding = kernel_size // 2

        layers = []

        in_channels = in_ch
        for _ in range(repeats):
            layers += [
                nn.Conv2d(
                    in_channels, out_ch, kernel_size=kernel_size, padding=padding
                ),
                get_norm(norm_2d, out_ch),
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


class LocalTFConvBlock(nn.Module):
    """
    Lightweight local 2D TF refinement.

    Input/output: [B, D, F, T]
    """

    def __init__(self, d_model, dropout=0.1, activation="silu"):
        super().__init__()

        self.net = nn.Sequential(
            get_norm("group", d_model, max_groups=8),
            get_activation(activation),
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=3,
                padding=1,
                groups=d_model,
            ),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution that only sees the current and past positions.

    Input/output: [B, C, L]
    """

    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x)


class GroupedConvModule1d(nn.Module):
    """Layer norm, same-padded grouped 1D convolution, and activation.

    Input/output: [B, L, C]
    """

    def __init__(self, channels, kernel_size, groups=1, activation="prelu", eps=1e-5):
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if groups < 1 or channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        self.norm = nn.LayerNorm(channels, eps=eps)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding="same",
            groups=groups,
        )
        self.activation = get_activation(activation, channels=channels)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("GroupedConvModule1d expects input with shape [B, L, C]")

        x = self.norm(x).transpose(1, 2)
        x = self.activation(self.conv(x))
        return x.transpose(1, 2)
