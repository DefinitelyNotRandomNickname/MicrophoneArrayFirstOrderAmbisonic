import torch.nn as nn

from models.layers import get_activation, get_norm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation, norm, repeats):
        super().__init__()

        padding = kernel_size // 2

        layers = []

        in_channels = in_ch
        for _ in range(repeats):
            layers += [
                nn.Conv2d(in_channels, out_ch, kernel_size=kernel_size, padding=padding),
                get_norm(norm, out_ch),
                get_activation(activation),
            ]
            in_channels = out_ch

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
