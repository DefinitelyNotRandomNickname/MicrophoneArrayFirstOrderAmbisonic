import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.convs import ConvBlock


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation, norm, repeats):
        super().__init__()

        self.conv = ConvBlock(in_ch, out_ch, kernel_size, activation, norm, repeats)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation, norm, repeats):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=stride, stride=stride)
        self.conv = ConvBlock(in_ch, out_ch, kernel_size, activation, norm, repeats)

    def forward(self, x, skip):
        x = self.up(x)

        # padding alignment
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.out_channels = cfg["out_channels"]
        self.activation = cfg["activation"]
        self.norm = cfg["norm"]
        self.repeats = cfg["repeats"]

        self.channels = cfg["channels"]

        self.kernel_size = cfg.get("kernel_size", 3)
        self.stride = cfg.get("stride", 2)

        # Encoder
        self.down_blocks = nn.ModuleList()

        for in_ch, out_ch in self.channels:
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    self.kernel_size,
                    self.stride,
                    self.activation,
                    self.norm,
                    self.repeats,
                )
            )

        # Bottleneck
        last_ch = self.channels[-1][1]
        self.bottleneck = ConvBlock(
            last_ch,
            last_ch * 2,
            self.kernel_size,
            self.activation,
            self.norm,
            self.repeats,
        )

        # Decoder
        self.up_blocks = nn.ModuleList()

        rev_channels = list(reversed(self.channels))

        in_ch = last_ch * 2
        for _, skip_out in rev_channels:
            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    skip_out,
                    self.kernel_size,
                    self.stride,
                    self.activation,
                    self.norm,
                    self.repeats,
                )
            )
            in_ch = skip_out

        # Output
        self.final_conv = nn.Conv2d(in_ch, self.out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.down_blocks:
            s, x = down(x)
            skips.append(s)

        x = self.bottleneck(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        x = self.final_conv(x)

        return x
