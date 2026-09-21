import math

import torch
import torch.nn as nn

from models.layers.activations import get_activation
from models.layers.convs import GroupedConvModule1d
from models.layers.feedforwards import ConvFeedForwardModule
from models.layers.linears import GroupedLinear


class CrossBandBlock(nn.Module):
    """Frequency-convolutional and full-band linear modules of SpatialNet.

    Every time frame is processed as a sequence over frequency bins: two grouped
    convolutions capture local cross-band correlations and a squeezed grouped
    linear map mixes all frequency bins of the frame at once. The grouped linear
    map may be shared between layers to keep the parameter count low.

    Input/output: [B, F, T, H]
    """

    def __init__(
        self,
        dim_hidden,
        dim_squeeze,
        num_freqs,
        kernel_size=5,
        conv_groups=8,
        activation="silu",
        conv_activation="prelu",
        dropout=0.0,
        eps=1e-5,
        full_band=None,
    ):
        super().__init__()

        if dim_hidden < 1 or dim_squeeze < 1 or num_freqs < 1:
            raise ValueError("dim_hidden, dim_squeeze, and num_freqs must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        if full_band is None:
            full_band = GroupedLinear(num_freqs, num_freqs, num_groups=dim_squeeze)
        elif (
            full_band.num_groups != dim_squeeze
            or full_band.in_features != num_freqs
            or full_band.out_features != num_freqs
        ):
            raise ValueError(
                "Shared full-band module must map num_freqs to num_freqs "
                "with dim_squeeze groups"
            )

        self.num_freqs = num_freqs

        self.freq_conv_in = GroupedConvModule1d(
            dim_hidden,
            kernel_size,
            groups=conv_groups,
            activation=conv_activation,
            eps=eps,
        )

        self.full_band_norm = nn.LayerNorm(dim_hidden, eps=eps)
        self.squeeze = nn.Sequential(
            nn.Linear(dim_hidden, dim_squeeze),
            get_activation(activation),
        )
        self.full_band_dropout = nn.Dropout2d(dropout)
        self.full_band = full_band
        self.unsqueeze = nn.Sequential(
            nn.Linear(dim_squeeze, dim_hidden),
            get_activation(activation),
        )

        self.freq_conv_out = GroupedConvModule1d(
            dim_hidden,
            kernel_size,
            groups=conv_groups,
            activation=conv_activation,
            eps=eps,
        )

    @staticmethod
    def _along_frequency(module, x):
        batch_size, num_freqs, frames, channels = x.shape
        sequence = x.transpose(1, 2).reshape(batch_size * frames, num_freqs, channels)
        sequence = module(sequence)
        return sequence.reshape(batch_size, frames, num_freqs, channels).transpose(1, 2)

    def _full_band(self, x):
        x = self.squeeze(self.full_band_norm(x))
        # Dropout2d treats frequency as the channel axis, so it drops whole bins.
        x = self.full_band_dropout(x)
        # [B, F, T, S] -> [B, T, S, F]: every squeezed channel mixes all bins.
        x = self.full_band(x.permute(0, 2, 3, 1))
        return self.unsqueeze(x.permute(0, 3, 1, 2))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("CrossBandBlock expects input with shape [B, F, T, H]")
        if x.size(1) != self.num_freqs:
            raise ValueError(
                f"Expected {self.num_freqs} frequency bins, got {x.size(1)}"
            )

        x = x + self._along_frequency(self.freq_conv_in, x)
        x = x + self._full_band(x)
        return x + self._along_frequency(self.freq_conv_out, x)


class NarrowBandBlock(nn.Module):
    """Time self-attention and convolutional feed-forward modules of SpatialNet.

    Every frequency bin is processed as a sequence over time frames, so the
    block learns narrow-band spatial cues that stay stable along time.

    Input/output: [B, F, T, H]
    """

    def __init__(
        self,
        dim_hidden,
        num_heads=4,
        ff_expansion_factor=2,
        kernel_size=3,
        conv_groups=8,
        activation="silu",
        dropout=0.0,
        eps=1e-5,
    ):
        super().__init__()

        if dim_hidden < 1:
            raise ValueError("dim_hidden must be positive")
        if num_heads < 1 or dim_hidden % num_heads != 0:
            raise ValueError("dim_hidden must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        self.attention_norm = nn.LayerNorm(dim_hidden, eps=eps)
        self.attention = nn.MultiheadAttention(
            dim_hidden,
            num_heads,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward = ConvFeedForwardModule(
            dim_hidden,
            ff_expansion_factor,
            dropout,
            activation,
            kernel_size=kernel_size,
            groups=conv_groups,
            eps=eps,
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("NarrowBandBlock expects input with shape [B, F, T, H]")

        batch_size, num_freqs, frames, channels = x.shape
        sequence = x.reshape(batch_size * num_freqs, frames, channels)

        attention_input = self.attention_norm(sequence)
        attended, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        sequence = sequence + self.attention_dropout(attended)
        sequence = sequence + self.feed_forward(sequence)

        return sequence.view(batch_size, num_freqs, frames, channels)


class SpatialNetLayer(nn.Module):
    """Cross-band block followed by a narrow-band block.

    Input/output: [B, F, T, H]
    """

    def __init__(
        self,
        dim_hidden,
        dim_squeeze,
        num_freqs,
        num_heads=4,
        ff_expansion_factor=2,
        freq_kernel_size=5,
        time_kernel_size=3,
        freq_conv_groups=8,
        time_conv_groups=8,
        activation="silu",
        conv_activation="prelu",
        dropout=0.0,
        eps=1e-5,
        full_band=None,
    ):
        super().__init__()

        self.cross_band = CrossBandBlock(
            dim_hidden=dim_hidden,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs,
            kernel_size=freq_kernel_size,
            conv_groups=freq_conv_groups,
            activation=activation,
            conv_activation=conv_activation,
            dropout=dropout,
            eps=eps,
            full_band=full_band,
        )
        self.narrow_band = NarrowBandBlock(
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ff_expansion_factor=ff_expansion_factor,
            kernel_size=time_kernel_size,
            conv_groups=time_conv_groups,
            activation=activation,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, x):
        return self.narrow_band(self.cross_band(x))


class SpatialNet(nn.Module):
    """SpatialNet complex spectral mapper for microphone-array FOA prediction.

    The architecture follows Quan & Li, "SpatialNet: Extensively Learning
    Spatial Information for Multichannel Joint Speech Separation, Denoising and
    Dereverberation" (2024). A temporal convolution encodes every frequency bin,
    a stack of layers alternates cross-band blocks (frequency convolutions and a
    full-band grouped linear map, shared across layers by default) with
    narrow-band blocks (time self-attention and a convolutional feed-forward
    network), and a linear decoder maps back to real/imaginary channels.

    Input:  [B, in_channels, F, T]
    Output: [B, out_channels, F, T]
    """

    def __init__(self, cfg):
        super().__init__()

        self.in_channels = int(cfg.get("in_channels", 8))
        self.out_channels = int(cfg["out_channels"])
        self.num_freqs = int(cfg.get("num_freqs", 513))
        self.dim_hidden = int(cfg.get("dim_hidden", 96))
        self.num_layers = int(cfg.get("num_layers", 8))
        self.normalize_input = cfg.get("normalize_input", False)
        self.eps = float(cfg.get("eps", 1e-5))

        if self.in_channels < 2 or self.in_channels % 2 != 0:
            raise ValueError("in_channels must contain real/imaginary channel pairs")
        if self.out_channels < 2 or self.out_channels % 2 != 0:
            raise ValueError("out_channels must contain real/imaginary channel pairs")
        if self.num_freqs < 1 or self.num_layers < 1 or self.dim_hidden < 1:
            raise ValueError("num_freqs, num_layers, and dim_hidden must be positive")
        if not isinstance(self.normalize_input, bool):
            raise ValueError("normalize_input must be a boolean")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        dim_squeeze = int(cfg.get("dim_squeeze", 8))
        num_heads = int(cfg.get("num_heads", 4))
        ff_expansion_factor = float(cfg.get("ff_expansion_factor", 2))
        encoder_kernel_size = int(cfg.get("encoder_kernel_size", 5))
        freq_kernel_size = int(cfg.get("freq_kernel_size", 5))
        time_kernel_size = int(cfg.get("time_kernel_size", 3))
        freq_conv_groups = int(cfg.get("freq_conv_groups", 8))
        time_conv_groups = int(cfg.get("time_conv_groups", 8))
        share_full_band = cfg.get("share_full_band", True)
        activation = cfg.get("activation", "silu")
        conv_activation = cfg.get("conv_activation", "prelu")
        dropout = float(cfg.get("dropout", 0.0))

        if encoder_kernel_size < 1:
            raise ValueError("encoder_kernel_size must be positive")
        if not isinstance(share_full_band, bool):
            raise ValueError("share_full_band must be a boolean")

        self.encoder = nn.Conv1d(
            self.in_channels,
            self.dim_hidden,
            kernel_size=encoder_kernel_size,
            padding="same",
        )

        full_band = None
        if share_full_band:
            full_band = GroupedLinear(
                self.num_freqs, self.num_freqs, num_groups=dim_squeeze
            )

        self.layers = nn.ModuleList(
            [
                SpatialNetLayer(
                    dim_hidden=self.dim_hidden,
                    dim_squeeze=dim_squeeze,
                    num_freqs=self.num_freqs,
                    num_heads=num_heads,
                    ff_expansion_factor=ff_expansion_factor,
                    freq_kernel_size=freq_kernel_size,
                    time_kernel_size=time_kernel_size,
                    freq_conv_groups=freq_conv_groups,
                    time_conv_groups=time_conv_groups,
                    activation=activation,
                    conv_activation=conv_activation,
                    dropout=dropout,
                    eps=self.eps,
                    full_band=full_band,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.decoder = nn.Linear(self.dim_hidden, self.out_channels)

        nn.init.normal_(self.decoder.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("SpatialNet expects input with shape [B, C, F, T]")
        if torch.is_complex(x):
            raise ValueError("SpatialNet expects real/imaginary values as channels")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )
        if x.size(2) != self.num_freqs:
            raise ValueError(
                f"Expected {self.num_freqs} frequency bins, got {x.size(2)}"
            )

        scale = None
        if self.normalize_input:
            scale = x.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            scale = scale.clamp_min(self.eps)
            x = x / scale

        batch_size, channels, num_freqs, frames = x.shape

        # Temporal encoder for every frequency bin: [B, C, F, T] -> [B*F, C, T]
        x = x.transpose(1, 2).reshape(batch_size * num_freqs, channels, frames)
        x = self.encoder(x)
        # [B*F, H, T] -> [B, F, T, H]
        x = x.transpose(1, 2).reshape(batch_size, num_freqs, frames, self.dim_hidden)

        for layer in self.layers:
            x = layer(x)

        # [B, F, T, H] -> [B, F, T, C_out] -> [B, C_out, F, T]
        x = self.decoder(x).permute(0, 3, 1, 2).contiguous()

        if scale is not None:
            x = x * scale

        return x
