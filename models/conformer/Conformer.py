import torch.nn as nn

from models.conformer.encoders import ConformerEncoder
from models.layers import SinusoidalPositionalEncoding, get_activation


class Conformer(nn.Module):
    """
    Spectrogram Conformer for MEMS-to-FOA complex mask prediction.

    Input:  [B, in_channels, F, T]
    Output: [B, out_channels, F, T]

    Each time frame is treated as one token with all microphone/RI/frequency
    bins flattened into the feature dimension.
    """

    def __init__(self, cfg):
        super().__init__()

        self.in_channels = cfg.get("in_channels", 8)
        self.out_channels = cfg["out_channels"]
        self.num_freqs = cfg.get("num_freqs", 513)
        self.d_model = cfg.get("d_model", 128)
        self.num_layers = cfg.get("num_layers", 6)
        self.num_heads = cfg.get("num_heads", 4)
        self.ff_expansion_factor = cfg.get("ff_expansion_factor", 4)
        self.conv_kernel_size = cfg.get("conv_kernel_size", 31)
        self.dropout = cfg.get("dropout", 0.1)
        self.activation = cfg.get("activation", "silu")
        self.conv_norm = cfg.get("conv_norm", "batch")
        self.positional_encoding = cfg.get("positional_encoding", "sinusoidal")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        input_dim = self.in_channels * self.num_freqs
        output_dim = self.out_channels * self.num_freqs

        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.d_model),
            get_activation(self.activation),
            nn.Dropout(self.dropout),
        )

        if self.positional_encoding == "sinusoidal":
            self.position = SinusoidalPositionalEncoding(self.d_model)
        elif self.positional_encoding in (None, "none", "identity"):
            self.position = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported positional_encoding: {self.positional_encoding}"
            )

        self.encoder = ConformerEncoder(
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_expansion_factor=self.ff_expansion_factor,
            conv_kernel_size=self.conv_kernel_size,
            dropout=self.dropout,
            activation=self.activation,
            conv_norm=self.conv_norm,
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Conformer expects input with shape [B, C, F, T]")

        b, c, f, t = x.shape

        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")
        if f != self.num_freqs:
            raise ValueError(f"Expected {self.num_freqs} frequency bins, got {f}")

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * f)

        x = self.input_projection(x)
        x = self.position(x)

        x = self.encoder(x)

        x = self.output_projection(x)
        x = x.view(b, t, self.out_channels, f)

        return x.permute(0, 2, 3, 1).contiguous()
