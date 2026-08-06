import torch
import torch.nn as nn

from models.layers.activations import get_activation
from models.layers.norms import get_norm
from models.layers.feedforwards import FeedForwardModule
from models.layers.convs import SamePadDepthwiseConv1d, LocalTFConvBlock


class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1, activation="silu"):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.pointwise_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        self.depthwise = SamePadDepthwiseConv1d(d_model, kernel_size)
        self.norm = get_norm("group", d_model, max_groups=8)
        self.activation = get_activation(activation)

        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        residual = x

        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]

        x = self.pointwise_in(x)
        x = self.glu(x)

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.activation(x)

        x = self.pointwise_out(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # [B, T, D]
        return x


class ConformerSequenceBlock(nn.Module):
    """
    Conformer block for one sequence axis.

    Input/output: [B, L, D]
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ff_expansion_factor=4,
        conv_kernel_size=31,
        dropout=0.1,
        activation="silu",
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )

        self.ff1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout=dropout,
            activation=activation,
        )

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.conv = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            activation=activation,
        )

        self.ff2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout=dropout,
            activation=activation,
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)

        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        x = x + self.conv(x)

        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class FrequencyDPRNNBlock(nn.Module):
    """
    DPRNN-style frequency-axis modeling.

    Input/output: [B, D, F, T]

    For every time frame, this runs a bidirectional RNN over frequency bins.
    """

    def __init__(self, d_model, rnn_hidden=None, dropout=0.1):
        super().__init__()

        if rnn_hidden is None:
            rnn_hidden = d_model // 2

        self.norm = nn.LayerNorm(d_model)

        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(rnn_hidden * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, D, F, T]
        b, d, f, t = x.shape

        residual = x

        # [B, D, F, T] -> [B, T, F, D] -> [B*T, F, D]
        y = x.permute(0, 3, 2, 1).contiguous()
        y = y.view(b * t, f, d)

        y = self.norm(y)
        y, _ = self.rnn(y)
        y = self.proj(y)

        # [B*T, F, D] -> [B, T, F, D] -> [B, D, F, T]
        y = y.view(b, t, f, d)
        y = y.permute(0, 3, 2, 1).contiguous()

        return residual + y


class DualPathConformerBlock(nn.Module):
    """
    One spectrogram block:

    1. Temporal Conformer over T, independently for each frequency bin.
    2. Frequency DPRNN over F, independently for each time frame.
    3. Local 2D TF convolution refinement.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ff_expansion_factor=4,
        temporal_kernel_size=31,
        rnn_hidden=None,
        dropout=0.1,
        activation="silu",
    ):
        super().__init__()

        self.temporal = ConformerSequenceBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_expansion_factor=ff_expansion_factor,
            conv_kernel_size=temporal_kernel_size,
            dropout=dropout,
            activation=activation,
        )

        self.frequency = FrequencyDPRNNBlock(
            d_model=d_model,
            rnn_hidden=rnn_hidden,
            dropout=dropout,
        )

        self.local_tf = LocalTFConvBlock(
            d_model=d_model,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        # x: [B, D, F, T]
        b, d, f, t = x.shape

        # Temporal path: [B, D, F, T] -> [B*F, T, D]
        y = x.permute(0, 2, 3, 1).contiguous()
        y = y.view(b * f, t, d)

        y = self.temporal(y)

        # [B*F, T, D] -> [B, D, F, T]
        y = y.view(b, f, t, d)
        x = y.permute(0, 3, 1, 2).contiguous()

        # Frequency path
        x = self.frequency(x)

        # Local TF refinement
        x = self.local_tf(x)

        return x


class Conformer(nn.Module):
    """
    Dual-path spectrogram Conformer for complex mask prediction.

    Input:  [B, in_channels, F, T]
    Output: [B, out_channels, F, T]
    """

    def __init__(self, cfg):
        super().__init__()

        self.in_channels = cfg.get("in_channels", 8)
        self.out_channels = cfg["out_channels"]
        self.num_freqs = cfg.get("num_freqs", 513)

        self.d_model = cfg.get("tf_channels", min(cfg.get("d_model", 256), 256))

        self.num_layers = cfg.get("num_layers", 4)
        self.num_heads = cfg.get("num_heads", 8)
        self.ff_expansion_factor = cfg.get("ff_expansion_factor", 4)

        self.temporal_kernel_size = cfg.get("conv_kernel_size", 31)
        self.dropout = cfg.get("dropout", 0.1)
        self.activation = cfg.get("activation", "silu")

        self.rnn_hidden = cfg.get("rnn_hidden", self.d_model // 2)

        self.add_identity_mask = cfg.get("add_identity_mask", False)
        self.identity_real_channels = cfg.get("identity_real_channels", [])

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"tf_channels/d_model={self.d_model} must be divisible by "
                f"num_heads={self.num_heads}"
            )

        self.input_projection = nn.Sequential(
            nn.Conv2d(self.in_channels, self.d_model, kernel_size=3, padding=1),
            get_norm("group", self.d_model, max_groups=8),
            get_activation(self.activation),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
            get_norm("group", self.d_model, max_groups=8),
            get_activation(self.activation),
        )

        self.blocks = nn.ModuleList(
            [
                DualPathConformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    ff_expansion_factor=self.ff_expansion_factor,
                    temporal_kernel_size=self.temporal_kernel_size,
                    rnn_hidden=self.rnn_hidden,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_projection = nn.Sequential(
            get_norm("group", self.d_model, max_groups=8),
            get_activation(self.activation),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
            get_norm("group", self.d_model, max_groups=8),
            get_activation(self.activation),
            nn.Conv2d(self.d_model, self.out_channels, kernel_size=1),
        )

        final_conv = self.output_projection[-1]
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Conformer expects input with shape [B, C, F, T]")

        b, c, f, t = x.shape

        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")

        if f != self.num_freqs:
            raise ValueError(f"Expected {self.num_freqs} frequency bins, got {f}")

        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_projection(x)

        if self.add_identity_mask:
            identity_mask = torch.zeros_like(x)

            for ch in self.identity_real_channels:
                if ch < 0 or ch >= self.out_channels:
                    raise ValueError(
                        f"Invalid identity real channel {ch}; "
                        f"out_channels={self.out_channels}"
                    )
                identity_mask[:, ch, :, :] = 1.0

            x = x + identity_mask

        return x
