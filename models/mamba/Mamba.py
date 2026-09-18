import math

import torch
import torch.nn as nn

from models.mamba.mixers import BidirectionalMamba, resolve_dt_rank
from models.mamba.ssm import SCAN_BACKENDS


class DualPathMambaBlock(nn.Module):
    """Intra-frame (frequency) and inter-frame (time) bidirectional Mamba block.

    Input/output: [B, T, F, C]
    """

    def __init__(
        self,
        emb_dim,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        activation="silu",
        dropout=0.0,
        eps=1e-5,
        scan_backend="auto",
    ):
        super().__init__()

        if emb_dim < 1:
            raise ValueError("emb_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_mamba = BidirectionalMamba(
            emb_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            activation=activation,
            scan_backend=scan_backend,
        )
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_mamba = BidirectionalMamba(
            emb_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            activation=activation,
            scan_backend=scan_backend,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("DualPathMambaBlock expects input with shape [B, T, F, C]")

        batch_size, frames, num_freqs, channels = x.shape

        # Intra-frame path: every frame is a sequence over frequency bins.
        sequence = x.reshape(batch_size * frames, num_freqs, channels)
        sequence = sequence + self.dropout(self.intra_mamba(self.intra_norm(sequence)))
        x = sequence.view(batch_size, frames, num_freqs, channels)

        # Inter-frame path: every frequency bin is a sequence over frames.
        sequence = x.transpose(1, 2).reshape(batch_size * num_freqs, frames, channels)
        sequence = sequence + self.dropout(self.inter_mamba(self.inter_norm(sequence)))
        return sequence.view(batch_size, num_freqs, frames, channels).transpose(1, 2)


class Mamba(nn.Module):
    """Dual-path bidirectional Mamba complex spectral mapper for FOA prediction.

    Every block runs a bidirectional selective state-space (Mamba) mixer along
    the frequency axis of each frame and then along the time axis of each bin,
    in the spirit of dual-path TF models such as TF-GridNet but with the RNN and
    attention replaced by Mamba (Gu & Dao, 2023).

    Input:  [B, in_channels, F, T]
    Output: [B, out_channels, F, T]
    """

    def __init__(self, cfg):
        super().__init__()

        self.in_channels = int(cfg.get("in_channels", 8))
        self.out_channels = int(cfg["out_channels"])
        self.num_freqs = int(cfg.get("num_freqs", 513))
        self.emb_dim = int(cfg.get("emb_dim", 64))
        self.num_layers = int(cfg.get("num_layers", 4))
        self.normalize_input = cfg.get("normalize_input", False)
        self.eps = float(cfg.get("eps", 1e-5))

        if self.in_channels < 2 or self.in_channels % 2 != 0:
            raise ValueError("in_channels must contain real/imaginary channel pairs")
        if self.out_channels < 2 or self.out_channels % 2 != 0:
            raise ValueError("out_channels must contain real/imaginary channel pairs")
        if self.num_freqs < 1 or self.num_layers < 1 or self.emb_dim < 1:
            raise ValueError("num_freqs, num_layers, and emb_dim must be positive")
        if not isinstance(self.normalize_input, bool):
            raise ValueError("normalize_input must be a boolean")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        d_state = int(cfg.get("d_state", 16))
        d_conv = int(cfg.get("d_conv", 4))
        expand = float(cfg.get("expand", 2))
        dt_rank = cfg.get("dt_rank", "auto")
        activation = cfg.get("activation", "silu")
        dropout = float(cfg.get("dropout", 0.0))
        scan_backend = cfg.get("scan_backend", "auto")

        if d_state < 1 or d_conv < 1:
            raise ValueError("d_state and d_conv must be positive")
        if int(expand * self.emb_dim) < 1:
            raise ValueError("expand * emb_dim must be at least 1")
        resolve_dt_rank(dt_rank, self.emb_dim)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if scan_backend not in SCAN_BACKENDS:
            raise ValueError(f"Unsupported scan backend: {scan_backend}")

        self.input_projection = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.emb_dim,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.GroupNorm(1, self.emb_dim, eps=self.eps),
        )
        self.blocks = nn.ModuleList(
            [
                DualPathMambaBlock(
                    emb_dim=self.emb_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    activation=activation,
                    dropout=dropout,
                    eps=self.eps,
                    scan_backend=scan_backend,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_projection = nn.ConvTranspose2d(
            self.emb_dim,
            self.out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Mamba expects input with shape [B, C, F, T]")
        if torch.is_complex(x):
            raise ValueError("Mamba expects real/imaginary values as channels")
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

        x = self.input_projection(x)

        # [B, C, F, T] -> [B, T, F, C]
        x = x.permute(0, 3, 2, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 2, 1).contiguous()

        x = self.output_projection(x)

        if scale is not None:
            x = x * scale

        return x
