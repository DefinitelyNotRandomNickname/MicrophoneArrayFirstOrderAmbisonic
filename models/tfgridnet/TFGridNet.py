import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.activations import get_activation
from models.layers.norms import get_norm_4d


class GridNetBlock(nn.Module):
    """Full-band, sub-band, and cross-frame TF-GridNet processing block."""

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        num_freqs,
        lstm_hidden_units,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        activation="prelu",
        dropout=0.0,
        eps=1e-5,
    ):
        super().__init__()

        if emb_dim < 1 or lstm_hidden_units < 1:
            raise ValueError("Embedding and LSTM dimensions must be positive")
        if emb_ks < 1 or emb_hs < 1 or emb_hs > emb_ks:
            raise ValueError("emb_ks and emb_hs must satisfy 1 <= emb_hs <= emb_ks")
        if num_freqs < 1:
            raise ValueError("num_freqs must be positive")
        if attn_n_head < 1 or emb_dim % attn_n_head != 0:
            raise ValueError("emb_dim must be divisible by attn_n_head")
        if attn_approx_qk_dim < 1:
            raise ValueError("attn_approx_qk_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        recurrent_input_size = emb_dim * emb_ks

        self.intra_norm = get_norm_4d("layer", emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            recurrent_input_size,
            lstm_hidden_units,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_linear = nn.ConvTranspose1d(
            lstm_hidden_units * 2,
            emb_dim,
            kernel_size=emb_ks,
            stride=emb_hs,
        )

        self.inter_norm = get_norm_4d("layer", emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            recurrent_input_size,
            lstm_hidden_units,
            batch_first=True,
            bidirectional=True,
        )
        self.inter_linear = nn.ConvTranspose1d(
            lstm_hidden_units * 2,
            emb_dim,
            kernel_size=emb_ks,
            stride=emb_hs,
        )

        qk_channels = math.ceil(attn_approx_qk_dim / num_freqs)
        value_channels = emb_dim // attn_n_head

        self.query_projections = nn.ModuleList(
            [
                self._attention_projection(
                    emb_dim, qk_channels, num_freqs, activation, eps
                )
                for _ in range(attn_n_head)
            ]
        )
        self.key_projections = nn.ModuleList(
            [
                self._attention_projection(
                    emb_dim, qk_channels, num_freqs, activation, eps
                )
                for _ in range(attn_n_head)
            ]
        )
        self.value_projections = nn.ModuleList(
            [
                self._attention_projection(
                    emb_dim, value_channels, num_freqs, activation, eps
                )
                for _ in range(attn_n_head)
            ]
        )
        self.attention_output = self._attention_projection(
            emb_dim, emb_dim, num_freqs, activation, eps
        )

        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.attn_n_head = attn_n_head
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout

    @staticmethod
    def _attention_projection(in_channels, out_channels, num_freqs, activation, eps):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            get_activation(activation, channels=out_channels),
            get_norm_4d("layer_cf", out_channels, num_freqs, eps=eps),
        )

    def _padded_length(self, length):
        if length <= self.emb_ks:
            return self.emb_ks
        steps = math.ceil((length - self.emb_ks) / self.emb_hs)
        return steps * self.emb_hs + self.emb_ks

    def _run_intra_rnn(self, x):
        batch_size, channels, frames, num_freqs = x.shape
        sequence = x.transpose(1, 2).reshape(batch_size * frames, channels, num_freqs)
        sequence = F.unfold(
            sequence.unsqueeze(-1),
            kernel_size=(self.emb_ks, 1),
            stride=(self.emb_hs, 1),
        ).transpose(1, 2)
        sequence, _ = self.intra_rnn(sequence)
        sequence = self.intra_linear(sequence.transpose(1, 2))
        return sequence.view(batch_size, frames, channels, num_freqs).transpose(1, 2)

    def _run_inter_rnn(self, x):
        batch_size, channels, frames, num_freqs = x.shape
        sequence = x.permute(0, 3, 1, 2).reshape(
            batch_size * num_freqs, channels, frames
        )
        sequence = F.unfold(
            sequence.unsqueeze(-1),
            kernel_size=(self.emb_ks, 1),
            stride=(self.emb_hs, 1),
        ).transpose(1, 2)
        sequence, _ = self.inter_rnn(sequence)
        sequence = self.inter_linear(sequence.transpose(1, 2))
        return sequence.view(batch_size, num_freqs, channels, frames).permute(
            0, 2, 3, 1
        )

    def _run_attention(self, x):
        queries = torch.stack(
            [projection(x) for projection in self.query_projections], 1
        )
        keys = torch.stack([projection(x) for projection in self.key_projections], 1)
        values = torch.stack(
            [projection(x) for projection in self.value_projections], 1
        )

        batch_size, _, _, frames, num_freqs = queries.shape
        queries = queries.permute(0, 1, 3, 2, 4).flatten(start_dim=3)
        keys = keys.permute(0, 1, 3, 2, 4).flatten(start_dim=3)

        value_channels = values.size(2)
        values = values.permute(0, 1, 3, 2, 4).flatten(start_dim=3)

        attention = torch.matmul(queries, keys.transpose(-1, -2))
        attention = F.softmax(attention / math.sqrt(queries.size(-1)), dim=-1)
        attention = F.dropout(
            attention,
            p=self.attention_dropout,
            training=self.training,
        )
        attended = torch.matmul(attention, values)
        attended = attended.view(
            batch_size,
            self.attn_n_head,
            frames,
            value_channels,
            num_freqs,
        )
        attended = attended.permute(0, 1, 3, 2, 4).reshape(
            batch_size, -1, frames, num_freqs
        )
        return self.attention_output(attended)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("GridNetBlock expects input with shape [B, C, T, F]")

        original_frames = x.size(2)
        original_freqs = x.size(3)
        padded_frames = self._padded_length(original_frames)
        padded_freqs = self._padded_length(original_freqs)
        x = F.pad(
            x,
            (0, padded_freqs - original_freqs, 0, padded_frames - original_frames),
        )

        residual = x
        x = residual + self.dropout(self._run_intra_rnn(self.intra_norm(x)))

        residual = x
        x = residual + self.dropout(self._run_inter_rnn(self.inter_norm(x)))

        x = x[:, :, :original_frames, :original_freqs]
        return x + self.dropout(self._run_attention(x))


class TFGridNet(nn.Module):
    """TF-GridNet complex spectral mapper for microphone-array FOA prediction.

    The architecture follows the full-band, sub-band, and cross-frame design in
    Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech
    Separation" (2023), adapted to the repository's real/imaginary channel API.

    Input:  [B, in_channels, F, T]
    Output: [B, out_channels, F, T]
    """

    def __init__(self, cfg):
        super().__init__()

        self.in_channels = int(cfg.get("in_channels", 8))
        self.out_channels = int(cfg["out_channels"])
        self.num_freqs = int(cfg.get("num_freqs", 513))
        self.emb_dim = int(cfg.get("emb_dim", 48))
        self.num_layers = int(cfg.get("num_layers", 6))
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

        activation = cfg.get("activation", "prelu")
        dropout = float(cfg.get("dropout", 0.0))
        emb_ks = int(cfg.get("emb_ks", 4))
        emb_hs = int(cfg.get("emb_hs", 1))
        hidden_units = int(cfg.get("lstm_hidden_units", 192))
        attention_heads = int(cfg.get("attn_n_head", 4))
        attention_dim = int(cfg.get("attn_approx_qk_dim", 512))

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
                GridNetBlock(
                    emb_dim=self.emb_dim,
                    emb_ks=emb_ks,
                    emb_hs=emb_hs,
                    num_freqs=self.num_freqs,
                    lstm_hidden_units=hidden_units,
                    attn_n_head=attention_heads,
                    attn_approx_qk_dim=attention_dim,
                    activation=activation,
                    dropout=dropout,
                    eps=self.eps,
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
            raise ValueError("TFGridNet expects input with shape [B, C, F, T]")
        if torch.is_complex(x):
            raise ValueError("TFGridNet expects real/imaginary values as channels")
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

        x = x.transpose(2, 3).contiguous()
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_projection(x)
        x = x.transpose(2, 3).contiguous()

        if scale is not None:
            x = x * scale

        return x
