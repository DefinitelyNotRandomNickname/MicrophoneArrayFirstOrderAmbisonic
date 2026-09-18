import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.activations import get_activation
from models.layers.convs import CausalDepthwiseConv1d
from models.mamba.ssm import SCAN_BACKENDS, selective_scan


def resolve_dt_rank(dt_rank, d_model):
    if dt_rank == "auto":
        return math.ceil(d_model / 16)
    if isinstance(dt_rank, bool) or not isinstance(dt_rank, int) or dt_rank < 1:
        raise ValueError("dt_rank must be 'auto' or a positive integer")
    return dt_rank


class MambaMixer(nn.Module):
    """Single-direction Mamba (S6) mixer from Gu & Dao, "Mamba: Linear-Time
    Sequence Modeling with Selective State Spaces" (2023).

    Input/output: [B, L, D]
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        activation="silu",
        scan_backend="auto",
    ):
        super().__init__()

        if d_model < 1 or d_state < 1 or d_conv < 1:
            raise ValueError("d_model, d_state, and d_conv must be positive")
        d_inner = int(expand * d_model)
        if d_inner < 1:
            raise ValueError("expand * d_model must be at least 1")
        if not 0.0 < dt_min <= dt_max:
            raise ValueError("dt_min and dt_max must satisfy 0 < dt_min <= dt_max")
        if scan_backend not in SCAN_BACKENDS:
            raise ValueError(f"Unsupported scan backend: {scan_backend}")

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = resolve_dt_rank(dt_rank, d_model)
        self.scan_backend = scan_backend

        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(d_inner, d_conv)
        self.conv_activation = get_activation(activation, channels=d_inner)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.gate_activation = get_activation(activation)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self._init_dt_proj(dt_min, dt_max, dt_init_floor)

    def _init_dt_proj(self, dt_min, dt_max, dt_init_floor):
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Bias so that softplus(bias) is log-uniform in [dt_min, dt_max].
        log_dt = torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
        dt = torch.exp(log_dt + math.log(dt_min)).clamp_min(dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("MambaMixer expects input with shape [B, L, D]")

        gate = self.gate_activation(self.gate_proj(x))

        x = self.in_proj(x).transpose(1, 2)
        x = self.conv_activation(self.conv(x)).transpose(1, 2)

        dt, B, C = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())

        y = selective_scan(x, delta, A, B, C, self.D.float(), backend=self.scan_backend)
        return self.out_proj(y * gate)


class BidirectionalMamba(nn.Module):
    """Forward and time-reversed Mamba mixers whose outputs are summed.

    Input/output: [B, L, D]
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        activation="silu",
        scan_backend="auto",
    ):
        super().__init__()

        self.forward_mixer = MambaMixer(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            activation=activation,
            scan_backend=scan_backend,
        )
        self.backward_mixer = MambaMixer(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            activation=activation,
            scan_backend=scan_backend,
        )

    def forward(self, x):
        return self.forward_mixer(x) + self.backward_mixer(x.flip(1)).flip(1)
