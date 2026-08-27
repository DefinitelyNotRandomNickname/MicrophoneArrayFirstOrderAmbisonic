import math

import torch
import torch.nn as nn


def _channel_layer_norm(num_channels):
    return nn.GroupNorm(1, num_channels)


def _group_norm(num_channels: int, max_groups: int = 8):
    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return _channel_layer_norm(num_channels)


class LayerNormalization4D(nn.Module):
    """Layer normalization over channels for a [B, C, T, F] tensor."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("LayerNormalization4D expects a 4D tensor")

        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class LayerNormalization4DCF(nn.Module):
    """Layer normalization over channels and frequency for every frame."""

    def __init__(self, channels, num_freqs, eps=1e-5):
        super().__init__()
        if channels < 1 or num_freqs < 1:
            raise ValueError("channels and num_freqs must be positive")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")

        self.num_freqs = num_freqs
        self.weight = nn.Parameter(torch.ones(1, channels, 1, num_freqs))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, num_freqs))
        self.eps = eps

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("LayerNormalization4DCF expects a 4D tensor")
        if x.size(-1) != self.num_freqs:
            raise ValueError(
                f"Expected {self.num_freqs} frequency bins, got {x.size(-1)}"
            )

        mean = x.mean(dim=(1, 3), keepdim=True)
        variance = x.var(dim=(1, 3), unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


NORMALIZATIONS_1D = {
    "batch": nn.BatchNorm1d,
    "instance": nn.InstanceNorm1d,
    "layer": _channel_layer_norm,
    "identity": lambda _: nn.Identity(),
    "none": lambda _: nn.Identity(),
    None: lambda _: nn.Identity(),
}

NORMALIZATIONS_2D = {
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
    "layer": _channel_layer_norm,
    "group": _group_norm,
    "identity": lambda _: nn.Identity(),
    "none": lambda _: nn.Identity(),
    None: lambda _: nn.Identity(),
}

NORMALIZATIONS_4D = {
    "layer": LayerNormalization4D,
    "layer_cf": LayerNormalization4DCF,
    None: lambda _: nn.Identity(),
}


def get_norm_1d(name, *args, **kwargs):
    try:
        return NORMALIZATIONS_1D[name](*args, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unsupported 1D normalization: {name}") from exc


def get_norm_2d(name, *args, **kwargs):
    try:
        return NORMALIZATIONS_2D[name](*args, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unsupported 2D normalization: {name}") from exc


def get_norm_4d(name, *args, **kwargs):
    try:
        return NORMALIZATIONS_4D[name](*args, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unsupported 4D normalization: {name}") from exc


def get_norm(name, *args, **kwargs):
    return get_norm_2d(name, *args, **kwargs)
