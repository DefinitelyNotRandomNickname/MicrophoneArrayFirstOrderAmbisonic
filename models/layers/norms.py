import torch.nn as nn


def _channel_layer_norm(num_channels):
    return nn.GroupNorm(1, num_channels)


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
    "identity": lambda _: nn.Identity(),
    "none": lambda _: nn.Identity(),
    None: lambda _: nn.Identity(),
}


def get_norm_1d(name, num_channels):
    try:
        return NORMALIZATIONS_1D[name](num_channels)
    except KeyError as exc:
        raise ValueError(f"Unsupported 1D normalization: {name}") from exc


def get_norm_2d(name, num_channels):
    try:
        return NORMALIZATIONS_2D[name](num_channels)
    except KeyError as exc:
        raise ValueError(f"Unsupported 2D normalization: {name}") from exc


def get_norm(name, num_channels):
    return get_norm_2d(name, num_channels)
