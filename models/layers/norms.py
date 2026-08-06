import torch.nn as nn


def _channel_layer_norm(num_channels):
    return nn.GroupNorm(1, num_channels)


def _group_norm(num_channels: int, max_groups: int = 8):
    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return _channel_layer_norm(num_channels)


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


def get_norm(name, *args, **kwargs):
    return get_norm_2d(name, *args, **kwargs)
