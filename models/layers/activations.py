import torch.nn as nn


ACTIVATIONS = {
    "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True),
    "relu": lambda: nn.ReLU(inplace=True),
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "none": nn.Identity,
    None: nn.Identity,
}


def get_activation(name):
    try:
        return ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc
