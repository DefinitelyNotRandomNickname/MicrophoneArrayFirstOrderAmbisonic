import torch.nn as nn

ACTIVATIONS = {
    "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True),
    "relu": lambda: nn.ReLU(inplace=True),
    "elu": lambda: nn.ELU(inplace=True),
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
    "identity": nn.Identity,
    "none": nn.Identity,
    None: nn.Identity,
}


def get_activation(name, *, channels=None):
    try:
        activation = ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc

    if name == "prelu":
        if channels is not None and channels < 1:
            raise ValueError("PReLU channels must be positive")
        return activation(num_parameters=channels or 1)

    return activation()
