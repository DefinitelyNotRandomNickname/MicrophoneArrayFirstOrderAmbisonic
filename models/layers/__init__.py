import torch.nn as nn

ACTIVATIONS = {
    "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
    "relu": nn.ReLU(inplace=True),
}

NORMALIZATIONS = {
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
    None: nn.Identity,

}


def get_activation(name):
    return ACTIVATIONS[name]


def get_norm(name, num_channels):
    return NORMALIZATIONS[name](num_channels)
