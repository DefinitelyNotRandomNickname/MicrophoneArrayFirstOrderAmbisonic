import torch
import torch.nn.functional as F

from utils.complex import split_complex


# -----------------------------
# L1
# -----------------------------

def l1_loss(estimate, target, **kwargs):
    return F.l1_loss(estimate, target)


# -----------------------------
# Magnitude
# -----------------------------


def _magnitude(x):
    xr, xi = split_complex(x)
    return torch.sqrt(xr**2 + xi**2 + 1e-8)


def magnitude_loss(estimate, target, **kwargs):
    mag_e = _magnitude(estimate)
    mag_t = _magnitude(target)
    return l1_loss(mag_e, mag_t)
