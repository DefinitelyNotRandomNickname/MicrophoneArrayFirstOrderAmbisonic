import torch
import torch.nn.functional as F

from utils.losses.utils import reduce_loss


# -----------------------------
# SI-SDR
# -----------------------------

def si_sdr_loss(estimate, target, eps=1e-9, reduction="mean", **kwargs):
    """
    estimate, target: [B, C, T]
    returns negative channel-wise SI-SDR
    """

    # zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # projection
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + eps

    s_target = dot * target / target_energy
    e_noise = estimate - s_target

    ratio = (torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
    si_sdr_val = 10 * torch.log10(ratio + eps)

    loss = -si_sdr_val

    return reduce_loss(loss, reduction)


# -----------------------------
# FOA covariance
# -----------------------------

def _foa_covariance(x, eps=1e-8, center=True, normalize="fro"):
    if center:
        x = x - x.mean(dim=-1, keepdim=True)

    denom = max(x.size(-1) - 1, 1)
    covariance = torch.matmul(x, x.transpose(-1, -2)) / denom

    if normalize == "fro":
        norm = torch.linalg.matrix_norm(covariance, ord="fro", dim=(-2, -1))
        covariance = covariance / norm[:, None, None].clamp_min(eps)
    elif normalize == "trace":
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        covariance = covariance / trace[:, None, None].abs().clamp_min(eps)
    elif normalize is None or normalize == "none":
        pass
    else:
        raise ValueError(f"Unsupported covariance normalization: {normalize}")

    return covariance


def foa_covariance_loss(
    estimate,
    target,
    eps=1e-8,
    center=True,
    normalize="fro",
    loss_type="l1",
    reduction="mean",
    **kwargs,
):
    """
    Compares per-example WXYZ covariance matrices from waveform FOA signals.
    """

    estimate_covariance = _foa_covariance(
        estimate,
        eps=eps,
        center=center,
        normalize=normalize,
    )
    target_covariance = _foa_covariance(
        target,
        eps=eps,
        center=center,
        normalize=normalize,
    )

    if loss_type == "l1":
        loss = torch.abs(estimate_covariance - target_covariance)
    elif loss_type == "mse":
        loss = F.mse_loss(estimate_covariance, target_covariance, reduction="none")
    else:
        raise ValueError(f"Unsupported covariance loss type: {loss_type}")

    return reduce_loss(loss, reduction)
