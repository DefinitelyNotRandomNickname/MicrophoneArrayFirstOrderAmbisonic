import torch
import torch.nn.functional as F

from utils.audio import compute_stft
from utils.complex import split_complex, to_complex
from utils.losses.utils import reduce_loss, weighted_reduce_loss


# -----------------------------
# L1
# -----------------------------

def l1_loss(estimate, target, **kwargs):
    return F.l1_loss(estimate, target)


# -----------------------------
# Magnitude
# -----------------------------

def _magnitude(x, eps=1e-8):
    xr, xi = split_complex(x)
    return torch.sqrt(xr**2 + xi**2 + eps)


def magnitude_loss(estimate, target, eps=1e-8, **kwargs):
    mag_e = _magnitude(estimate, eps)
    mag_t = _magnitude(target, eps)
    return l1_loss(mag_e, mag_t)


# -----------------------------
# DOA from FOA active-intensity loss
# -----------------------------

def _foa_energy(x):
    return x.abs().square().sum(dim=1)


def _energy_weights(target, power=1.0, eps=1e-8):
    weights = _foa_energy(target).clamp_min(eps)
    if power != 1.0:
        weights = weights.pow(power)
    return weights


def foa_active_intensity_doa_loss(
    estimate,
    target,
    eps=1e-8,
    energy_weighting=True,
    energy_weight_power=1.0,
    reduction="mean",
    **kwargs,
):
    """
    Direction-of-arrival loss from FOA active-intensity vectors.
    """
    estimate = to_complex(estimate)
    target = to_complex(target)

    estimate_w = estimate[:, 0:1]
    target_w = target[:, 0:1]
    estimate_xyz = estimate[:, 1:4]
    target_xyz = target[:, 1:4]

    estimate_intensity = torch.real(torch.conj(estimate_w) * estimate_xyz)
    target_intensity = torch.real(torch.conj(target_w) * target_xyz)

    estimate_dir = F.normalize(estimate_intensity, dim=1, eps=eps)
    target_dir = F.normalize(target_intensity, dim=1, eps=eps)

    cosine = (estimate_dir * target_dir).sum(dim=1).clamp(-1.0, 1.0)
    loss = 1.0 - cosine

    weights = None
    if energy_weighting:
        weights = _energy_weights(target, power=energy_weight_power, eps=eps)

    return weighted_reduce_loss(loss, weights=weights, reduction=reduction, eps=eps)


# -----------------------------
# FOA energy-ratio loss
# -----------------------------

def foa_energy_ratio_loss(
    estimate,
    target,
    eps=1e-8,
    ratio_type="directional_to_total",
    energy_weighting=True,
    energy_weight_power=1.0,
    reduction="mean",
    **kwargs,
):
    """
    FOA directional energy-ratio loss on WXYZ spectrograms.

    ratio_type:
    - directional_to_total: sum(|XYZ|^2) / sum(|WXYZ|^2)
    - directional_to_omni: sum(|XYZ|^2) / |W|^2
    """
    estimate = to_complex(estimate)
    target = to_complex(target)

    estimate_w_energy = estimate[:, 0].abs().square()
    target_w_energy = target[:, 0].abs().square()
    estimate_xyz_energy = estimate[:, 1:4].abs().square().sum(dim=1)
    target_xyz_energy = target[:, 1:4].abs().square().sum(dim=1)

    if ratio_type == "directional_to_total":
        estimate_ratio = estimate_xyz_energy / (
            estimate_w_energy + estimate_xyz_energy + eps
        )
        target_ratio = target_xyz_energy / (target_w_energy + target_xyz_energy + eps)
    elif ratio_type == "directional_to_omni":
        estimate_ratio = estimate_xyz_energy / (estimate_w_energy + eps)
        target_ratio = target_xyz_energy / (target_w_energy + eps)
    else:
        raise ValueError(f"Unsupported FOA energy ratio type: {ratio_type}")

    loss = torch.abs(estimate_ratio - target_ratio)

    weights = None
    if energy_weighting:
        weights = _energy_weights(target, power=energy_weight_power, eps=eps)

    return weighted_reduce_loss(loss, weights=weights, reduction=reduction, eps=eps)


# -----------------------------
# Spectral convergence loss
# -----------------------------

def spectral_convergence_loss(estimate_mag, target_mag, eps=1e-8, reduction="mean"):
    diff = torch.linalg.vector_norm(target_mag - estimate_mag, ord=2, dim=(-2, -1))
    denom = torch.linalg.vector_norm(target_mag, ord=2, dim=(-2, -1)).clamp_min(eps)
    loss = diff / denom

    return reduce_loss(loss, reduction)


# -----------------------------
# Multi resolution STFT loss
# -----------------------------

def _normalize_resolutions(fft_sizes=None, hop_sizes=None, win_lengths=None):
    if fft_sizes is None:
        fft_sizes = [512, 1024, 2048]
    if hop_sizes is None:
        hop_sizes = [128, 256, 512]
    if win_lengths is None:
        win_lengths = fft_sizes

    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
        raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length")

    return [
        (int(n_fft), int(hop_length), int(win_length))
        for n_fft, hop_length, win_length in zip(fft_sizes, hop_sizes, win_lengths)
    ]


def multi_resolution_stft_loss(
    estimate,
    target,
    fft_sizes=[512, 1024, 2048],
    hop_sizes=[128, 256, 512],
    win_lengths=[512, 1024, 2048],
    spectral_convergence_weight=1.0,
    magnitude_weight=1.0,
    log_magnitude_weight=1.0,
    eps=1e-8,
    reduction="mean",
    **kwargs,
):
    """
    Multi-resolution STFT loss for waveforms.

    Returns a weighted average of spectral convergence, linear magnitude L1,
    and log-magnitude L1 across all configured STFT resolutions.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("multi_resolution_stft_loss supports reduction: 'mean' or 'sum'")

    stft_resolutions = _normalize_resolutions(
        fft_sizes=fft_sizes,
        hop_sizes=hop_sizes,
        win_lengths=win_lengths,
    )

    total_loss = estimate.new_tensor(0.0)

    for n_fft, hop_length, win_length in stft_resolutions:
        estimate_stft = compute_stft(
            estimate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            interleave=False,
        )
        target_stft = compute_stft(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            interleave=False,
        )
        estimate_mag = torch.sqrt(estimate_stft.real**2 + estimate_stft.imag**2 + eps)
        target_mag = torch.sqrt(target_stft.real**2 + target_stft.imag**2 + eps)

        resolution_loss = estimate.new_tensor(0.0)

        if spectral_convergence_weight:
            resolution_loss += spectral_convergence_weight * spectral_convergence_loss(
                estimate_mag,
                target_mag,
                eps=eps,
                reduction=reduction,
            )

        if magnitude_weight:
            resolution_loss += magnitude_weight * F.l1_loss(
                estimate_mag,
                target_mag,
                reduction=reduction,
            )

        if log_magnitude_weight:
            resolution_loss += log_magnitude_weight * F.l1_loss(
                torch.log(estimate_mag.clamp_min(eps)),
                torch.log(target_mag.clamp_min(eps)),
                reduction=reduction,
            )

        total_loss += resolution_loss

    return total_loss / len(stft_resolutions)
