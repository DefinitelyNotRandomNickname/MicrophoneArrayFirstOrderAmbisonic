import torch


# -----------------------------
# SI-SDR
# -----------------------------

def si_sdr_loss(estimate, target, eps=1e-9, reduction="mean", **kwargs):
    """
    estimate, target: [B, C, T] or [B, T]
    returns negative SI-SDR (for minimization)
    """

    if estimate.dim() == 2:
        estimate = estimate.unsqueeze(1)
        target = target.unsqueeze(1)

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

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss