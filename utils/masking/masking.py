import torch


def complex_mask_apply(x, m):
    """
    x, m: (B, C, F, T, 2)

    Applies complex multiplication:
        (a + ib)(c + id)

    Returns: (B, C, F, T, 2)
    """

    xr, xi = x[..., 0], x[..., 1]
    mr, mi = m[..., 0], m[..., 1]

    yr = mr * xr - mi * xi
    yi = mr * xi + mi * xr

    return torch.stack([yr, yi], dim=-1)
