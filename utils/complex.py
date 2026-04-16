import torch


def to_complex(x):
    """
    Converts [B, C, F, T, 2] -> [B, C, F, T] complex
    """
    if torch.is_complex(x):
        return x

    if x.size(-1) == 2:
        return torch.view_as_complex(x)

    raise ValueError("Invalid shape for to_complex")


def split_complex(x):
    """
    Assumes complex [B, C, F, T], [B, C, F, T, 2] or [B, 2C, F, T] -> returns real, imag as [B, C, F, T]
    """
    if torch.is_complex(x):
        return x.real, x.imag
    elif x.size(-1) == 2:
        return x[..., 0], x[..., 1]
    else:
       return x[:, 0::2], x[:, 1::2]


def ri_to_channels(x):
    """
    (B, C, F, T, 2) -> (B, 2C, F, T)
    """
    B, C, F, T, _ = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(B, C * 2, F, T)


def channels_to_ri(x):
    """
    (B, 2C, F, T) -> (B, C, F, T, 2)
    """
    B, C, F, T = x.shape
    C = C // 2
    x = x.view(B, C, 2, F, T)
    return x.permute(0, 1, 3, 4, 2)
