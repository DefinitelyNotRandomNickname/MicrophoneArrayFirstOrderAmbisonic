import torch
import numpy as np

from utils.complex import to_complex


# -------------------------
# Torch
# -------------------------


def istft_from_spectrogram(x, n_fft, hop_length, win_length, window=None):
    """
    Handles both:
    - real/imag stacked: [B, 2C, F, T]
    - complex: [B, C, F, T]

    Returns: [B, C, T]
    """

    x = to_complex(x)  # [B, C, F, T]

    B, C, F, T = x.shape

    x = x.view(B * C, F, T)

    if isinstance(window, type(None)):
        window = torch.hann_window(win_length, device=x.device)

    wave = torch.istft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )

    wave = wave.view(B, C, -1)

    return wave


def compute_stft(x, n_fft, hop_length, win_length, window=None):
    x = torch.from_numpy(x).float()

    if isinstance(window, type(None)):
        window = torch.hann_window(win_length).to(device=x.device, dtype=x.dtype)

    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    return torch.view_as_real(stft)


# -------------------------
# Numpy
# -------------------------


def fft_convolve(signal, rir):
    T = signal.shape[0]
    L = rir.shape[1]

    n = T + L - 1

    S = np.fft.rfft(signal, n=n)
    R = np.fft.rfft(rir, n=n, axis=1)

    out = np.fft.irfft(S[None, :] * R, n=n, axis=1)
    return out[:, :T]
