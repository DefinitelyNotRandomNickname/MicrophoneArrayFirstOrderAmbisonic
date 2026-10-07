"""Pairwise PHAT features, represented as cosine/sine inter-channel phase."""

from itertools import combinations

import torch

from utils.complex import to_complex
from utils.features.base import SpectralFeature


def microphone_pairs(num_mics):
    """All unique microphone pairs in deterministic lexicographic order."""
    if int(num_mics) < 2:
        raise ValueError(
            f"Pairwise PHAT needs at least two microphones, got {num_mics}"
        )
    return tuple(combinations(range(int(num_mics)), 2))


def validate_pairs(pairs, num_mics):
    validated = []
    seen = set()

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                f"Each microphone pair must contain two indices, got {pair}"
            )

        left, right = (int(pair[0]), int(pair[1]))
        if left == right:
            raise ValueError(f"A microphone cannot be paired with itself: {pair}")
        if min(left, right) < 0 or max(left, right) >= num_mics:
            raise ValueError(
                f"Microphone pair {pair} is outside the valid range "
                f"[0, {num_mics - 1}]"
            )

        identity = frozenset((left, right))
        if identity in seen:
            raise ValueError(f"Duplicate microphone pair: {pair}")
        seen.add(identity)
        validated.append((left, right))

    if not validated:
        raise ValueError("At least one microphone pair is required")
    return tuple(validated)


def pairwise_phat_ipd(mic_spec, pairs=None, eps=1e-8):
    """Compute unit pairwise cross spectra whose RI parts are cos/sin(IPD).

    Args:
        mic_spec: ``(..., M, F, T)`` complex or ``(..., M, F, T, 2)`` RI.
        pairs: Iterable of ordered microphone-index pairs. Defaults to all pairs.
        eps: Lower bound for cross-spectrum magnitudes.

    Returns:
        One complex PHAT channel per pair, preserving the input representation.
    """
    if not torch.is_complex(mic_spec) and (
        mic_spec.dim() < 4 or mic_spec.size(-1) != 2
    ):
        raise ValueError(
            "mic_spec must be complex [..., M, F, T] or RI [..., M, F, T, 2]"
        )
    if not torch.isfinite(torch.as_tensor(eps)) or float(eps) <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps}")

    interleaved = not torch.is_complex(mic_spec)
    spec = to_complex(mic_spec)
    num_mics = spec.size(-3)
    pairs = (
        microphone_pairs(num_mics) if pairs is None else validate_pairs(pairs, num_mics)
    )

    left = torch.tensor([pair[0] for pair in pairs], device=spec.device)
    right = torch.tensor([pair[1] for pair in pairs], device=spec.device)
    cross_spectrum = spec.index_select(-3, left) * spec.index_select(-3, right).conj()
    phat = cross_spectrum / cross_spectrum.abs().clamp_min(float(eps))

    return torch.view_as_real(phat) if interleaved else phat


class PairwisePhatIpd(SpectralFeature):
    """Dataset/input hook for normalized cross-spectrum spatial cues."""

    name = "pairwise_phat_ipd"

    def __init__(
        self,
        cfg,
        n_fft,
        sr,
        num_mics=4,
    ):
        super().__init__(
            cfg,
            n_fft=n_fft,
            sr=sr,
            num_mics=num_mics,
        )
        configured_pairs = self.cfg.get("pairs", "all")
        self.pairs = (
            microphone_pairs(self.num_mics)
            if configured_pairs == "all"
            else validate_pairs(configured_pairs, self.num_mics)
        )
        self.eps = float(self.cfg.get("eps", 1e-8))
        if not torch.isfinite(torch.tensor(self.eps)) or self.eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {self.eps}")

    @property
    def feature_channels(self):
        return len(self.pairs)

    def dataset_hook(self, mic_spec, **context):
        del context
        return pairwise_phat_ipd(mic_spec, self.pairs, self.eps)
