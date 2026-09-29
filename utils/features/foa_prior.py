"""
Geometry / ATF-conditioned FOA prior.

Turns the known microphone-array geometry into a physics-based first-order
Ambisonic estimate that a network can refine instead of having to learn the
whole array-to-FOA mapping from scratch.

Pipeline, per frequency bin f:

1. Free-field steering matrix over a direction grid.

       S_f[m, d] = exp(+j * 2 * pi * f * (omega_d . r_m) / c)

   with r_m the m-th microphone position relative to the array centre and
   omega_d a unit vector pointing from the array towards the source. The sign
   is positive because a plane wave from omega_d reaches a microphone at r_m
   *earlier* than the centre by (omega_d . r_m) / c.

2. Target Ambisonic encoding of the same grid, in the WXYZ convention used by
   the RIR simulator (omni W, three figure-of-eight dipoles aligned with the
   room axes):

       Y[:, d] = [1, omega_dx, omega_dy, omega_dz]^T

3. Regularized least-squares encoder E_f (4 x M) minimising

       || E_f S_f - Y ||_F^2 + D * reg_eps * || E_f ||_F^2

   whose closed-form solution, after dividing both Gram terms by the number of
   grid directions D so that reg_eps is relative to a unit diagonal, is

       E_f = B_f (G_f + reg_eps * I)^-1,
       G_f = S_f S_f^H / D,   B_f = Y S_f^H / D.

   G_f is the diffuse-field coherence matrix of the array, so this is the usual
   diffuse-field-regularized A-format to B-format conversion.

4. Optional white-noise-gain limit. ||E_f[c, :]||_2 is exactly the gain applied
   to spatially white microphone noise on Ambisonic channel c, and it explodes
   for the dipoles at low frequencies (it grows like 1 / (k * r)). Rows above
   the limit are scaled down, which trades dipole accuracy for not handing the
   network an amplified-noise prior.

The prior itself is then a per-bin matrix product with the microphone STFT:

       P[:, f, t] = E_f X[:, f, t].

For a single far-field plane wave this returns the true WXYZ signals, so the
prior is directly comparable to the FOA target rather than being just a
feature: the network can be asked to refine it, mask it, or add a residual.
"""

import math
from collections import OrderedDict

import numpy as np
import torch

from utils.complex import to_complex

SOUND_SPEED = 343.0
LAYOUT_KEY = "mic_array_layout"


def fibonacci_sphere(num_directions, dtype=torch.float64):
    """
    Near-uniform unit vectors on the sphere, used as the LS design grid.
    """
    if num_directions < 1:
        raise ValueError(f"num_directions must be positive, got {num_directions}")

    index = torch.arange(num_directions, dtype=dtype) + 0.5

    z = 1.0 - 2.0 * index / num_directions
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    azimuth = math.pi * (math.sqrt(5.0) - 1.0) * index

    return torch.stack(
        [radius * torch.cos(azimuth), radius * torch.sin(azimuth), z],
        dim=-1,
    )


def steering_matrix(mic_positions, freqs, directions, sound_speed=SOUND_SPEED):
    """
    Free-field plane-wave steering matrix relative to the array centre.
    """
    # (D, M): projection of each microphone offset onto each look direction.
    projection = directions @ mic_positions.transpose(-1, -2)

    phase = 2.0 * math.pi * freqs.view(-1, 1, 1) * projection.unsqueeze(0)
    phase = phase / sound_speed

    return torch.polar(torch.ones_like(phase), phase).transpose(-1, -2)


def ambisonic_encoding_matrix(directions):
    """
    WXYZ encoding gains of a unit-amplitude plane wave from each direction.
    """
    omni = torch.ones_like(directions[:, :1])
    return torch.cat([omni, directions], dim=-1).transpose(-1, -2)


def ls_ambisonic_encoder(
    mic_positions,
    freqs,
    num_directions=256,
    sound_speed=SOUND_SPEED,
    reg_eps=1e-3,
    max_wng_db=20.0,
    directions=None,
    dtype=torch.complex64,
):
    """
    Regularized least-squares microphone-to-WXYZ encoder.
    """
    mic_positions = torch.as_tensor(mic_positions, dtype=torch.float64)
    freqs = torch.as_tensor(freqs, dtype=torch.float64)

    if mic_positions.dim() != 2 or mic_positions.size(-1) != 3:
        raise ValueError(
            f"mic_positions must have shape (M, 3), got {tuple(mic_positions.shape)}"
        )

    if reg_eps <= 0.0:
        raise ValueError(f"reg_eps must be positive, got {reg_eps}")

    if directions is None:
        directions = fibonacci_sphere(num_directions, dtype=torch.float64)
    else:
        directions = torch.as_tensor(directions, dtype=torch.float64)

    num_grid = directions.size(0)
    num_mics = mic_positions.size(0)

    steering = steering_matrix(mic_positions, freqs, directions, sound_speed)
    encoding = ambisonic_encoding_matrix(directions).to(steering.dtype)

    steering_h = steering.conj().transpose(-1, -2)

    # Diffuse-field coherence of the array, normalized to a unit diagonal so
    # that reg_eps reads directly as diagonal loading in dB.
    gram = steering @ steering_h / num_grid
    cross = encoding @ steering_h / num_grid

    loaded = gram + reg_eps * torch.eye(num_mics, dtype=gram.dtype, device=gram.device)

    # encoder = cross @ loaded^-1, solved via the Hermitian system
    # loaded @ x = cross^H so that encoder = x^H.
    solution = torch.linalg.solve(loaded, cross.conj().transpose(-1, -2))
    encoder = solution.conj().transpose(-1, -2)

    if max_wng_db is not None:
        encoder = limit_white_noise_gain(encoder, max_wng_db)

    return encoder.to(dtype)


def limit_white_noise_gain(encoder, max_wng_db, eps=1e-12):
    """
    Scales down encoder rows whose spatially-white noise gain exceeds the limit.
    """
    max_gain = 10.0 ** (max_wng_db / 20.0)

    gain = encoder.abs().square().sum(dim=-1).sqrt()
    scale = (max_gain / gain.clamp_min(eps)).clamp(max=1.0)

    return encoder * scale.unsqueeze(-1).to(encoder.dtype)


def encode_prior(mic_spec, encoder):
    """
    Applies a per-frequency encoder to a microphone spectrogram.
    """
    interleaved = not torch.is_complex(mic_spec)

    spec = to_complex(mic_spec)

    if spec.size(-2) != encoder.size(0):
        raise ValueError(
            f"Encoder has {encoder.size(0)} frequency bins but the spectrogram "
            f"has {spec.size(-2)}"
        )

    if spec.size(-3) != encoder.size(-1):
        raise ValueError(
            f"Encoder expects {encoder.size(-1)} microphones but the "
            f"spectrogram has {spec.size(-3)}"
        )

    encoder = encoder.to(device=spec.device, dtype=spec.dtype)
    prior = torch.einsum("fam,...mft->...aft", encoder, spec)

    if interleaved:
        return torch.view_as_real(prior)
    return prior


def relative_mic_positions(meta_entry, num_mics):
    """
    Perturbed microphone offsets from one HDF5 meta entry.

    The RIR generator stores `actual_mic_positions` as a (3, N) array of absolute
    room coordinates: the perturbed capsules first, then the co-located FOA
    capsules.
    """
    positions = np.asarray(meta_entry["actual_mic_positions"], dtype=np.float64)

    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError(
            f"Expected actual_mic_positions with shape (3, N), got {positions.shape}"
        )

    if positions.shape[1] < num_mics:
        raise ValueError(
            f"Metadata holds {positions.shape[1]} capsules but {num_mics} "
            "microphones were requested"
        )

    center = np.asarray(meta_entry["mic_center"], dtype=np.float64).reshape(3, 1)

    return (positions[:, :num_mics] - center).T


def nominal_mic_positions(meta, num_mics):
    """
    The array layout the corpus was generated with, recovered from the data.
    """
    if not meta:
        raise ValueError(
            "The nominal array layout is derived from the RIR metadata, but "
            "this dataset has none"
        )

    layout = meta[0].get(LAYOUT_KEY)
    if layout is not None:
        positions = np.asarray(layout, dtype=np.float64)

        if positions.shape != (num_mics, 3):
            raise ValueError(
                f"{LAYOUT_KEY} must have shape ({num_mics}, 3), "
                f"got {positions.shape}"
            )

        return positions

    offsets = [relative_mic_positions(entry, num_mics) for entry in meta]

    return np.mean(offsets, axis=0)


class AtfFoaPrior:
    """
    Config-driven wrapper that builds and caches LS Ambisonic encoders.

    Config keys (under `data.features.atf_foa_prior`):
        geometry_source: "actual" (per-sample perturbed positions from the RIR
            metadata) or "nominal" (the layout averaged out of that metadata,
            i.e. the realistic case where calibration knows the array it was
            designed with but not each unit's perturbation)
        num_directions:  size of the least-squares design grid
        sound_speed:     metres per second
        reg_eps:         diagonal loading relative to a unit Gram diagonal
        max_wng_db:      white-noise-gain limit per Ambisonic channel, or null
        cache_size:      number of encoders kept in memory
    """

    def __init__(self, cfg, n_fft, sr):
        cfg = cfg or {}

        self.geometry_source = cfg.get("geometry_source", "actual")
        if self.geometry_source not in ("actual", "nominal"):
            raise ValueError(
                f"Unknown geometry_source '{self.geometry_source}', "
                "expected 'actual' or 'nominal'"
            )

        self.num_directions = int(cfg.get("num_directions", 256))
        self.sound_speed = float(cfg.get("sound_speed", SOUND_SPEED))
        self.reg_eps = float(cfg.get("reg_eps", 1e-3))

        max_wng_db = cfg.get("max_wng_db", 20.0)
        self.max_wng_db = None if max_wng_db is None else float(max_wng_db)

        self.cache_size = int(cfg.get("cache_size", 256))

        self.freqs = torch.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
        self.directions = fibonacci_sphere(self.num_directions)

        self._nominal_positions = None
        self._cache = OrderedDict()

    @property
    def num_channels(self):
        return 4

    def nominal_positions(self, meta, num_mics):
        """
        The dataset's undisturbed array layout, derived once and reused.
        """
        if self._nominal_positions is None:
            self._nominal_positions = nominal_mic_positions(meta, num_mics)

        return self._nominal_positions

    def positions_for(self, meta, rir_idx, num_mics):
        """
        Microphone offsets to condition on, plus a cache key.

        The key is None when the geometry varies per sample, leaving the caller
        to key the cache on whatever identifies that sample.
        """
        if self.geometry_source == "nominal":
            return self.nominal_positions(meta, num_mics), "nominal"

        if not meta:
            raise ValueError(
                "geometry_source='actual' requires RIR metadata with "
                "'actual_mic_positions'; this dataset has none"
            )

        return relative_mic_positions(meta[rir_idx], num_mics), None

    def encoder(self, mic_positions, key=None):
        """
        Builds (or reuses) the (F, 4, M) encoder for one array geometry.
        """
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        encoder = ls_ambisonic_encoder(
            mic_positions,
            self.freqs,
            sound_speed=self.sound_speed,
            reg_eps=self.reg_eps,
            max_wng_db=self.max_wng_db,
            directions=self.directions,
        )

        if key is not None and self.cache_size > 0:
            self._cache[key] = encoder
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return encoder

    def __call__(self, mic_spec, mic_positions, key=None):
        """
        mic_spec: (..., M, F, T) complex or (..., M, F, T, 2) real

        Returns: the WXYZ prior in the same layout
        """
        return encode_prior(mic_spec, self.encoder(mic_positions, key=key))
