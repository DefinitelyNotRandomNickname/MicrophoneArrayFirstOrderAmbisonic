from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Callable

import torch

from utils.complex import ri_to_channels


def as_model_channels(value):
    """Convert a batched complex/RI feature to real-valued model channels."""
    if torch.is_complex(value):
        value = torch.view_as_real(value)

    if value.dim() == 5 and value.size(-1) == 2:
        return ri_to_channels(value)

    if value.dim() == 4:
        return value

    raise ValueError(
        "A model-input feature must have shape [B, C, F, T], "
        "[B, C, F, T, 2], or be a complex [B, C, F, T] tensor; "
        f"got {tuple(value.shape)}"
    )


def direct_prediction(reference, prediction):
    """Use the network prediction as a direct complex spectral mapping."""
    return prediction.contiguous().to(dtype=reference.dtype)


def residual_prediction(reference, prediction):
    """Add the network prediction to a feature-provided reference signal."""
    return (reference + prediction).contiguous().to(dtype=reference.dtype)


@dataclass(frozen=True)
class PredictionContext:
    """Composition selected by prediction hooks before producing an estimate."""

    reference: torch.Tensor
    combine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    masking_fn: Callable | None = None
    owner: str | None = None

    def replace(self, *, owner, reference, combine):
        if self.owner is not None:
            raise ValueError(
                f"Features '{self.owner}' and '{owner}' both configure how the "
                "network prediction is combined with a reference"
            )
        return replace(self, owner=owner, reference=reference, combine=combine)

    def estimate(self, prediction):
        return self.combine(self.reference, prediction)


class SpectralFeature:
    """Base hooks shared by config-driven spectrogram input features."""

    name = None

    def __init__(self, cfg, *, n_fft, sr, num_mics):
        del n_fft, sr
        self.cfg = cfg or {}
        redundant = {"enabled", "as_input"}.intersection(self.cfg)
        if redundant:
            raise ValueError(
                "Configured features are always active model inputs; remove "
                f"redundant options {sorted(redundant)} and omit the feature "
                "entry to run without it"
            )
        self.num_mics = int(num_mics)

    @property
    def feature_channels(self):
        """Number of logical feature channels before RI channel flattening."""
        raise NotImplementedError

    @property
    def input_channels(self):
        """Number of real-valued channels appended to the network input."""
        return 2 * self.feature_channels

    def dataset_hook(self, mic_spec, **context):
        """Dataset hook: derive this feature from one microphone STFT."""
        raise NotImplementedError

    def input_hook(self, model_input, value):
        """Training hook: append the feature to the real-valued model input."""
        return torch.cat([model_input, as_model_channels(value)], dim=1)

    def prediction_hook(self, context, value):
        """Training hook: optionally change prediction/reference composition."""
        del value
        return context

    def validate_model(self, masking_fn):
        """Validate training-only requirements after masking is configured."""
        del masking_fn


class FeaturePipeline:
    """Runs registered feature hooks without feature-specific branching."""

    def __init__(self, features):
        self.features = tuple(features)
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"Feature names must be unique, got {names}")

    @classmethod
    def from_config(
        cls,
        cfg,
        registry,
        *,
        n_fft,
        sr,
        num_mics,
    ):
        configured = cfg.get("data", {}).get("features") or {}
        features = []

        for name, feature_cfg in configured.items():
            feature_cfg = feature_cfg or {}
            if name not in registry:
                raise ValueError(
                    f"Unknown data feature '{name}'; available features are "
                    f"{sorted(registry)}"
                )

            feature = registry[name](
                feature_cfg,
                n_fft=n_fft,
                sr=sr,
                num_mics=num_mics,
            )
            # Registry aliases still expose one stable key in the batch.
            feature.name = name
            features.append(feature)

        return cls(features)

    @property
    def names(self):
        return tuple(feature.name for feature in self.features)

    @property
    def input_channels(self):
        return sum(feature.input_channels for feature in self.features)

    def describe_input_channels(self):
        return [
            f"{feature.name}={feature.input_channels}"
            for feature in self.features
            if feature.input_channels
        ]

    def extract(self, mic_spec, **context):
        values = {}
        for feature in self.features:
            value = feature.dataset_hook(mic_spec, **context)
            if not torch.isfinite(value).all():
                raise ValueError(f"Feature '{feature.name}' produced non-finite data")
            values[feature.name] = value
        return values

    def normalize_batch_values(self, values):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("Feature batch values must be a name -> tensor mapping")

        missing = [name for name in self.names if name not in values]
        if missing:
            raise ValueError(f"Batch is missing configured features: {missing}")
        return values

    def move_batch(self, values, *, device, dtype):
        values = self.normalize_batch_values(values)
        return {
            name: value.to(device=device, dtype=dtype) for name, value in values.items()
        }

    def model_input(self, model_input, values):
        values = self.normalize_batch_values(values)
        for feature in self.features:
            model_input = feature.input_hook(model_input, values[feature.name])
        return model_input

    def prediction_context(self, context, values):
        values = self.normalize_batch_values(values)
        for feature in self.features:
            context = feature.prediction_hook(context, values[feature.name])
        return context

    def validate_model(self, masking_fn):
        for feature in self.features:
            feature.validate_model(masking_fn)
