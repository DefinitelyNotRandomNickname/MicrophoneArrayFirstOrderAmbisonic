from utils.features.base import (
    FeaturePipeline,
    PredictionContext,
    SpectralFeature,
    as_model_channels,
    direct_prediction,
)
from utils.features.foa_prior import (
    LAYOUT_KEY,
    AtfFoaPrior,
    ambisonic_encoding_matrix,
    encode_prior,
    fibonacci_sphere,
    limit_white_noise_gain,
    ls_ambisonic_encoder,
    nominal_mic_positions,
    relative_mic_positions,
    steering_matrix,
)
from utils.features.pairwise_phat import (
    PairwisePhatIpd,
    microphone_pairs,
    pairwise_phat_ipd,
)

FEATURES = {
    "atf_foa_prior": AtfFoaPrior,
    "pairwise_phat_ipd": PairwisePhatIpd,
    "pairwise_phat": PairwisePhatIpd,
}


__all__ = [
    "FEATURES",
    "FeaturePipeline",
    "LAYOUT_KEY",
    "AtfFoaPrior",
    "PairwisePhatIpd",
    "PredictionContext",
    "SpectralFeature",
    "ambisonic_encoding_matrix",
    "as_model_channels",
    "direct_prediction",
    "encode_prior",
    "fibonacci_sphere",
    "limit_white_noise_gain",
    "ls_ambisonic_encoder",
    "microphone_pairs",
    "nominal_mic_positions",
    "pairwise_phat_ipd",
    "relative_mic_positions",
    "steering_matrix",
]
