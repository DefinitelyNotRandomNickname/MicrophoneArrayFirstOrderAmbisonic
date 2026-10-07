import math

import pytest
import torch

from utils.features import (
    FEATURES,
    FeaturePipeline,
    PairwisePhatIpd,
    microphone_pairs,
    pairwise_phat_ipd,
)


def test_pairwise_phat_is_cosine_and_sine_of_ipd_for_every_pair():
    phases = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2])
    spec = torch.polar(torch.ones(4, 1, 1), phases[:, None, None])

    feature = pairwise_phat_ipd(spec)
    expected_phases = torch.tensor(
        [-math.pi / 2, -math.pi, math.pi / 2, -math.pi / 2, math.pi, -math.pi / 2]
    )
    expected = torch.polar(torch.ones(6), expected_phases)[:, None, None]

    assert microphone_pairs(4) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    torch.testing.assert_close(feature, expected, atol=1e-6, rtol=1e-6)


def test_pairwise_phat_preserves_ri_layout_and_handles_silent_bins():
    spec = torch.zeros(4, 3, 2, dtype=torch.complex64)
    spec[:, 1] = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j])[
        :, None
    ]
    ri = torch.view_as_real(spec)

    feature = pairwise_phat_ipd(ri)

    assert feature.shape == (6, 3, 2, 2)
    assert feature.dtype == ri.dtype
    assert torch.equal(feature[:, 0], torch.zeros_like(feature[:, 0]))
    torch.testing.assert_close(
        torch.view_as_complex(feature)[:, 1].abs(),
        torch.ones(6, 2),
    )


def test_pairwise_feature_can_select_pairs_and_reports_model_width():
    feature = PairwisePhatIpd(
        {"pairs": [[0, 2], [1, 3]]},
        n_fft=32,
        sr=16000,
        num_mics=4,
    )

    assert feature.pairs == ((0, 2), (1, 3))
    assert feature.feature_channels == 2
    assert feature.input_channels == 4


def test_feature_pipeline_composes_input_hooks_in_config_order():
    cfg = {
        "data": {
            "features": {
                "pairwise_phat_ipd": {
                    "pairs": [[0, 1], [2, 3]],
                }
            }
        }
    }
    pipeline = FeaturePipeline.from_config(
        cfg,
        FEATURES,
        n_fft=32,
        sr=16000,
        num_mics=4,
    )
    mic_spec = torch.randn(4, 17, 5, 2)
    values = pipeline.extract(mic_spec)
    model_input = pipeline.model_input(
        torch.randn(1, 8, 17, 5),
        {name: value.unsqueeze(0) for name, value in values.items()},
    )

    assert pipeline.names == ("pairwise_phat_ipd",)
    assert pipeline.input_channels == 4
    assert model_input.shape == (1, 12, 17, 5)


@pytest.mark.parametrize("option", ["enabled", "as_input"])
def test_defining_a_feature_always_activates_it_as_model_input(option):
    cfg = {"data": {"features": {"pairwise_phat_ipd": {option: False}}}}

    with pytest.raises(ValueError, match="always active model inputs"):
        FeaturePipeline.from_config(
            cfg,
            FEATURES,
            n_fft=32,
            sr=16000,
            num_mics=4,
        )


@pytest.mark.parametrize(
    "pairs",
    [[[0, 0]], [[0, 4]], [[0, 1], [1, 0]], []],
)
def test_pairwise_feature_rejects_invalid_pairs(pairs):
    with pytest.raises(ValueError):
        PairwisePhatIpd(
            {"pairs": pairs},
            n_fft=32,
            sr=16000,
            num_mics=4,
        )
