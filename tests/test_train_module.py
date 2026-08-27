import torch
import torch.nn as nn

from projects.train_module import TrainingModule
from utils.complex import channels_to_ri
from utils.masking import complex_mask_apply

_OMIT_MASKING = object()


def _training_config(masking):
    config = {
        "model": {
            "model_name": "TFGridNet",
            "in_channels": 8,
            "out_channels": 8,
            "num_freqs": 17,
            "emb_dim": 8,
            "num_layers": 1,
            "lstm_hidden_units": 8,
            "attn_n_head": 2,
            "attn_approx_qk_dim": 16,
            "emb_ks": 4,
            "emb_hs": 1,
            "activation": "prelu",
        },
        "training": {"losses": {"stft_losses": {}, "wave_losses": {}}},
        "data": {"stft": {"n_fft": 32, "hop_length": 8, "win_length": 32}},
    }
    if masking is not _OMIT_MASKING:
        config["training"]["masking"] = masking
    return config


class FixedPrediction(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, x):
        return self.prediction.expand(x.size(0), -1, -1, -1)


def _capture_step_estimate(module, x, prediction):
    captured = {}
    module.model = FixedPrediction(prediction)

    def capture_loss(estimate, target, stage):
        captured["estimate"] = estimate
        captured["stage"] = stage
        return estimate.sum() * 0.0

    module.calculate_losses = capture_loss
    target = torch.zeros_like(x)
    module._step((x, target), stage="train")
    return captured


def test_mapping_mode_uses_model_output_as_direct_foa_estimate():
    module = TrainingModule(_training_config(None))
    x = torch.randn(2, 4, 17, 9, 2)
    prediction = torch.randn(1, 8, 17, 9, dtype=torch.bfloat16)

    captured = _capture_step_estimate(module, x, prediction)
    expected = channels_to_ri(prediction.expand(2, -1, -1, -1)).float().contiguous()

    torch.testing.assert_close(captured["estimate"], expected)
    assert captured["estimate"].dtype == torch.float32
    assert captured["estimate"].is_contiguous()
    assert captured["stage"] == "train"


def test_omitted_masking_defaults_to_legacy_complex_masking():
    module = TrainingModule(_training_config(_OMIT_MASKING))
    x = torch.randn(2, 4, 17, 9, 2)
    prediction = torch.randn(1, 8, 17, 9)

    captured = _capture_step_estimate(module, x, prediction)
    mask = channels_to_ri(prediction.expand(2, -1, -1, -1)).contiguous()

    torch.testing.assert_close(captured["estimate"], complex_mask_apply(x, mask))


def test_mapping_mode_runs_real_foa_and_wave_losses_backward():
    config = _training_config(None)
    config["training"]["losses"] = {
        "stft_losses": {
            "foa_active_intensity_doa": {
                "weight": 1.0,
                "doa_method": "cosine_sim",
                "sim_dim": 1,
                "energy_weighting": True,
            },
            "foa_energy_ratio": {
                "weight": 1.0,
                "ratio_type": "directional_to_total",
            },
        },
        "wave_losses": {
            "multi_resolution_stft": {
                "weight": 1.0,
                "fft_sizes": [8, 16, 32],
                "hop_sizes": [2, 4, 8],
                "win_lengths": [8, 16, 32],
            },
            "si_sdr": {"weight": 0.2},
            "foa_covariance": {
                "weight": 1.0,
                "normalize": "fro",
                "loss_type": "l1",
            },
        },
    }
    module = TrainingModule(config)
    module.log = lambda *args, **kwargs: None
    x = torch.randn(1, 4, 17, 9, 2)
    target = torch.randn_like(x)

    loss = module._step((x, target), stage="train")
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in module.model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_training_module_rejects_stft_frequency_mismatch_early():
    config = _training_config(None)
    config["model"]["num_freqs"] = 19

    try:
        TrainingModule(config)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Expected an STFT frequency mismatch error")
