import pytest
import torch

from models.tfgridnet import TFGridNet


def _test_config(**overrides):
    config = {
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
        "eps": 1e-5,
        "dropout": 0.0,
        "normalize_input": True,
    }
    config.update(overrides)
    return config


def test_tfgridnet_preserves_odd_time_frequency_shape():
    model = TFGridNet(_test_config())
    x = torch.randn(2, 8, 17, 19)

    output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_tfgridnet_backward_produces_finite_gradients():
    model = TFGridNet(_test_config())
    x = torch.randn(2, 8, 17, 19, requires_grad=True)

    model(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_normalized_mapping_preserves_input_scale():
    model = TFGridNet(_test_config()).eval()
    x = torch.randn(1, 8, 17, 19)

    with torch.no_grad():
        output = model(x)
        scaled_output = model(3.0 * x)

    torch.testing.assert_close(scaled_output, 3.0 * output)


def test_normalized_mapping_handles_silent_input():
    model = TFGridNet(_test_config()).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 8, 17, 19))

    assert torch.isfinite(output).all()


def test_tfgridnet_supports_distinct_output_channel_count():
    model = TFGridNet(_test_config(out_channels=6))

    output = model(torch.randn(1, 8, 17, 19))

    assert output.shape == (1, 6, 17, 19)


@pytest.mark.parametrize(
    "invalid_input",
    [
        torch.randn(8, 17, 19),
        torch.randn(1, 6, 17, 19),
        torch.randn(1, 8, 19, 19),
    ],
    ids=["rank", "channels", "frequencies"],
)
def test_tfgridnet_rejects_invalid_input_shapes(invalid_input):
    model = TFGridNet(_test_config())

    with pytest.raises(ValueError):
        model(invalid_input)


@pytest.mark.parametrize(
    "overrides",
    [
        {"attn_n_head": 3},
        {"emb_ks": 0},
        {"emb_hs": 0},
        {"emb_ks": 2, "emb_hs": 3},
        {"out_channels": 7},
        {"eps": 0.0},
        {"eps": float("inf")},
        {"normalize_input": "true"},
    ],
    ids=[
        "attention-head-divisibility",
        "nonpositive-kernel",
        "nonpositive-hop",
        "hop-larger-than-kernel",
        "odd-output-channels",
        "zero-epsilon",
        "infinite-epsilon",
        "nonboolean-normalization",
    ],
)
def test_tfgridnet_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        TFGridNet(_test_config(**overrides))
