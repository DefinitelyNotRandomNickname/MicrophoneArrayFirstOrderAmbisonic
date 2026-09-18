import pytest
import torch

from models.mamba import (
    BidirectionalMamba,
    Mamba,
    MambaMixer,
    selective_scan,
    selective_scan_reference,
)


def _scan_inputs(batch_size=2, length=11, channels=3, d_state=4, dtype=torch.float64):
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, generator=generator)

    def rand(*shape):
        return torch.rand(*shape, dtype=dtype, generator=generator)

    u = randn(batch_size, length, channels)
    delta = rand(batch_size, length, channels) + 0.05
    A = -(rand(channels, d_state) * 4.0 + 0.5)
    B = randn(batch_size, length, d_state)
    C = randn(batch_size, length, d_state)
    D = randn(channels)
    return u, delta, A, B, C, D


@pytest.mark.parametrize(
    "length,chunk_size",
    [(1, None), (5, None), (11, 3), (12, 4), (7, 16)],
    ids=["single", "auto", "ragged", "aligned", "one-chunk"],
)
def test_chunked_selective_scan_matches_reference(length, chunk_size):
    inputs = _scan_inputs(length=length)

    output = selective_scan(*inputs, backend="torch", chunk_size=chunk_size)

    torch.testing.assert_close(output, selective_scan_reference(*inputs))


def test_chunked_selective_scan_gradients_match_reference():
    inputs = [tensor.requires_grad_() for tensor in _scan_inputs(length=11)]
    reference_inputs = [tensor.detach().requires_grad_() for tensor in inputs]

    selective_scan(*inputs, backend="torch", chunk_size=3).square().sum().backward()
    selective_scan_reference(*reference_inputs).square().sum().backward()

    for tensor, reference in zip(inputs, reference_inputs):
        torch.testing.assert_close(tensor.grad, reference.grad)


def test_chunked_selective_scan_passes_gradcheck():
    inputs = [
        tensor.requires_grad_()
        for tensor in _scan_inputs(batch_size=1, length=6, channels=2, d_state=2)
    ]

    assert torch.autograd.gradcheck(
        lambda *args: selective_scan(*args, backend="torch", chunk_size=3), inputs
    )


def test_selective_scan_computes_low_precision_inputs_in_float32():
    inputs = [tensor.to(torch.bfloat16) for tensor in _scan_inputs()]

    output = selective_scan(*inputs, backend="torch")

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.float(),
        selective_scan_reference(*[tensor.float() for tensor in inputs]),
        atol=0.1,
        rtol=0.05,
    )


def test_selective_scan_rejects_unknown_backend():
    with pytest.raises(ValueError, match="scan backend"):
        selective_scan(*_scan_inputs(), backend="triton")


def _perturbed_outputs(module, split=6):
    torch.manual_seed(0)
    x = torch.randn(1, 10, 8)
    perturbed = x.clone()
    perturbed[:, split:] += 1.0

    with torch.no_grad():
        return module.eval()(x), module(perturbed)


def test_mamba_mixer_is_causal():
    torch.manual_seed(0)
    output, perturbed = _perturbed_outputs(
        MambaMixer(8, d_state=4, d_conv=3, expand=2, scan_backend="torch")
    )

    torch.testing.assert_close(output[:, :6], perturbed[:, :6])
    assert not torch.allclose(output[:, 6:], perturbed[:, 6:])


def test_bidirectional_mamba_uses_future_context():
    torch.manual_seed(0)
    output, perturbed = _perturbed_outputs(
        BidirectionalMamba(8, d_state=4, d_conv=3, expand=2, scan_backend="torch")
    )

    assert not torch.allclose(output[:, :6], perturbed[:, :6])


def test_mamba_mixer_uses_auto_dt_rank():
    mixer = MambaMixer(40, d_state=4, dt_rank="auto", scan_backend="torch")

    assert mixer.dt_rank == 3
    assert mixer.x_proj.out_features == 3 + 2 * 4


def _test_config(**overrides):
    config = {
        "in_channels": 8,
        "out_channels": 8,
        "num_freqs": 17,
        "emb_dim": 8,
        "num_layers": 1,
        "d_state": 4,
        "d_conv": 3,
        "expand": 2,
        "dt_rank": "auto",
        "activation": "silu",
        "dropout": 0.0,
        "eps": 1e-5,
        "normalize_input": True,
        "scan_backend": "torch",
    }
    config.update(overrides)
    return config


def test_mamba_preserves_odd_time_frequency_shape():
    model = Mamba(_test_config())
    x = torch.randn(2, 8, 17, 19)

    output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_mamba_backward_produces_finite_gradients():
    model = Mamba(_test_config())
    x = torch.randn(2, 8, 17, 19, requires_grad=True)

    model(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_normalized_mapping_preserves_input_scale():
    model = Mamba(_test_config()).eval()
    x = torch.randn(1, 8, 17, 19)

    with torch.no_grad():
        output = model(x)
        scaled_output = model(3.0 * x)

    torch.testing.assert_close(scaled_output, 3.0 * output)


def test_normalized_mapping_handles_silent_input():
    model = Mamba(_test_config()).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 8, 17, 19))

    assert torch.isfinite(output).all()


def test_mamba_supports_distinct_output_channel_count():
    model = Mamba(_test_config(out_channels=6))

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
def test_mamba_rejects_invalid_input_shapes(invalid_input):
    model = Mamba(_test_config())

    with pytest.raises(ValueError):
        model(invalid_input)


@pytest.mark.parametrize(
    "overrides",
    [
        {"in_channels": 7},
        {"out_channels": 7},
        {"d_state": 0},
        {"d_conv": 0},
        {"expand": 0},
        {"dt_rank": 0},
        {"dt_rank": "full"},
        {"dropout": 1.0},
        {"eps": 0.0},
        {"eps": float("inf")},
        {"normalize_input": "true"},
        {"scan_backend": "triton"},
    ],
    ids=[
        "odd-input-channels",
        "odd-output-channels",
        "nonpositive-state",
        "nonpositive-conv",
        "nonpositive-expand",
        "nonpositive-dt-rank",
        "unknown-dt-rank",
        "dropout-one",
        "zero-epsilon",
        "infinite-epsilon",
        "nonboolean-normalization",
        "unknown-scan-backend",
    ],
)
def test_mamba_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        Mamba(_test_config(**overrides))
