import pytest
import torch

from models.spatialnet import CrossBandBlock, NarrowBandBlock, SpatialNet


def _test_config(**overrides):
    config = {
        "in_channels": 8,
        "out_channels": 8,
        "num_freqs": 17,
        "dim_hidden": 8,
        "dim_squeeze": 2,
        "num_layers": 2,
        "num_heads": 2,
        "ff_expansion_factor": 2,
        "encoder_kernel_size": 5,
        "freq_kernel_size": 5,
        "time_kernel_size": 3,
        "freq_conv_groups": 2,
        "time_conv_groups": 2,
        "share_full_band": True,
        "activation": "silu",
        "conv_activation": "prelu",
        "dropout": 0.0,
        "eps": 1e-5,
        "normalize_input": True,
    }
    config.update(overrides)
    return config


def test_spatialnet_preserves_odd_time_frequency_shape():
    model = SpatialNet(_test_config())
    x = torch.randn(2, 8, 17, 19)

    output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_spatialnet_backward_produces_finite_gradients():
    model = SpatialNet(_test_config())
    x = torch.randn(2, 8, 17, 19, requires_grad=True)

    model(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_normalized_mapping_preserves_input_scale():
    model = SpatialNet(_test_config()).eval()
    x = torch.randn(1, 8, 17, 19)

    with torch.no_grad():
        output = model(x)
        scaled_output = model(3.0 * x)

    torch.testing.assert_close(scaled_output, 3.0 * output)


def test_normalized_mapping_handles_silent_input():
    model = SpatialNet(_test_config()).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 8, 17, 19))

    assert torch.isfinite(output).all()


def test_spatialnet_supports_distinct_output_channel_count():
    model = SpatialNet(_test_config(out_channels=6))

    output = model(torch.randn(1, 8, 17, 19))

    assert output.shape == (1, 6, 17, 19)


def test_spatialnet_shares_full_band_module_across_layers():
    shared = SpatialNet(_test_config(share_full_band=True))
    separate = SpatialNet(_test_config(share_full_band=False))

    shared_modules = [layer.cross_band.full_band for layer in shared.layers]
    separate_modules = [layer.cross_band.full_band for layer in separate.layers]

    assert all(module is shared_modules[0] for module in shared_modules)
    assert len(set(map(id, separate_modules))) == len(separate_modules)

    full_band_parameters = sum(
        parameter.numel() for parameter in shared_modules[0].parameters()
    )
    shared_total = sum(parameter.numel() for parameter in shared.parameters())
    separate_total = sum(parameter.numel() for parameter in separate.parameters())
    assert separate_total - shared_total == full_band_parameters * (
        len(separate_modules) - 1
    )


def test_cross_band_block_processes_frames_independently():
    torch.manual_seed(0)
    block = CrossBandBlock(
        dim_hidden=8, dim_squeeze=2, num_freqs=17, conv_groups=2
    ).eval()
    x = torch.randn(1, 17, 9, 8)
    perturbed = x.clone()
    perturbed[:, :, 4] += torch.randn_like(perturbed[:, :, 4])

    with torch.no_grad():
        output = block(x)
        perturbed_output = block(perturbed)

    unchanged = [frame for frame in range(9) if frame != 4]
    torch.testing.assert_close(
        perturbed_output[:, :, unchanged], output[:, :, unchanged]
    )
    assert not torch.allclose(perturbed_output[:, :, 4], output[:, :, 4])


def test_cross_band_block_mixes_all_frequency_bins():
    torch.manual_seed(0)
    block = CrossBandBlock(
        dim_hidden=8, dim_squeeze=2, num_freqs=17, conv_groups=2
    ).eval()
    x = torch.randn(1, 17, 9, 8)
    perturbed = x.clone()
    perturbed[:, 0] += torch.randn_like(perturbed[:, 0])

    with torch.no_grad():
        output = block(x)
        perturbed_output = block(perturbed)

    # Bin 16 is far outside the kernel-5 reach of bin 0, so only the full-band
    # linear module can propagate the change there.
    assert not torch.allclose(perturbed_output[:, 16], output[:, 16])


def test_narrow_band_block_processes_frequency_bins_independently():
    torch.manual_seed(0)
    block = NarrowBandBlock(dim_hidden=8, num_heads=2, conv_groups=2).eval()
    x = torch.randn(1, 17, 9, 8)
    perturbed = x.clone()
    perturbed[:, 3] += torch.randn_like(perturbed[:, 3])

    with torch.no_grad():
        output = block(x)
        perturbed_output = block(perturbed)

    unchanged = [bin_index for bin_index in range(17) if bin_index != 3]
    torch.testing.assert_close(perturbed_output[:, unchanged], output[:, unchanged])
    assert not torch.allclose(perturbed_output[:, 3], output[:, 3])


def test_narrow_band_block_uses_whole_time_context():
    torch.manual_seed(0)
    block = NarrowBandBlock(dim_hidden=8, num_heads=2, conv_groups=2).eval()
    x = torch.randn(1, 17, 9, 8)
    perturbed = x.clone()
    perturbed[:, :, 8] += torch.randn_like(perturbed[:, :, 8])

    with torch.no_grad():
        output = block(x)
        perturbed_output = block(perturbed)

    assert not torch.allclose(perturbed_output[:, :, 0], output[:, :, 0])


@pytest.mark.parametrize(
    "invalid_input",
    [
        torch.randn(8, 17, 19),
        torch.randn(1, 6, 17, 19),
        torch.randn(1, 8, 19, 19),
    ],
    ids=["rank", "channels", "frequencies"],
)
def test_spatialnet_rejects_invalid_input_shapes(invalid_input):
    model = SpatialNet(_test_config())

    with pytest.raises(ValueError):
        model(invalid_input)


def test_spatialnet_rejects_complex_input():
    model = SpatialNet(_test_config())

    with pytest.raises(ValueError, match="real/imaginary"):
        model(torch.randn(1, 8, 17, 19, dtype=torch.complex64))


@pytest.mark.parametrize(
    "overrides",
    [
        {"in_channels": 7},
        {"out_channels": 7},
        {"num_heads": 3},
        {"freq_conv_groups": 3},
        {"time_conv_groups": 3},
        {"dim_squeeze": 0},
        {"encoder_kernel_size": 0},
        {"freq_kernel_size": 0},
        {"time_kernel_size": 0},
        {"ff_expansion_factor": 0},
        {"share_full_band": "yes"},
        {"dropout": 1.0},
        {"eps": 0.0},
        {"eps": float("inf")},
        {"normalize_input": "true"},
    ],
    ids=[
        "odd-input-channels",
        "odd-output-channels",
        "attention-head-divisibility",
        "frequency-group-divisibility",
        "time-group-divisibility",
        "nonpositive-squeeze",
        "nonpositive-encoder-kernel",
        "nonpositive-frequency-kernel",
        "nonpositive-time-kernel",
        "nonpositive-expansion",
        "nonboolean-sharing",
        "dropout-one",
        "zero-epsilon",
        "infinite-epsilon",
        "nonboolean-normalization",
    ],
)
def test_spatialnet_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        SpatialNet(_test_config(**overrides))
