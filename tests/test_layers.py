import pytest
import torch
import torch.nn as nn

from models.layers import (
    CausalDepthwiseConv1d,
    ConvFeedForwardModule,
    GroupedConvModule1d,
    GroupedLinear,
    get_activation,
    get_norm_4d,
)


@pytest.mark.parametrize(
    "name",
    [
        "leaky_relu",
        "relu",
        "elu",
        "silu",
        "swish",
        "gelu",
        "prelu",
        "identity",
        "none",
        None,
    ],
)
def test_shared_activations_are_available(name):
    assert isinstance(get_activation(name), nn.Module)


def test_prelu_supports_channelwise_parameters():
    activation = get_activation("prelu", channels=6)

    assert activation.num_parameters == 6
    assert activation.weight.shape == (6,)


def test_shared_activation_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("unknown")


def test_channel_normalization_uses_channel_axis_only():
    normalization = get_norm_4d("layer", 4)
    channel_values = torch.tensor([-3.0, -1.0, 1.0, 3.0]).view(1, 4, 1, 1)
    spatial_offsets = torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 1, 3, 5)
    x = (channel_values + spatial_offsets).requires_grad_()

    output = normalization(x)
    output.square().mean().backward()

    torch.testing.assert_close(
        output.mean(dim=1), torch.zeros(2, 3, 5), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        output.var(dim=1, unbiased=False),
        torch.ones(2, 3, 5),
        atol=2e-3,
        rtol=0,
    )
    assert torch.isfinite(x.grad).all()


def test_channel_frequency_normalization_uses_each_frame():
    normalization = get_norm_4d("layer_cf", 4, num_freqs=5)
    x = torch.randn(2, 4, 3, 5, requires_grad=True)

    output = normalization(x)
    output.square().mean().backward()

    torch.testing.assert_close(
        output.mean(dim=(1, 3)), torch.zeros(2, 3), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        output.var(dim=(1, 3), unbiased=False),
        torch.ones(2, 3),
        atol=2e-4,
        rtol=0,
    )
    assert torch.isfinite(x.grad).all()


def test_shared_normalizations_validate_tensor_shape():
    with pytest.raises(ValueError, match="4D tensor"):
        get_norm_4d("layer", 4)(torch.randn(2, 4, 5))
    with pytest.raises(ValueError, match="frequency bins"):
        get_norm_4d("layer_cf", 4, num_freqs=5)(torch.randn(2, 4, 3, 6))


def test_causal_depthwise_conv_preserves_length_and_ignores_future():
    conv = CausalDepthwiseConv1d(3, kernel_size=4)
    x = torch.randn(2, 3, 9)
    perturbed = x.clone()
    perturbed[..., 5:] += 1.0

    output = conv(x)

    assert output.shape == x.shape
    torch.testing.assert_close(conv(perturbed)[..., :5], output[..., :5])
    assert not torch.allclose(conv(perturbed)[..., 5:], output[..., 5:])


def test_causal_depthwise_conv_rejects_nonpositive_kernel():
    with pytest.raises(ValueError, match="kernel_size"):
        CausalDepthwiseConv1d(3, kernel_size=0)


def test_grouped_linear_matches_independent_linear_layers():
    torch.manual_seed(0)
    grouped = GroupedLinear(5, 3, num_groups=4)
    x = torch.randn(2, 7, 4, 5)

    expected = torch.stack(
        [
            nn.functional.linear(
                x[..., group, :], grouped.weight[group], grouped.bias[group]
            )
            for group in range(4)
        ],
        dim=-2,
    )

    torch.testing.assert_close(grouped(x), expected)
    assert grouped(x).shape == (2, 7, 4, 3)


def test_grouped_linear_initializes_like_linear_and_supports_no_bias():
    grouped = GroupedLinear(16, 3, num_groups=2, bias=False)

    assert grouped.bias is None
    assert grouped.weight.abs().max() <= 0.25
    assert grouped(torch.zeros(4, 2, 16)).abs().sum() == 0


@pytest.mark.parametrize(
    "x",
    [torch.randn(5), torch.randn(2, 3, 5), torch.randn(2, 4, 6)],
    ids=["rank", "groups", "features"],
)
def test_grouped_linear_rejects_mismatched_input(x):
    grouped = GroupedLinear(5, 3, num_groups=4)

    with pytest.raises(ValueError):
        grouped(x)


def test_grouped_linear_rejects_nonpositive_sizes():
    with pytest.raises(ValueError):
        GroupedLinear(5, 3, num_groups=0)


@pytest.mark.parametrize("kernel_size", [1, 3], ids=["one", "odd"])
def test_conv_feed_forward_module_preserves_shape(kernel_size):
    module = ConvFeedForwardModule(8, 2, 0.0, "silu", kernel_size=kernel_size, groups=4)
    x = torch.randn(2, 9, 8, requires_grad=True)

    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_conv_feed_forward_module_validates_configuration():
    with pytest.raises(ValueError, match="divisible"):
        ConvFeedForwardModule(8, 2, 0.0, "silu", groups=3)
    with pytest.raises(ValueError, match="kernel_size"):
        ConvFeedForwardModule(8, 2, 0.0, "silu", kernel_size=0)
    with pytest.raises(ValueError, match="\\[B, L, D\\]"):
        ConvFeedForwardModule(8, 2, 0.0, "silu")(torch.randn(9, 8))


@pytest.mark.parametrize("kernel_size", [1, 5], ids=["one", "odd"])
def test_grouped_conv_module_preserves_shape(kernel_size):
    module = GroupedConvModule1d(8, kernel_size, groups=2, activation="prelu")
    x = torch.randn(2, 11, 8, requires_grad=True)

    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert module.activation.num_parameters == 8
    assert torch.isfinite(x.grad).all()


def test_grouped_conv_module_validates_configuration():
    with pytest.raises(ValueError, match="divisible"):
        GroupedConvModule1d(8, 3, groups=3)
    with pytest.raises(ValueError, match="kernel_size"):
        GroupedConvModule1d(8, 0)
    with pytest.raises(ValueError, match="\\[B, L, C\\]"):
        GroupedConvModule1d(8, 3)(torch.randn(2, 8))
