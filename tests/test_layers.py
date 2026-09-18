import pytest
import torch
import torch.nn as nn

from models.layers import CausalDepthwiseConv1d, get_activation, get_norm_4d


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
    x = torch.randn(2, 4, 3, 5, requires_grad=True)

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
