import pytest
import torch

from utils.complex import channels_to_ri, ri_to_channels, split_complex, to_complex


def test_real_imag_and_channel_representations_round_trip():
    ri = torch.tensor(
        [[[[[1.0, -1.0], [2.0, -2.0]]], [[[3.0, -3.0], [4.0, -4.0]]]]]
    )

    channels = ri_to_channels(ri)
    restored = channels_to_ri(channels)
    complex_tensor = to_complex(ri)
    real, imaginary = split_complex(complex_tensor)

    assert channels.shape == (1, 4, 1, 2)
    torch.testing.assert_close(restored, ri)
    torch.testing.assert_close(real, ri[..., 0])
    torch.testing.assert_close(imaginary, ri[..., 1])


def test_to_complex_rejects_tensor_without_real_imag_axis():
    with pytest.raises(ValueError, match="Invalid shape"):
        to_complex(torch.zeros(1, 2, 3))
