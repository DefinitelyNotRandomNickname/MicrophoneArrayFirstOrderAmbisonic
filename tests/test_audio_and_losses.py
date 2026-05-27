import numpy as np
import pytest
import torch

from utils.audio import fft_convolve
from utils.losses.utils import weighted_reduce_loss


def test_fft_convolve_applies_each_room_impulse_response_and_truncates_signal():
    signal = np.array([1.0, 2.0, 0.0])
    impulse_responses = np.array([[1.0, 0.5], [0.0, 1.0]])

    result = fft_convolve(signal, impulse_responses)

    np.testing.assert_allclose(result, np.array([[1.0, 2.5, 1.0], [0.0, 1.0, 2.0]]))


def test_weighted_reduce_loss_computes_normalized_weighted_mean():
    loss = torch.tensor([1.0, 3.0])
    weights = torch.tensor([1.0, 3.0])

    result = weighted_reduce_loss(loss, weights=weights)

    assert result.item() == pytest.approx(2.5)


def test_weighted_reduce_loss_handles_all_zero_weights():
    result = weighted_reduce_loss(torch.tensor([1.0, 3.0]), weights=torch.zeros(2))

    assert result.item() == pytest.approx(0.0)
