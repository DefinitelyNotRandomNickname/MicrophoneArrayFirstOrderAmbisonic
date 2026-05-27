import torch

from utils.masking import complex_mask_apply


def test_complex_mask_apply_multiplies_real_imaginary_pairs():
    spectrogram = torch.tensor([[[[[2.0, 3.0], [-1.0, 4.0]]]]])
    mask = torch.tensor([[[[[4.0, -1.0], [0.0, 2.0]]]]])

    masked = complex_mask_apply(spectrogram, mask)

    expected = torch.tensor([[[[[11.0, 10.0], [-8.0, -2.0]]]]])
    torch.testing.assert_close(masked, expected)
