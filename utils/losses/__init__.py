from utils.losses.spectrogram import (
    foa_active_intensity_doa_loss,
    foa_energy_ratio_loss,
    l1_loss,
    magnitude_loss,
    multi_resolution_stft_loss,
)
from utils.losses.wav import foa_covariance_loss, si_sdr_loss


SPECTROGRAM_LOSSES = {
    "magnitude": magnitude_loss,
    "l1": l1_loss,
    "foa_active_intensity_doa": foa_active_intensity_doa_loss,
    "foa_energy_ratio": foa_energy_ratio_loss,
}

WAVE_LOSSES = {
    "si_sdr": si_sdr_loss,
    "multi_resolution_stft": multi_resolution_stft_loss,
    "foa_covariance": foa_covariance_loss,
}
