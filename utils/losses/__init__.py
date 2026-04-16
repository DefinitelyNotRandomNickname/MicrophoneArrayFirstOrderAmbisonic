from utils.losses.spectrogram import magnitude_loss, l1_loss
from utils.losses.wav import si_sdr_loss

SPECTROGRAM_LOSSES = {
    "magnitude": magnitude_loss,
    "l1": l1_loss,
}

WAVE_LOSSES = {
    "si_sdr": si_sdr_loss,
}
