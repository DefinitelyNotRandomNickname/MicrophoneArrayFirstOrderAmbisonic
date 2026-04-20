import torch
from torch.utils.data import Dataset

import numpy as np
import random
import h5py
import soundfile as sf
import os
from glob import glob

from utils.audio import fft_convolve, compute_stft


class SpatialAudioDataset(Dataset):
    def __init__(self, cfg, rir_path):

        dcfg = cfg["data"]

        self.rir_file = h5py.File(rir_path, "r")
        self.rir_mems = self.rir_file["rir/mems"]
        self.rir_foa = self.rir_file["rir/foa"]

        self.audio_paths = self._collect_wavs(dcfg["audio_paths"])
        self.noise_paths = self._collect_wavs(dcfg.get("noise_paths", []))

        self.sr = dcfg["sr"]
        self.segment_samples = int(dcfg["seconds"] * self.sr)

        self.n_fft = dcfg["stft"]["n_fft"]
        self.hop = dcfg["stft"]["hop_length"]
        self.win = dcfg["stft"]["win_length"]

        self.normalize = dcfg.get("normalize", True)
        self.num_sources = dcfg.get("num_sources", 1)
        self.snr_db = dcfg.get("snr_db", None)

        self.window = torch.hann_window(self.win)
        self.max_len = int(dcfg.get("max_len", 1e5))

    def __len__(self):
        return self.max_len
    
    def _collect_wavs(self, paths):
        if not paths:
            return []

        all_files = []

        for p in paths:
            if os.path.isdir(p):
                all_files.extend(glob(os.path.join(p, "**", "*.wav"), recursive=True))

        if len(all_files) == 0:
            raise RuntimeError(f"No wav files found in: {paths}")

        return all_files

    def _load_audio(self, path):
        audio, sr = sf.read(path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # TODO: up/down sample
        assert sr == self.sr

        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        else:
            start = random.randint(0, len(audio) - self.segment_samples)
            audio = audio[start:start + self.segment_samples]

        return audio.astype(np.float32)

    def _get_sources(self):
        sources = []
        for _ in range(self.num_sources):
            path = random.choice(self.audio_paths)
            audio = self._load_audio(path)
            sources.append(audio)
        return sources
    
    def _generate_noise(self, signal):
        if not self.noise_paths or self.snr_db is None:
            return np.zeros_like(signal)

        noise_path = random.choice(self.noise_paths)
        noise = self._load_audio(noise_path)

        sig_power = np.mean(signal**2)
        noise_power = np.mean(noise**2) + 1e-12

        target_noise_power = sig_power / (10 ** (self.snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)

        return noise

    def __getitem__(self, idx):

        # Get audio sources
        sources = self._get_sources()

        # Sample RIRs
        rir_idx = random.randint(0, len(self.rir_mems) - 1)
        mems_rirs = self.rir_mems[rir_idx]  # (num_sources, 4, rir_length)
        foa_rirs = self.rir_foa[rir_idx]    # (num_sources, 4, rir_length)

        # Convolve each source with its corresponding RIRs and mix
        mems = None
        foa = None
        
        for src_idx, source in enumerate(sources):
            # Get RIRs for this source
            mems_rir = mems_rirs[src_idx]  # (4, rir_length)
            foa_rir = foa_rirs[src_idx]    # (4, rir_length)
            
            # Convolve source with its RIRs
            src_mems = fft_convolve(source, mems_rir)
            src_foa = fft_convolve(source, foa_rir)
            
            # Add to mix
            if mems is None:
                mems = src_mems
                foa = src_foa
            else:
                mems += src_mems
                foa += src_foa

        # Mixing normalization
        mems /= self.num_sources
        foa /= self.num_sources

        # Add noise with given SNR
        noise = self._generate_noise(mems)
        mems += noise
        foa += noise

        # Peak-Norm
        if self.normalize:
            max_val = max(np.abs(mems).max(), np.abs(foa).max(), 1e-6)
            mems = mems / max_val
            foa = foa / max_val

        # Compute STFT
        mems_spec = compute_stft(mems, self.n_fft, self.hop, self.win, self.window)
        foa_spec = compute_stft(foa, self.n_fft, self.hop, self.win, self.window)

        return mems_spec, foa_spec
