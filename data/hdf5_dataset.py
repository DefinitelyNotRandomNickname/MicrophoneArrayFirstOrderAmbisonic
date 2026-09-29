import torch
from torch.utils.data import Dataset

import numpy as np
import json
import random
import h5py
import soundfile as sf
import os
from glob import glob

from utils.audio import fft_convolve, compute_stft
from utils.features import FEATURES


class SpatialAudioDataset(Dataset):
    def __init__(self, cfg, rir_path, stage):

        dcfg = cfg["data"]

        self.rir_file = h5py.File(rir_path, "r")
        self.rir_mems = self.rir_file["rir/mems"]
        self.rir_foa = self.rir_file["rir/foa"]
        self.meta = self._load_meta()
        print(self.meta)

        self.audio_paths = self._collect_wavs(dcfg["audio_paths"])
        self.noise_paths = self._collect_wavs(dcfg.get("noise_paths", []))

        self.sr = dcfg["sr"]
        self.segment_samples = int(dcfg["seconds"] * self.sr)

        self.n_fft = dcfg["stft"]["n_fft"]
        self.hop = dcfg["stft"]["hop_length"]
        self.win = dcfg["stft"]["win_length"]

        self.normalize = dcfg.get("normalize", True)
        self.num_sources = dcfg.get("num_sources", 1)

        # Environmental noise
        self.snr_db = dcfg.get("snr_db", [-10, 10])
        self.noise_in_target = dcfg.get("noise_in_target", True)

        # MEMS electrical self-noise
        self.self_noise_db = dcfg.get("self_noise_db", None)

        self.rir_slots = int(self.rir_mems.shape[1])
        self.num_mics = int(self.rir_mems.shape[2])
        self.max_speech_sources = min(self.num_sources, self.rir_slots)

        # Noise source
        if self._environmental_noise_enabled():
            self.max_speech_sources = max(
                1, min(self.max_speech_sources, self.rir_slots - 1)
            )

        self.window = torch.hann_window(self.win)
        self.max_len = int(dcfg.get("max_len", 1e5))

        if stage == "valid":
            self.max_len //= 50

        self.foa_prior = self._build_feature(dcfg, "atf_foa_prior")

    def __len__(self):
        return self.max_len

    def _load_meta(self):
        """Per-RIR simulation metadata (room, positions)."""
        raw = self.rir_file["meta/json"][()]

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        return json.loads(raw)

    def _build_feature(self, dcfg, name):
        """
        Instantiates an optional input feature from `data.features.<name>`.
        """
        cfg = (dcfg.get("features") or {}).get(name, None)

        if cfg is None or not cfg.get("enabled", True):
            return None

        return FEATURES[name](cfg, n_fft=self.n_fft, sr=self.sr)

    def _prior_geometry(self, rir_idx):
        """
        Microphone offsets to condition the FOA prior on, plus a cache key.

        The key is the RIR index whenever the geometry varies per sample, so
        that repeated draws of the same room reuse the same encoder.

        Static geometry is may be more or less useful during actual inference
        depending on the array.
        """
        positions, key = self.foa_prior.positions_for(
            self.meta, rir_idx, self.num_mics
        )

        return positions, rir_idx if key is None else key

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
        for _ in range(random.randint(1, self.max_speech_sources)):
            path = random.choice(self.audio_paths)
            audio = self._load_audio(path)
            sources.append(audio)
        return sources

    def _environmental_noise_enabled(self):
        return bool(self.noise_paths) and self.snr_db is not None

    def _noise_slot(self, num_speech):
        """
        RIR source slot for the noise, preferring one the speech is not using.
        """
        if num_speech < self.rir_slots:
            return random.randrange(num_speech, self.rir_slots)

        return random.randrange(self.rir_slots)

    def _environmental_noise(self, mems, mems_rirs, foa_rirs, num_speech):
        """
        Convolved noise for both source and target, e.g., a wind.
        """
        if not self._environmental_noise_enabled():
            return 0.0, 0.0

        slot = self._noise_slot(num_speech)
        noise = self._load_audio(random.choice(self.noise_paths))

        mems_noise = fft_convolve(noise, mems_rirs[slot])
        foa_noise = fft_convolve(noise, foa_rirs[slot])

        db = random.uniform(self.snr_db[0], self.snr_db[1])

        target_power = np.mean(mems**2) / (10 ** (db / 10))
        scale = np.sqrt(target_power / (np.mean(mems_noise**2) + 1e-12))

        return mems_noise * scale, foa_noise * scale

    def _self_noise(self, mems):
        """Electrical noise in absolute SPL scale."""
        if self.self_noise_db is None:
            return 0.0

        db = random.uniform(self.self_noise_db[0], self.self_noise_db[1])
        power = np.mean(mems**2) / (10 ** (db / 10))

        return np.random.randn(*mems.shape) * np.sqrt(power)

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
        mems /= len(sources)
        foa /= len(sources)

        mems_noise, foa_noise = self._environmental_noise(
            mems, mems_rirs, foa_rirs, len(sources)
        )

        mems = mems + mems_noise
        if self.noise_in_target:
            foa = foa + foa_noise

        mems = mems + self._self_noise(mems)

        # Peak-Norm
        if self.normalize:
            max_val = max(np.abs(mems).max(), np.abs(foa).max(), 1e-6)
            mems = mems / max_val
            foa = foa / max_val

        # Compute STFT
        mems_spec = compute_stft(mems, self.n_fft, self.hop, self.win, self.window)
        foa_spec = compute_stft(foa, self.n_fft, self.hop, self.win, self.window)

        if self.foa_prior is None:
            return mems_spec, foa_spec

        positions, key = self._prior_geometry(rir_idx)
        prior_spec = self.foa_prior(mems_spec, positions, key=key)

        return mems_spec, foa_spec, prior_spec
