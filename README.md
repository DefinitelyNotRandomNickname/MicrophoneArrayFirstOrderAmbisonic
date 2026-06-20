# Microphone Array First-Order Ambisonic

Project for estimating first-order ambisonic (FOA) signals from an
irregular omnidirectional microphone-array input using deep learning.

The current pipeline builds synthetic spatial samples from
speech/noise WAV files and room impulse responses (RIRs), converts them to
network features (STFT), and trains a PyTorch Lightning models to predict a complex mask
that maps microphone-array spectrograms toward FOA WXYZ spectrograms.

## What is included

- Config-driven training with separate data, model, and training YAML files.
- Models for complex spectrogram processing.
- HDF5-backed spatial audio dataset generation.
- Complex tensor helpers for real/imaginary channel conversions.
- FOA-aware losses, including active-intensity DOA, energy ratio, covariance,
  SI-SDR, and multi-resolution STFT losses.
- Pytest coverage for config merging, complex helpers, masking, audio
  convolution, and loss utilities.

## Repository layout

```text
configs/
  data/               Data paths, sample rate, STFT, source/noise settings
  models/             Model architecture configs
  training/           Optimizer, scheduler, loss, and trainer settings
data/                 PyTorch dataset code
models/               Model definitions and layers
notebooks/            Data generation notebooks
projects/             Training entry point and Lightning module
scripts/              Convenience shell scripts
tests/                Unit tests
utils/                Audio, complex-number, masking, loss, and config helpers
```

## Installation

Create and activate a virtual environment, then install the package in editable
mode:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

The full dependency list is declared in `pyproject.toml`.

## Configuration

Training merges three YAML files:

- `configs/data/32khz.yaml`
- `configs/models/UNet.yaml`
- `configs/training/base.yaml`

The model config defines network model and layout. The base
training config sets Lightning trainer options, optimizer settings, a
scheduler, masking mode, and the active loss terms. Lastly data config specifies
paths to the training HDF datasets and wav pre-convoluted samples. Additionally, 
microphone matrices come in three different shapes:
- tetrahedral (default A-Format matrix),
- square (lowest conversion quality, due to aliasing in 3D),
- Quasi-Line (for example smartphone microphone array). 

## Training

Run the script from the repository root:

```bash
bash scripts/train.sh
```

Or call the training entry point directly:

```bash
python projects/train.py \
  --data_cfg configs/data/32khz.yaml \
  --model_cfg configs/models/UNet.yaml \
  --train_cfg configs/training/base.yaml \
  --exp_name UNet_base_32khz \
  --log_dir .logs
```

Training automatically uses GPU acceleration when CUDA is available. Logs and
checkpoints are written under `.logs/<exp_name>/`.

To inspect TensorBoard logs:

```bash
tensorboard --logdir .logs
```

## Testing

Install the development dependencies and run:

```bash
pytest
```

## Future development

This project is in early stage. With time, more DNN networks, augmentations, and input features will be implemented.

Model roadmap:
- [x] UNet,
- [ ] Conformer,
- [ ] TF-GridNet,
- [ ] Mamba,
- [ ] DPRNN/DPTNet.

Input roadmap:
- [x] STFT,
- [ ] wave,
- [ ] HTT,
- [ ] IPD/ILD.

## Notes

- The default config references local example paths such as `tmp/*.h5` and
  `/data/soundsource/...`; these are placeholders for the expected dataset
  locations.
- Validation uses the same dataset logic as training, with a shorter synthetic
  epoch length and different rooms.
- The package metadata declares the project license as MIT.
