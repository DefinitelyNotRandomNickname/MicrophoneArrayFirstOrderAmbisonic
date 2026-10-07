import torch
import pytorch_lightning as pl

from models import MODELS
from utils.audio import istft_from_spectrogram
from utils.complex import channels_to_ri
from utils.features import (
    FEATURES,
    FeaturePipeline,
    PredictionContext,
    as_model_channels,
    direct_prediction,
)
from utils.losses import SPECTROGRAM_LOSSES, WAVE_LOSSES
from utils.masking import MASKS
from projects.schedulers import build_lr_scheduler


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.mcfg = cfg["model"]
        self.tcfg = cfg["training"]
        self.dcfg = cfg["data"]

        self.num_mics = int(self.dcfg.get("num_mics", 4))
        self.feature_hooks = FeaturePipeline.from_config(
            cfg,
            FEATURES,
            n_fft=self.dcfg["stft"]["n_fft"],
            sr=self.dcfg.get("sr", 1),
            num_mics=self.num_mics,
        )

        configured_freqs = self.mcfg.get("num_freqs")
        n_fft = self.dcfg.get("stft", {}).get("n_fft")
        if configured_freqs is not None and n_fft is not None:
            expected_freqs = int(n_fft) // 2 + 1
            if int(configured_freqs) != expected_freqs:
                raise ValueError(
                    f"model.num_freqs={configured_freqs} does not match "
                    f"data.stft.n_fft={n_fft} ({expected_freqs} bins)"
                )

        self.model = MODELS[self.mcfg["model_name"]](self.mcfg).to(
            self.device, dtype=self.dtype
        )

        masking = self.tcfg.get("masking", "complex")
        self.masking_fn = None if masking is None else MASKS[masking]

        self.feature_hooks.validate_model(self.masking_fn)

        self._check_input_channels()

        # In place compilation
        if self.tcfg.get("compile", False):
            self.model.compile(mode=self.tcfg.get("compile_mode", "default"))

        self.lr = self.tcfg.get("lr", 1e-3)
        self.weight_decay = self.tcfg.get("weight_decay", 0.0)

        self.stft_losses = self.tcfg["losses"].get("stft_losses", {})
        self.wave_losses = self.tcfg["losses"].get("wave_losses", {})

        self.save_hyperparameters(cfg)

    def _check_input_channels(self):
        """
        Catches a model config whose width does not match the feature hooks.

        Reads the width from the built model, since models configure it
        differently.
        """
        in_channels = getattr(self.model, "in_channels", None)

        if in_channels is None:
            return

        raw_channels = 2 * self.num_mics
        expected = raw_channels + self.feature_hooks.input_channels

        if int(in_channels) != expected:
            components = [f"microphone_stft={raw_channels}"]
            components.extend(self.feature_hooks.describe_input_channels())
            raise ValueError(
                f"model.in_channels={in_channels} does not match the configured "
                f"input features ({expected} real channels: "
                + ", ".join(components)
                + ")"
            )

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch[0], batch[1]
        feature_values = batch[2] if len(batch) > 2 else {}

        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)
        feature_values = self.feature_hooks.move_batch(
            feature_values,
            device=self.device,
            dtype=self.dtype,
        )

        model_in = self.feature_hooks.model_input(
            as_model_channels(x),
            feature_values,
        )

        pred = self(model_in)
        pred = channels_to_ri(pred)

        reference = torch.view_as_real(x) if torch.is_complex(x) else x
        combine = direct_prediction if self.masking_fn is None else self.masking_fn
        prediction = PredictionContext(
            reference=reference,
            combine=combine,
            masking_fn=self.masking_fn,
        )
        prediction = self.feature_hooks.prediction_context(
            prediction,
            feature_values,
        )
        y_hat = prediction.estimate(pred)

        loss = self.calculate_losses(y_hat, y, stage)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def calculate_losses(self, estimate, target, stage):
        with torch.autocast(device_type=estimate.device.type, enabled=False):
            estimate = estimate.to(
                dtype=torch.complex64 if estimate.is_complex() else torch.float32
            )
            target = target.to(
                dtype=torch.complex64 if target.is_complex() else torch.float32
            )
            return self._calculate_losses(estimate, target, stage)

    def _calculate_losses(self, estimate, target, stage):
        total_loss = estimate.real.new_tensor(0.0)

        for loss, loss_params in self.stft_losses.items():
            loss_fn = SPECTROGRAM_LOSSES[loss]
            loss_val = loss_fn(estimate, target, **loss_params)
            total_loss += loss_val * loss_params["weight"]
            self.log(
                f"{stage}_{loss}_loss",
                loss_val,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=target.size(0),
            )

        estimate = istft_from_spectrogram(estimate, **self.dcfg["stft"])
        target = istft_from_spectrogram(target, **self.dcfg["stft"])

        for loss, loss_params in self.wave_losses.items():
            loss_fn = WAVE_LOSSES[loss]
            loss_val = loss_fn(estimate, target, **loss_params)
            total_loss += loss_val * loss_params["weight"]
            self.log(
                f"{stage}_{loss}_loss",
                loss_val,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=target.size(0),
            )

        self.log(
            f"{stage}_total_loss",
            total_loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=target.size(0),
        )

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        sched_cfg = self.tcfg.get("scheduler", None)

        if sched_cfg is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": build_lr_scheduler(optimizer, sched_cfg, self),
        }
