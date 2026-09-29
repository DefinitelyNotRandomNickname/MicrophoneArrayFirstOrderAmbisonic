import torch
import pytorch_lightning as pl

from models import MODELS
from utils.audio import istft_from_spectrogram
from utils.complex import ri_to_channels, channels_to_ri
from utils.losses import SPECTROGRAM_LOSSES, WAVE_LOSSES
from utils.masking import MASKS
from projects.schedulers import build_lr_scheduler


# How the network output is combined with the geometry/ATF FOA prior:
#   none        the prior is only an extra input feature
#   residual    the network predicts a correction added to the prior
#   mask        the masking function is applied to the prior instead of the
#               microphone signals
PRIOR_COMBINE_MODES = ("none", "residual", "mask")


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.mcfg = cfg["model"]
        self.tcfg = cfg["training"]
        self.dcfg = cfg["data"]

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

        self.prior_as_input = self.tcfg.get("prior_as_input", True)
        self.prior_combine = self.tcfg.get("prior_combine", "none")

        if self.prior_combine not in PRIOR_COMBINE_MODES:
            raise ValueError(
                f"training.prior_combine={self.prior_combine} is not one of "
                f"{sorted(PRIOR_COMBINE_MODES)}"
            )

        if self.prior_combine == "mask" and self.masking_fn is None:
            raise ValueError(
                "training.prior_combine='mask' needs training.masking to be set"
            )

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
        Catches a model config that was not widened for the extra prior
        channels (or was widened without the feature being enabled).

        Reads the width from the built model, since models configure it
        differently.
        """
        in_channels = getattr(self.model, "in_channels", None)

        if in_channels is None:
            return

        features = self.dcfg.get("features") or {}
        prior_cfg = features.get("atf_foa_prior", None)
        has_prior = prior_cfg is not None and prior_cfg.get("enabled", True)

        expected = 8 + 8 * int(has_prior and self.prior_as_input)

        if int(in_channels) != expected:
            raise ValueError(
                f"model.in_channels={in_channels} does not match the enabled "
                f"input features ({expected} channels: 4 microphones"
                + (" + 4 FOA prior" if has_prior and self.prior_as_input else "")
                + ", real and imaginary)"
            )

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch[0], batch[1]
        prior = batch[2] if len(batch) > 2 else None

        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)

        if prior is not None:
            prior = prior.to(self.device, dtype=self.dtype)
        elif self.prior_combine != "none":
            raise ValueError(
                f"training.prior_combine={self.prior_combine} needs the dataset "
                "to provide a FOA prior; enable data.features.atf_foa_prior"
            )

        x_complex = x
        if not torch.is_complex(x):
            x_complex = ri_to_channels(x)

        model_in = x_complex
        if prior is not None and self.prior_as_input:
            model_in = torch.cat([x_complex, ri_to_channels(prior)], dim=1)

        pred = self(model_in)
        pred = channels_to_ri(pred)

        if self.prior_combine == "residual":
            y_hat = (prior + pred).contiguous().to(dtype=x.dtype)
        elif self.prior_combine == "mask":
            y_hat = self.masking_fn(prior, pred)
        elif self.masking_fn is None:
            y_hat = pred.contiguous().to(dtype=x.dtype)
        else:
            y_hat = self.masking_fn(x, pred)

        loss = self.calculate_losses(y_hat, y, stage)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def calculate_losses(self, estimate, target, stage):
        total_loss = 0.0

        for loss, loss_params in self.stft_losses.items():
            loss_fn = SPECTROGRAM_LOSSES[loss]
            loss_val = loss_fn(estimate, target, **loss_params)
            total_loss += loss_val * loss_params["weight"]
            self.log(
                f"{stage}_{loss}_loss", loss_val, prog_bar=True, on_epoch=True
            )

        estimate = istft_from_spectrogram(estimate, **self.dcfg["stft"])
        target = istft_from_spectrogram(target, **self.dcfg["stft"])

        for loss, loss_params in self.wave_losses.items():
            loss_fn = WAVE_LOSSES[loss]
            loss_val = loss_fn(estimate, target, **loss_params)
            total_loss += loss_val * loss_params["weight"]
            self.log(
                f"{stage}_{loss}_loss", loss_val, prog_bar=True, on_epoch=True
            )

        self.log(
            f"{stage}_total_loss", total_loss, prog_bar=True, on_epoch=True
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

    def on_before_optimizer_step(self, optimizer):
        clip_val = self.tcfg.get("grad_clip_val", None)
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
