import torch
import pytorch_lightning as pl

from models import MODELS
from utils.audio import istft_from_spectrogram
from utils.complex import ri_to_channels, channels_to_ri
from utils.losses import SPECTROGRAM_LOSSES, WAVE_LOSSES
from utils.masking import MASKS


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.mcfg = cfg["model"]
        self.tcfg = cfg["training"]
        self.dcfg = cfg["data"]

        self.model = MODELS[self.mcfg["model_name"]](self.mcfg).to(self.device, dtype=self.dtype)

        self.masking_fn = MASKS[self.tcfg.get("masking", "complex")]

        self.lr = self.tcfg.get("lr", 1e-3)
        self.weight_decay = self.tcfg.get("weight_decay", 0.0)

        self.stft_losses = self.tcfg["losses"].get("stft_losses", {})
        self.wave_losses = self.tcfg["losses"].get("wave_losses", {})

        self.save_hyperparameters(cfg)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch

        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)

        x_complex = x
        if not torch.is_complex(x):
            x_complex = ri_to_channels(x)

        pred = self(x_complex)
        pred = channels_to_ri(pred)
        
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
            self.log(f"{stage}_{loss}", loss_val, prog_bar=True, on_epoch=True)

        estimate = istft_from_spectrogram(estimate, **self.dcfg["stft"])
        target = istft_from_spectrogram(target, **self.dcfg["stft"])

        for loss, loss_params in self.wave_losses.items():
            loss_fn = WAVE_LOSSES[loss]
            loss_val = loss_fn(estimate, target, **loss_params)
            total_loss += loss_val * loss_params["weight"]
            self.log(f"{stage}_{loss}", loss_val, prog_bar=True, on_epoch=True)

        self.log(f"{stage}_loss", total_loss, prog_bar=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        sched_cfg = self.dcfg.get("training", {}).get("scheduler", None)

        if sched_cfg is None:
            return optimizer

        if sched_cfg["type"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.5)
            )

        elif sched_cfg["type"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("T_max", 50)
            )
        else:
            raise ValueError("Unsupported scheduler type")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_before_optimizer_step(self, optimizer):
        clip_val = self.tcfg.get("grad_clip_val", None)
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
