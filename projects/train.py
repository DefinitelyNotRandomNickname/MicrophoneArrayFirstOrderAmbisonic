import argparse
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils.configs import load_and_merge_configs
from data.hdf5_dataset import SpatialAudioDataset
from projects.train_module import TrainingModule


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_cfg", type=str, required=True)
    parser.add_argument("--model_cfg", type=str, required=True)
    parser.add_argument("--train_cfg", type=str, required=True)

    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--log_dir", type=str, default=".logs")

    return parser.parse_args()


def build_dataloaders(cfg):
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    train_rir = dcfg["train_rir_path"]
    valid_rir = dcfg["valid_rir_path"]

    train_ds = SpatialAudioDataset(cfg, train_rir)
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 16),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    valid_ds = SpatialAudioDataset(cfg, valid_rir)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=tcfg.get("num_workers", 16),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    return train_loader, valid_loader


def build_model(cfg):
    model = TrainingModule(cfg)
    return model


def build_trainer(cfg, args):
    tcfg = cfg["training"]

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name,
        default_hp_metric=False,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=tcfg.get("monitor", "val_loss"),
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=tcfg.get("max_epochs", 100),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=tcfg.get("strategy", "auto"),
        devices=tcfg.get("devices", 1),
        precision=tcfg.get("precision", 32),
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=tcfg.get("grad_clip_val", 1.0),
    )

    return trainer


def main():
    args = parse_args()

    cfg = load_and_merge_configs(
        args.data_cfg,
        args.model_cfg,
        args.train_cfg,
    )

    pl.seed_everything(cfg["training"].get("seed", 42))

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg)

    trainer = build_trainer(cfg, args)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
