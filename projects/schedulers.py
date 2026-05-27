import torch


def build_lr_scheduler(optimizer, sched_cfg, module):
    sched_type = sched_cfg["type"].lower()

    if sched_type == "step":
        interval = sched_cfg.get("interval", "epoch")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.5),
        )
    elif sched_type == "cosine":
        interval = sched_cfg.get("interval", "step")
        scheduler = _build_cosine_scheduler(optimizer, sched_cfg, interval, module)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_cfg['type']}")

    return {
        "scheduler": scheduler,
        "interval": interval,
        "frequency": sched_cfg.get("frequency", 1),
        "name": sched_cfg.get("name", "lr"),
    }


def _build_cosine_scheduler(optimizer, sched_cfg, interval, module):
    total_iters = _resolve_total_iters(sched_cfg, interval, module)
    warmup_iters = _resolve_warmup_iters(sched_cfg, interval, total_iters)
    min_lr = sched_cfg.get("min_lr", sched_cfg.get("eta_min", 0.0))

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_iters - warmup_iters,
        eta_min=min_lr,
    )

    if warmup_iters == 0:
        return cosine

    # Warmup is linear from a low LR to the optimizer's configured LR,
    # then cosine decay owns the remaining training span.
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=sched_cfg.get("warmup_start_factor", 1e-3),
        end_factor=1.0,
        total_iters=warmup_iters,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_iters],
    )


def _resolve_total_iters(sched_cfg, interval, module):
    total_key = "total_steps" if interval == "step" else "total_epochs"
    total_iters = sched_cfg.get(total_key, None)

    if total_iters in (None, "auto"):
        total_iters = (
            _trainer_value(module, "estimated_stepping_batches")
            if interval == "step"
            else _trainer_value(module, "max_epochs")
        )

    if total_iters is None:
        total_iters = sched_cfg.get("T_max", None)

    if total_iters is None or total_iters <= 0:
        raise ValueError(
            f"Cosine scheduler needs a positive {total_key} or T_max value."
        )

    return int(total_iters)


def _resolve_warmup_iters(sched_cfg, interval, total_iters):
    warmup_key = "warmup_steps" if interval == "step" else "warmup_epochs"
    warmup_iters = sched_cfg.get(warmup_key, 0)

    if warmup_iters == 0 and "warmup_ratio" in sched_cfg:
        warmup_iters = round(total_iters * sched_cfg["warmup_ratio"])

    # Keep at least one cosine step after warmup.
    return min(max(int(warmup_iters), 0), total_iters - 1)


def _trainer_value(module, attr):
    try:
        return getattr(module.trainer, attr)
    except RuntimeError:
        return None
