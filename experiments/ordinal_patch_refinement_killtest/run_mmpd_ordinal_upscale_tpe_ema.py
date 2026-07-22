#!/usr/bin/env python3
"""Official MMPD-equivalent TPE/EMA entry point for ordinal upscaling."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hold the repo namespaces before the module imports temp/MMPD, whose utils/
# package would otherwise shadow them when this file is invoked directly.
from models.diffusion_tsf.ordinal_window_norm import ordinal_encode  # noqa: F401, E402
from utils.eval_discriminator_texture_staged_vs_mmpd import HorizonSliceDataset  # noqa: F401, E402
from experiments.ordinal_patch_refinement_killtest import mmpd_ordinal_upscale as experiment  # noqa: E402


class _Ema:
    """Matches the MMPD train-loop floating-state EMA behavior."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        backup = {key: state[key].detach().clone() for key in self.shadow}
        for key, value in self.shadow.items():
            state[key].copy_(value)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)


def _fit_with_mmpd_schedule(
    train: dict[str, torch.Tensor], val: dict[str, torch.Tensor], config: Any, device: torch.device, *,
    lr: float, point_weight: float, batch_size: int, dropout: float, epochs: int, patience: int, seed: int,
):
    experiment._set_seed(seed)
    model = experiment._make_model(config, train["condition"].shape[1], dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = _Ema(model, decay=0.99)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train["condition"], train["target_hi"]), batch_size=batch_size, shuffle=True,
        generator=generator, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val["condition"], val["target_hi"]), batch_size=batch_size, shuffle=False,
        pin_memory=device.type == "cuda",
    )
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for condition, target in train_loader:
            loss = model.compute_loss(condition.to(device), target.to(device), point_weight=point_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)
        backup = ema.swap_in(model)
        val_loss = experiment._epoch(model, val_loader, device, point_weight, None)
        ema.restore(model, backup)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            backup = ema.swap_in(model)
            best_state = copy.deepcopy(model.state_dict())
            ema.restore(model, backup)
        else:
            stale += 1
            if stale >= patience:
                break
        # The established MMPD command defaults --lradj cosine.
        next_lr = 0.01 * lr + 0.99 * lr * (1.0 + math.cos(math.pi * epoch / epochs)) / 2.0
        for group in optimizer.param_groups:
            group["lr"] = next_lr
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def _tpe_tune_and_refit(train: dict[str, Any], val: dict[str, Any], config: Any, device: Any, seed: int, smoke: bool):
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), study_name="mmpd_ordinal_upscale")
    history: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": config.learning_rate if smoke else trial.suggest_float("learning_rate", config.tune_lr_min, config.tune_lr_max, log=True),
            "point_weight": config.point_weight if smoke else trial.suggest_float("point_weight", config.tune_point_weight_min, config.tune_point_weight_max, log=True),
            "batch_size": config.batch_size if smoke else trial.suggest_int("batch_size", config.tune_batch_min, config.tune_batch_max),
            "dropout": config.dropout,
        }
        model, val_loss, best_epoch = _fit_with_mmpd_schedule(
            train, val, config, device, seed=seed + trial.number, lr=float(params["learning_rate"]),
            point_weight=float(params["point_weight"]), batch_size=int(params["batch_size"]), dropout=float(params["dropout"]),
            epochs=1 if smoke else config.tune_epochs, patience=1 if smoke else config.tune_patience,
        )
        del model
        history.append({"trial": trial.number, **params, "val_loss": val_loss, "best_epoch": best_epoch, "ema_decay": 0.99, "lradj": "cosine"})
        return val_loss

    study.optimize(objective, n_trials=1 if smoke else config.tune_trials, show_progress_bar=False)
    best = study.best_trial
    params = {
        "trial": best.number, "val_loss": float(best.value), "learning_rate": float(best.params.get("learning_rate", config.learning_rate)),
        "point_weight": float(best.params.get("point_weight", config.point_weight)), "batch_size": int(best.params.get("batch_size", config.batch_size)),
        "dropout": config.dropout, "ema_decay": 0.99, "lradj": "cosine",
    }
    model, refit_loss, refit_epoch = _fit_with_mmpd_schedule(
        train, val, config, device, seed=seed + 1000, lr=params["learning_rate"], point_weight=params["point_weight"],
        batch_size=params["batch_size"], dropout=params["dropout"], epochs=1 if smoke else config.train_epochs,
        patience=1 if smoke else config.patience,
    )
    return model, {**params, "refit_val_loss": refit_loss, "refit_best_epoch": refit_epoch}, history


experiment._tune_and_refit = _tpe_tune_and_refit

if __name__ == "__main__":
    experiment.main()
