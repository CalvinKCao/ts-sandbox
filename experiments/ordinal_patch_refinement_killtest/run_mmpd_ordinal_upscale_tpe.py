#!/usr/bin/env python3
"""Cluster-safe TPE entry point for the 1D MMPD ordinal upscaling kill test."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load repo namespaces before the vendored MMPD tree is imported by the module.
from models.diffusion_tsf.ordinal_window_norm import ordinal_encode  # noqa: F401, E402
from utils.eval_discriminator_texture_staged_vs_mmpd import HorizonSliceDataset  # noqa: F401, E402

from experiments.ordinal_patch_refinement_killtest import mmpd_ordinal_upscale as experiment  # noqa: E402


def _tpe_tune_and_refit(
    train: dict[str, Any], val: dict[str, Any], config: Any, device: Any, seed: int, smoke: bool,
):
    """Use the same seeded Optuna TPE semantics as utils/mmpd_subset_tune.py."""
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name="mmpd_ordinal_upscale",
    )
    history: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        if smoke:
            params = {
                "learning_rate": config.learning_rate,
                "point_weight": config.point_weight,
                "batch_size": config.batch_size,
                "dropout": config.dropout,
            }
        else:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", config.tune_lr_min, config.tune_lr_max, log=True,
                ),
                "point_weight": trial.suggest_float(
                    "point_weight", config.tune_point_weight_min, config.tune_point_weight_max, log=True,
                ),
                "batch_size": trial.suggest_int("batch_size", config.tune_batch_min, config.tune_batch_max),
                "dropout": config.dropout,
            }
        model, val_loss, best_epoch = experiment._fit(
            train, val, config, device, seed=seed + trial.number,
            lr=float(params["learning_rate"]), point_weight=float(params["point_weight"]),
            batch_size=int(params["batch_size"]), dropout=float(params["dropout"]),
            epochs=1 if smoke else config.tune_epochs,
            patience=1 if smoke else config.tune_patience,
        )
        del model
        history.append({"trial": trial.number, **params, "val_loss": val_loss, "best_epoch": best_epoch})
        return val_loss

    study.optimize(objective, n_trials=1 if smoke else config.tune_trials, show_progress_bar=False)
    best = study.best_trial
    params = {
        "learning_rate": float(best.params.get("learning_rate", config.learning_rate)),
        "point_weight": float(best.params.get("point_weight", config.point_weight)),
        "batch_size": int(best.params.get("batch_size", config.batch_size)),
        "dropout": config.dropout,
        "trial": best.number,
        "val_loss": float(best.value),
    }
    model, refit_loss, refit_epoch = experiment._fit(
        train, val, config, device, seed=seed + 1000,
        lr=params["learning_rate"], point_weight=params["point_weight"],
        batch_size=params["batch_size"], dropout=params["dropout"],
        epochs=1 if smoke else config.train_epochs,
        patience=1 if smoke else config.patience,
    )
    return model, {**params, "refit_val_loss": refit_loss, "refit_best_epoch": refit_epoch}, history


experiment._tune_and_refit = _tpe_tune_and_refit

if __name__ == "__main__":
    experiment.main()
