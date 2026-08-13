"""Checkpoint I/O and early stopping."""

from __future__ import annotations

import os

import torch


class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def amp_context(use_amp: bool):
    if use_amp and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def ensure_checkpoint_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if not parent or os.path.isdir(parent):
        return
    if os.path.isfile(parent):
        raise FileExistsError(
            f"checkpoint parent exists as a file, not a directory: {parent}"
        )
    try:
        os.makedirs(parent, exist_ok=True)
    except FileExistsError:
        if not os.path.isdir(parent):
            raise


def _diffusion_state_dict_without_guidance(model) -> dict:
    """Drop frozen guidance weights — guidance is loaded separately at runtime."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("guidance_model.")
    }


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, path, extra=None):
    ensure_checkpoint_dir(path)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": _diffusion_state_dict_without_guidance(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    if extra:
        ckpt.update(extra)
    tmp_path = f"{path}.tmp"
    try:
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
    except OSError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        err = getattr(e, "errno", None)
        if err in (28, 122) or "quota" in str(e).lower() or "no space" in str(e).lower():
            raise RuntimeError(
                f"Disk quota/space exhausted while saving {path}. "
                "Free scratch (old results/ckpts, wandb, trial_*.pt) then --resume."
            ) from e
        raise
    except RuntimeError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        if "quota" in str(e).lower() or "no space" in str(e).lower():
            raise RuntimeError(
                f"Disk quota/space exhausted while saving {path}. "
                "Free scratch (old results/ckpts, wandb, trial_*.pt) then --resume."
            ) from e
        raise
