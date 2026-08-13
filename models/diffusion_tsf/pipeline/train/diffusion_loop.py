"""Shared diffusion train/val epoch loops."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from models.diffusion_tsf.pipeline.train.checkpointing import amp_context


def train_diffusion_epoch(
    model,
    train_loader: DataLoader,
    device: torch.device,
    optimizer,
    *,
    accum_steps: int = 1,
    clip_grad: float = 1.0,
    set_loader_mode: Optional[Callable] = None,
    set_training_epoch: Optional[Callable] = None,
    epoch: Optional[int] = None,
    ema=None,
) -> float:
    if set_training_epoch is not None and epoch is not None:
        set_training_epoch(train_loader, epoch)

    model.train()
    if set_loader_mode is not None:
        set_loader_mode(model, train_loader, eval_mode=False)

    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    accum_steps = max(1, int(accum_steps))

    for batch_idx, (past, future) in enumerate(train_loader):
        past, future = past.to(device), future.to(device)
        with amp_context(bool(model.config.use_amp)):
            loss = model.get_loss(past, future) / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.item()) * accum_steps
        n_batches += 1

    if accum_steps > 1 and len(train_loader) % accum_steps != 0:
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    return total_loss / max(n_batches, 1)


def validate_diffusion_epoch(
    model,
    val_loader: DataLoader,
    device: torch.device,
    *,
    set_loader_mode: Optional[Callable] = None,
) -> float:
    model.eval()
    if set_loader_mode is not None:
        set_loader_mode(model, val_loader, eval_mode=True)

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for past, future in val_loader:
            past, future = past.to(device), future.to(device)
            with amp_context(bool(model.config.use_amp)):
                loss = model.get_loss(past, future)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)
