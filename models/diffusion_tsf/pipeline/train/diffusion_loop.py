"""Shared diffusion train/val epoch loops."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from models.diffusion_tsf.pipeline.train.checkpointing import amp_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffusionEpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    selection_score: float
    best_val: float
    lr: float
    saved: bool
    train_seconds: float
    val_seconds: float
    epoch_seconds: float


@dataclass(frozen=True)
class DiffusionTrainingResult:
    best_val: float
    best_epoch: int
    history: List[DiffusionEpochMetrics]
    elapsed_seconds: float


class ExponentialMovingAverage:
    """Parameter EMA used only for validation and best-checkpoint selection."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, avg in self.shadow.items():
            avg.mul_(self.decay).add_(model.state_dict()[key].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        state = model.state_dict()
        backup = {key: state[key].detach().clone() for key in self.shadow}
        for key, avg in self.shadow.items():
            state[key].copy_(avg)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)


class DiffusionTrainer:
    """Own the raw diffusion optimization loop, independent of pipeline phases.

    Pipeline phases supply batch decoding, dataset/ordinal-mode hooks, checkpoint
    callbacks, and Optuna reporting. This class only performs PyTorch iteration.
    """

    def __init__(
        self,
        *,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        optimizer,
        accum_steps: int = 1,
        clip_grad: Optional[float] = 1.0,
        scheduler=None,
        ema_decay: float = 0.0,
        unpack_batch: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, Any, Any]]] = None,
        set_loader_mode: Optional[Callable] = None,
        set_training_epoch: Optional[Callable[[DataLoader, int], None]] = None,
        sequential_anchor_backward: bool = False,
        log_prefix: str = "diffusion",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.accum_steps = max(1, int(accum_steps))
        self.clip_grad = clip_grad
        self.scheduler = scheduler
        self.ema = ExponentialMovingAverage(model, ema_decay) if ema_decay else None
        self.unpack_batch = unpack_batch or self._unpack_standard_batch
        self.set_loader_mode = set_loader_mode
        self.set_training_epoch = set_training_epoch
        self.sequential_anchor_backward = bool(sequential_anchor_backward)
        self.log_prefix = log_prefix

    @staticmethod
    def _unpack_standard_batch(batch):
        past, future = batch
        return past, future, None, None

    def _inputs_from_batch(self, batch):
        past, future, patch_col0, variate_keep = self.unpack_batch(batch)
        past = past.to(self.device)
        future = future.to(self.device)
        if patch_col0 is not None:
            patch_col0 = patch_col0.to(self.device)
        if variate_keep is not None:
            variate_keep = variate_keep.to(self.device)
        return past, future, patch_col0, variate_keep

    def _loss_from_inputs(
        self,
        inputs,
        *,
        loss_mode: str = "combined",
        scale_for_accumulation: bool,
    ) -> torch.Tensor:
        past, future, patch_col0, variate_keep = inputs
        with amp_context(bool(self.model.config.use_amp)):
            loss = self.model.get_loss(
                past,
                future,
                patch_col0=patch_col0,
                variate_keep=variate_keep,
                loss_mode=loss_mode,
            )
        return loss / self.accum_steps if scale_for_accumulation else loss

    def _loss_from_batch(self, batch, *, scale_for_accumulation: bool) -> torch.Tensor:
        return self._loss_from_inputs(
            self._inputs_from_batch(batch),
            scale_for_accumulation=scale_for_accumulation,
        )

    def _train_epoch(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        if self.set_training_epoch is not None:
            self.set_training_epoch(self.train_loader, epoch)
        self.model.train()
        if self.set_loader_mode is not None:
            self.set_loader_mode(self.model, self.train_loader, eval_mode=False)
        total_loss = 0.0
        n_batches = 0
        n_train_batches = len(self.train_loader)
        log_stride = max(1, n_train_batches // 4)
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx == 0 or (batch_idx + 1) % log_stride == 0 or batch_idx + 1 == n_train_batches:
                logger.info(
                    "  [%s] epoch %d/%d train_batch %d/%d",
                    self.log_prefix, epoch + 1, max_epochs, batch_idx + 1, n_train_batches,
                )
            if self.sequential_anchor_backward and bool(
                self.model.config.use_deterministic_anchor_loss
            ):
                # Both passes must see the same sampled timestep and conditioning
                # dropout as the original combined forward.  Replaying the RNG state
                # keeps this memory-saving split algebraically equivalent per update.
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda"
                    else None
                )
                regular_inputs = self._inputs_from_batch(batch)
                regular_loss = self._loss_from_inputs(
                    regular_inputs,
                    loss_mode="regular",
                    scale_for_accumulation=False,
                )
                anchor_weight = 1.0 - float(self.model.config.deterministic_anchor_lambda)
                regular_weight = float(self.model.config.deterministic_anchor_lambda)
                (regular_loss * regular_weight / self.accum_steps).backward()
                regular_value = float(regular_loss.detach().item())
                del regular_loss, regular_inputs

                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, self.device)
                anchor_inputs = self._inputs_from_batch(batch)
                anchor_loss = self._loss_from_inputs(
                    anchor_inputs,
                    loss_mode="anchor",
                    scale_for_accumulation=False,
                )
                (anchor_loss * anchor_weight / self.accum_steps).backward()
                loss = regular_weight * regular_value + anchor_weight * float(anchor_loss.detach().item())
                del anchor_loss, anchor_inputs
            else:
                loss = self._loss_from_batch(batch, scale_for_accumulation=True)
                loss.backward()
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
            batch_loss = float(loss.item()) * self.accum_steps if torch.is_tensor(loss) else float(loss)
            total_loss += batch_loss
            n_batches += 1
        if self.accum_steps > 1 and n_train_batches % self.accum_steps != 0:
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)
        return total_loss / max(n_batches, 1), time.perf_counter() - started

    def _validate_epoch(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        self.model.eval()
        if self.set_loader_mode is not None:
            self.set_loader_mode(self.model, self.val_loader, eval_mode=True)
        total_loss = 0.0
        n_batches = 0
        n_val_batches = len(self.val_loader)
        log_stride = max(1, n_val_batches // 2)
        started = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx == 0 or (batch_idx + 1) % log_stride == 0 or batch_idx + 1 == n_val_batches:
                    logger.info(
                        "  [%s] epoch %d/%d val_batch %d/%d",
                        self.log_prefix, epoch + 1, max_epochs, batch_idx + 1, n_val_batches,
                    )
                loss = self._loss_from_batch(batch, scale_for_accumulation=False)
                total_loss += float(loss.item())
                n_batches += 1
        return total_loss / max(n_batches, 1), time.perf_counter() - started

    def fit(
        self,
        *,
        max_epochs: int,
        early_stopping: Optional[Callable[[float], bool]] = None,
        on_best: Optional[Callable[[DiffusionEpochMetrics], None]] = None,
        on_epoch_end: Optional[Callable[[DiffusionEpochMetrics], None]] = None,
    ) -> DiffusionTrainingResult:
        best_val = float("inf")
        best_epoch = 0
        history: List[DiffusionEpochMetrics] = []
        started = time.perf_counter()
        for epoch in range(max_epochs):
            epoch_started = time.perf_counter()
            logger.info("  [%s] epoch %d/%d train_start", self.log_prefix, epoch + 1, max_epochs)
            train_loss, train_seconds = self._train_epoch(epoch, max_epochs)
            logger.info(
                "  [%s] epoch %d/%d train_done loss=%.4f time=%.1fs",
                self.log_prefix, epoch + 1, max_epochs, train_loss, train_seconds,
            )
            if self.scheduler is not None:
                self.scheduler.step()
            backup = self.ema.swap_in(self.model) if self.ema is not None else None
            logger.info("  [%s] epoch %d/%d val_start", self.log_prefix, epoch + 1, max_epochs)
            val_loss, val_seconds = self._validate_epoch(epoch, max_epochs)
            selection_score = float(val_loss)
            saved = selection_score < best_val
            if saved:
                best_val = selection_score
                best_epoch = epoch + 1
            metrics = DiffusionEpochMetrics(
                epoch=epoch + 1,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                selection_score=selection_score,
                best_val=float(best_val),
                lr=float(self.optimizer.param_groups[0]["lr"]),
                saved=saved,
                train_seconds=train_seconds,
                val_seconds=val_seconds,
                epoch_seconds=time.perf_counter() - epoch_started,
            )
            try:
                if saved and on_best is not None:
                    on_best(metrics)
            finally:
                if backup is not None:
                    self.ema.restore(self.model, backup)
            history.append(metrics)
            logger.info(
                "  [%s] epoch %d/%d done train=%.4f val=%.4f best=%.4f best_ep=%d "
                "lr=%.2e saved=%s train_t=%.1fs val_t=%.1fs epoch_t=%.1fs",
                self.log_prefix, metrics.epoch, max_epochs, metrics.train_loss,
                metrics.val_loss, metrics.best_val, best_epoch, metrics.lr, metrics.saved,
                metrics.train_seconds, metrics.val_seconds, metrics.epoch_seconds,
            )
            if on_epoch_end is not None:
                on_epoch_end(metrics)
            if early_stopping is not None and early_stopping(selection_score):
                logger.info(
                    "  [%s] epoch %d/%d early_stop val=%.4f best=%.4f best_ep=%d",
                    self.log_prefix, metrics.epoch, max_epochs, selection_score, best_val, best_epoch,
                )
                break
        return DiffusionTrainingResult(
            best_val=float(best_val),
            best_epoch=best_epoch,
            history=history,
            elapsed_seconds=time.perf_counter() - started,
        )


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
