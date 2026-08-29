"""Shared diffusion train/val epoch loops."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Sampler

from models.diffusion_tsf.diffusion_model import NoVisiblePatchTransitions
from models.diffusion_tsf.pipeline.train.checkpointing import amp_context
from models.diffusion_tsf.pipeline.train.univariate_microbatch import (
    iter_flat_row_slices,
    loss_scale_for_rows,
    next_row_take,
)

logger = logging.getLogger(__name__)


def fp32_window_nbytes(
    n_variates: int,
    lookback: int,
    horizon: int,
    overlap: int,
) -> int:
    """Bytes for one loader window: fp32 past (V, L) + future (V, K+H)."""
    n_variates = int(n_variates)
    lookback = int(lookback)
    horizon = int(horizon)
    overlap = int(overlap)
    if n_variates < 1 or lookback < 1 or horizon < 1:
        raise ValueError(
            "fp32 window nbytes needs n_variates, lookback, horizon >= 1, got "
            f"V={n_variates} L={lookback} H={horizon} K={overlap}"
        )
    if overlap < 0:
        raise ValueError(f"lookback_overlap must be >= 0, got {overlap}")
    return n_variates * (lookback + overlap + horizon) * 4


def resolve_train_epoch_groups(
    *,
    n_samples: int,
    batch_size: int,
    n_groups: int,
    max_bytes: Optional[int] = None,
    n_variates: Optional[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[int, Optional[int], Optional[int]]:
    """Pick N so a packed group stays under ``max_bytes`` when that cap is set.

    ``train_epoch_max_bytes`` wins over an explicit ``train_epoch_groups > 1``.
    One compile-constant batch is atomic: if a single batch already exceeds the
    cap, this fails rather than splitting B.
    Returns ``(n_groups, bytes_per_window, bytes_per_group_upper)``.
    """
    n_groups = int(n_groups)
    if n_groups < 1:
        raise ValueError(f"train_epoch_groups must be >= 1, got {n_groups!r}")
    n_samples = int(n_samples)
    batch_size = int(batch_size)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_bytes is None:
        return n_groups, None, None
    max_bytes = int(max_bytes)
    if max_bytes < 1:
        raise ValueError(f"train_epoch_max_bytes must be >= 1, got {max_bytes!r}")
    missing = [
        name
        for name, val in (
            ("n_variates", n_variates),
            ("lookback", lookback),
            ("horizon", horizon),
            ("overlap", overlap),
        )
        if val is None
    ]
    if missing:
        raise ValueError(
            f"train_epoch_max_bytes={max_bytes} requires {missing}; "
            "cannot compute bytes/window"
        )
    nbytes = fp32_window_nbytes(n_variates, lookback, horizon, overlap)
    batch_bytes = nbytes * batch_size
    if batch_bytes > max_bytes:
        raise ValueError(
            f"one packed batch is {batch_bytes} bytes "
            f"(B={batch_size} * {nbytes}/window) > train_epoch_max_bytes={max_bytes}; "
            "cannot split a compile-constant batch"
        )
    max_windows = max_bytes // nbytes
    max_batches = max(1, max_windows // batch_size)
    n_batches = (n_samples + batch_size - 1) // batch_size
    computed = max(1, (n_batches + max_batches - 1) // max_batches)
    if n_groups > 1 and n_groups != computed:
        logger.info(
            "train_epoch_max_bytes=%d wins over train_epoch_groups=%d -> N=%d "
            "(window=%d B=%d batch_bytes=%d max_batches/group=%d packed_batches=%d)",
            max_bytes, n_groups, computed, nbytes, batch_size, batch_bytes,
            max_batches, n_batches,
        )
    else:
        logger.info(
            "train_epoch_max_bytes=%d -> N=%d "
            "(window=%d B=%d batch_bytes=%d max_batches/group=%d packed_batches=%d)",
            max_bytes, computed, nbytes, batch_size, batch_bytes,
            max_batches, n_batches,
        )
    return computed, nbytes, max_batches * batch_bytes


def pack_constant_size_batches(
    indices: Sequence[int],
    batch_size: int,
) -> Tuple[List[Tuple[int, ...]], int]:
    """Pack shuffled indices into batches of ``batch_size``.

    The last incomplete batch is padded by repeating indices already in that
    batch so every batch has length B (needed for torch.compile shapes).
    Returns ``(batches, n_padded)``. Does not drop leftover windows.
    """
    packed = [int(i) for i in indices]
    n = len(packed)
    bsz = int(batch_size)
    if bsz < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if n == 0:
        raise ValueError("cannot pack an empty index list into batches")
    batches: List[Tuple[int, ...]] = []
    n_padded = 0
    for start in range(0, n, bsz):
        chunk = packed[start:start + bsz]
        if len(chunk) < bsz:
            need = bsz - len(chunk)
            chunk = chunk + [chunk[i % len(chunk)] for i in range(need)]
            n_padded += need
        batches.append(tuple(chunk))
    return batches, n_padded


def split_batches_into_groups(
    batches: Sequence[Sequence[int]],
    n_groups: int,
) -> List[List[Tuple[int, ...]]]:
    """Split a packed batch list into ``n_groups`` contiguous groups.

    Fails if any group would have zero batches (cannot fill one batch).
    """
    n_groups = int(n_groups)
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups!r}")
    batch_tuples = [tuple(int(i) for i in batch) for batch in batches]
    n_batches = len(batch_tuples)
    if n_batches < n_groups:
        raise ValueError(
            f"train_epoch_groups={n_groups} but only {n_batches} packed batches; "
            "a group would have zero batches"
        )
    base, extra = divmod(n_batches, n_groups)
    groups: List[List[Tuple[int, ...]]] = []
    cursor = 0
    for group_i in range(n_groups):
        take = base + (1 if group_i < extra else 0)
        group = batch_tuples[cursor:cursor + take]
        cursor += take
        if not group:
            raise ValueError(
                f"train_epoch_groups={n_groups} left group {group_i} empty"
            )
        groups.append(group)
    return groups


class EpochGroupBatchSampler(Sampler[List[int]]):
    """Pack shuffled indices to constant B, then cycle N batch-groups.

    Epoch ``e`` yields group ``e % N``. After each full cycle (when
    ``e // N`` advances), indices are reshuffled and batches are repacked so
    the next cycle does not repeat the same batch membership. Batch order
    inside a group is shuffled every epoch. ``batch_size`` is constant.
    """

    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        n_groups: int,
        *,
        seed: int,
        smoke_test: bool = False,
    ) -> None:
        n_samples = int(n_samples)
        batch_size = int(batch_size)
        n_groups = int(n_groups)
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {n_groups}")
        n_batches = (n_samples + batch_size - 1) // batch_size
        groups_used = n_groups
        if bool(smoke_test) and n_batches < n_groups:
            logger.info(
                "smoke-test: clamping train_epoch_groups %d -> %d (only %d packed batches)",
                n_groups, n_batches, n_batches,
            )
            groups_used = n_batches
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_groups = groups_used
        self.n_groups_requested = n_groups
        self.seed = int(seed)
        self.epoch = 0
        self._cycle = 0
        self._repack_for_cycle(0)

    def _repack_for_cycle(self, cycle: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(self.seed + 2_000_003 * int(cycle))
        shuffled = torch.randperm(self.n_samples, generator=gen).tolist()
        packed, n_padded = pack_constant_size_batches(shuffled, self.batch_size)
        groups = split_batches_into_groups(packed, self.n_groups)
        self.groups = groups
        self.n_padded = int(n_padded)
        logger.info(
            "epoch-group sampler: cycle=%d n=%d B=%d groups=%d (requested=%d) "
            "packed_batches=%d padded=%d group_batch_counts=%s",
            int(cycle),
            self.n_samples,
            self.batch_size,
            self.n_groups,
            self.n_groups_requested,
            len(packed),
            n_padded,
            [len(g) for g in groups],
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        cycle = self.epoch // self.n_groups
        if cycle != self._cycle:
            self._repack_for_cycle(cycle)
            self._cycle = cycle

    def _group(self) -> List[Tuple[int, ...]]:
        return self.groups[self.epoch % self.n_groups]

    def __iter__(self) -> Iterable[List[int]]:
        group = self._group()
        gen = torch.Generator()
        gen.manual_seed(self.seed + 1_000_003 * self.epoch)
        order = torch.randperm(len(group), generator=gen).tolist()
        for idx in order:
            yield list(group[idx])

    def __len__(self) -> int:
        return len(self._group())


def make_epoch_group_train_loader(
    dataset,
    *,
    batch_size: int,
    n_groups: int,
    seed: int,
    smoke_test: bool = False,
) -> DataLoader:
    sampler = EpochGroupBatchSampler(
        len(dataset),
        batch_size,
        n_groups,
        seed=int(seed),
        smoke_test=bool(smoke_test),
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def make_grouped_train_loader(
    dataset,
    *,
    batch_size: int,
    n_groups: int,
    seed: int,
    max_bytes: Optional[int] = None,
    n_variates: Optional[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    overlap: Optional[int] = None,
    smoke_test: bool = False,
) -> Tuple[DataLoader, int, Optional[int], Optional[int]]:
    """Build a constant-B grouped loader; ``max_bytes`` wins when set.

    Returns ``(loader, n_groups_used, bytes_per_window, bytes_per_group_upper)``.
    """
    n_groups_used, nbytes, group_bytes = resolve_train_epoch_groups(
        n_samples=len(dataset),
        batch_size=int(batch_size),
        n_groups=int(n_groups),
        max_bytes=max_bytes,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
    )
    loader = make_epoch_group_train_loader(
        dataset,
        batch_size=int(batch_size),
        n_groups=n_groups_used,
        seed=int(seed),
        smoke_test=bool(smoke_test),
    )
    return loader, n_groups_used, nbytes, group_bytes


def log_epoch_shard_contract(
    *,
    name: str,
    n_groups: int,
    max_epochs: int,
    patience: Optional[int],
) -> None:
    """Patience / max_epochs count shards when N>1, not full dataset passes."""
    if int(n_groups) <= 1:
        return
    patience_s = "none" if patience is None else str(int(patience))
    logger.info(
        "  [%s] epoch is a shard: train_epoch_groups=%d so patience=%s and "
        "max_epochs=%d count shards, not full dataset passes "
        "(one full pass = %d shards)",
        name, int(n_groups), patience_s, int(max_epochs), int(n_groups),
    )


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
        unpack_batch: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, Any]]] = None,
        set_loader_mode: Optional[Callable] = None,
        set_training_epoch: Optional[Callable[[DataLoader, int], None]] = None,
        sequential_anchor_backward: bool = False,
        deterministic_anchor_every_n_batches: int = 1,
        train_token_cache=None,
        val_token_cache=None,
        log_prefix: str = "diffusion",
        univariate_micro_batch: Optional[int] = None,
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
        self.deterministic_anchor_every_n_batches = int(deterministic_anchor_every_n_batches)
        self.train_token_cache = train_token_cache
        self.val_token_cache = val_token_cache
        if self.deterministic_anchor_every_n_batches < 1:
            raise ValueError("deterministic_anchor_every_n_batches must be >= 1")
        self.log_prefix = log_prefix
        self.univariate_micro_batch = (
            None if univariate_micro_batch is None else max(1, int(univariate_micro_batch))
        )

    @staticmethod
    def _unpack_standard_batch(batch):
        past, future = batch
        return past, future, None

    def _inputs_from_batch(self, batch):
        past, future, patch_col0 = self.unpack_batch(batch)
        cache_key_past = past
        past = past.to(self.device)
        future = future.to(self.device)
        if patch_col0 is not None:
            patch_col0 = patch_col0.to(self.device)
        return past, future, patch_col0, cache_key_past

    def _opt_target_rows(self) -> Optional[int]:
        if self.univariate_micro_batch is None:
            return None
        return int(self.univariate_micro_batch) * int(self.accum_steps)

    def _sample_window_timesteps(self, n_windows: int) -> torch.Tensor:
        n_steps = int(self.model.config.binary_num_steps)
        if n_steps < 1:
            raise ValueError(f"binary_num_steps must be >= 1, got {n_steps}")
        return torch.randint(0, n_steps, (int(n_windows),), device=self.device)

    def _row_slices_for_past(self, past: torch.Tensor):
        if self.univariate_micro_batch is None:
            yield None
            return
        n_windows, n_variates = int(past.shape[0]), int(past.shape[1])
        for start, end in iter_flat_row_slices(
            n_windows, n_variates, self.univariate_micro_batch,
        ):
            yield torch.arange(start, end, device=self.device)

    def _loss_from_inputs(
        self,
        inputs,
        *,
        loss_mode: str = "combined",
        include_anchor: bool = True,
        scale_for_accumulation: bool,
        token_cache=None,
        univariate_row_index=None,
        t=None,
    ) -> torch.Tensor:
        past, future, patch_col0, cache_key_past = inputs
        cached = token_cache.get(cache_key_past) if token_cache is not None else None
        with amp_context(bool(self.model.config.use_amp)):
            loss = self.model.get_loss(
                past,
                future,
                t=t,
                patch_col0=patch_col0,
                loss_mode=loss_mode,
                include_anchor=include_anchor,
                cross_variate_context=None if cached is None else cached.tokens,
                context_token_variate_ids=None if cached is None else cached.token_variate_ids,
                univariate_row_index=univariate_row_index,
            )
        return loss / self.accum_steps if scale_for_accumulation else loss

    def _loss_from_batch(
        self,
        batch,
        *,
        include_anchor: bool = True,
        scale_for_accumulation: bool,
        token_cache=None,
        univariate_row_index=None,
        t=None,
    ) -> torch.Tensor:
        return self._loss_from_inputs(
            self._inputs_from_batch(batch),
            include_anchor=include_anchor,
            scale_for_accumulation=scale_for_accumulation,
            token_cache=token_cache,
            univariate_row_index=univariate_row_index,
            t=t,
        )

    def _optimizer_step(self) -> None:
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)

    def _backward_slice(
        self,
        inputs,
        *,
        row_index,
        t,
        include_anchor: bool,
        scale: float,
    ) -> Optional[float]:
        """Backward one row-slice; return unscaled mean loss for logging.

        Returns None when the slice has no visible patch-refine GT transitions
        (skip the update rather than apply a silent zero masked BCE).
        """
        past, future, patch_col0, cache_key_past = inputs
        try:
            if (
                self.sequential_anchor_backward
                and bool(self.model.config.use_deterministic_anchor_loss)
                and include_anchor
                and self.model.stage_strategy.name == "patch_refine"
                and not self._has_trainable_context_encoder()
            ):
                with amp_context(bool(self.model.config.use_amp)):
                    prepared = self.model.prepare_patch_refine_loss_inputs(
                        past,
                        future,
                        t=t,
                        patch_col0=patch_col0,
                        cross_variate_context=(self.train_token_cache.get(cache_key_past).tokens
                                                if self.train_token_cache is not None else None),
                        context_token_variate_ids=(self.train_token_cache.get(cache_key_past).token_variate_ids
                                                   if self.train_token_cache is not None else None),
                        univariate_row_index=row_index,
                    )
                    cpu_model_rng_state = torch.get_rng_state()
                    cuda_model_rng_state = (
                        torch.cuda.get_rng_state(self.device)
                        if self.device.type == "cuda"
                        else None
                    )
                    regular_loss = self.model.patch_refine_loss_from_prepared(
                        prepared,
                        loss_mode="regular",
                    )["loss"]
                anchor_weight = 1.0 - float(self.model.config.deterministic_anchor_lambda)
                regular_weight = float(self.model.config.deterministic_anchor_lambda)
                (regular_loss * regular_weight * scale).backward()
                regular_value = float(regular_loss.detach().item())
                del regular_loss

                torch.set_rng_state(cpu_model_rng_state)
                if cuda_model_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_model_rng_state, self.device)
                with amp_context(bool(self.model.config.use_amp)):
                    anchor_loss = self.model.patch_refine_loss_from_prepared(
                        prepared,
                        loss_mode="anchor",
                    )["loss"]
                (anchor_loss * anchor_weight * scale).backward()
                unscaled = regular_weight * regular_value + anchor_weight * float(
                    anchor_loss.detach().item()
                )
                del anchor_loss, prepared
                return unscaled
            if (
                self.sequential_anchor_backward
                and bool(self.model.config.use_deterministic_anchor_loss)
                and include_anchor
            ):
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda"
                    else None
                )
                regular_loss = self._loss_from_inputs(
                    inputs,
                    loss_mode="regular",
                    scale_for_accumulation=False,
                    token_cache=self.train_token_cache,
                    univariate_row_index=row_index,
                    t=t,
                )
                anchor_weight = 1.0 - float(self.model.config.deterministic_anchor_lambda)
                regular_weight = float(self.model.config.deterministic_anchor_lambda)
                (regular_loss * regular_weight * scale).backward()
                regular_value = float(regular_loss.detach().item())
                del regular_loss

                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, self.device)
                anchor_loss = self._loss_from_inputs(
                    inputs,
                    loss_mode="anchor",
                    scale_for_accumulation=False,
                    token_cache=self.train_token_cache,
                    univariate_row_index=row_index,
                    t=t,
                )
                (anchor_loss * anchor_weight * scale).backward()
                unscaled = regular_weight * regular_value + anchor_weight * float(
                    anchor_loss.detach().item()
                )
                del anchor_loss
                return unscaled
            loss = self._loss_from_inputs(
                inputs,
                include_anchor=include_anchor,
                scale_for_accumulation=False,
                token_cache=self.train_token_cache,
                univariate_row_index=row_index,
                t=t,
            )
            (loss * scale).backward()
            unscaled = float(loss.detach().item())
            del loss
            return unscaled

        except NoVisiblePatchTransitions:
            return None

    def _has_trainable_context_encoder(self) -> bool:
        context_encoder = getattr(self.model, "context_encoder", None)
        return context_encoder is not None and any(
            parameter.requires_grad for parameter in context_encoder.parameters()
        )

    def _train_epoch(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        if self.set_training_epoch is not None:
            self.set_training_epoch(self.train_loader, epoch)
        self.model.train()
        if self.set_loader_mode is not None:
            self.set_loader_mode(self.model, self.train_loader, eval_mode=False)
        weighted_loss = 0.0
        n_loss_units = 0
        n_train_batches = len(self.train_loader)
        log_stride = max(1, n_train_batches // 4)
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        target_rows = self._opt_target_rows()
        gpu_u = self.univariate_micro_batch
        rows_in_step = 0
        logical_u_idx = 0
        micro_idx = 0
        use_det = bool(self.model.config.use_deterministic_anchor_loss)
        every_n = self.deterministic_anchor_every_n_batches
        n_skipped_slices = 0
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx == 0 or (batch_idx + 1) % log_stride == 0 or batch_idx + 1 == n_train_batches:
                logger.info(
                    "  [%s] epoch %d/%d train_batch %d/%d",
                    self.log_prefix, epoch + 1, max_epochs, batch_idx + 1, n_train_batches,
                )
            inputs = self._inputs_from_batch(batch)
            past = inputs[0]
            if gpu_u is None:
                include_anchor = (not use_det) or (batch_idx % every_n == 0)
                unscaled = self._backward_slice(
                    inputs,
                    row_index=None,
                    t=None,
                    include_anchor=include_anchor,
                    scale=1.0 / self.accum_steps,
                )
                if unscaled is None:
                    n_skipped_slices += 1
                    continue
                micro_idx += 1
                if micro_idx % self.accum_steps == 0:
                    self._optimizer_step()
                weighted_loss += unscaled
                n_loss_units += 1
            else:
                n_windows, n_variates = int(past.shape[0]), int(past.shape[1])
                n_rows = n_windows * n_variates
                window_t = self._sample_window_timesteps(n_windows)
                cursor = 0
                while cursor < n_rows:
                    remaining_budget = int(target_rows) - rows_in_step
                    take = next_row_take(
                        n_rows - cursor,
                        gpu_u=gpu_u,
                        remaining_budget=remaining_budget,
                        rows_in_step=rows_in_step,
                    )
                    row_index = torch.arange(cursor, cursor + take, device=self.device)
                    include_anchor = (not use_det) or (logical_u_idx % every_n == 0)
                    scale = loss_scale_for_rows(take, int(target_rows))
                    unscaled = self._backward_slice(
                        inputs,
                        row_index=row_index,
                        t=window_t,
                        include_anchor=include_anchor,
                        scale=scale,
                    )
                    cursor += take
                    if unscaled is None:
                        n_skipped_slices += 1
                        continue
                    rows_in_step += take
                    weighted_loss += unscaled * take
                    n_loss_units += take
                    if rows_in_step % gpu_u == 0:
                        logical_u_idx += 1
                    if rows_in_step == int(target_rows):
                        self._optimizer_step()
                        rows_in_step = 0
            del inputs
        if n_skipped_slices:
            logger.warning(
                "  [%s] skipped %d train slice(s) with no visible patch_refine GT transitions",
                self.log_prefix, n_skipped_slices,
            )
        if n_loss_units < 1:
            raise RuntimeError(
                f"{self.log_prefix}: entire train epoch had no visible patch_refine "
                "GT transitions (all slices skipped)"
            )
        if gpu_u is None:
            if self.accum_steps > 1 and micro_idx % self.accum_steps != 0:
                self._optimizer_step()
        elif rows_in_step > 0:
            logger.info(
                "  [%s] dropping %d leftover univariate rows (< target %d)",
                self.log_prefix, rows_in_step, int(target_rows),
            )
            self.optimizer.zero_grad(set_to_none=True)
        return weighted_loss / max(n_loss_units, 1), time.perf_counter() - started

    def _validate_epoch(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        self.model.eval()
        if self.set_loader_mode is not None:
            self.set_loader_mode(self.model, self.val_loader, eval_mode=True)
        weighted_loss = 0.0
        n_rows_seen = 0
        n_skipped_slices = 0
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
                inputs = self._inputs_from_batch(batch)
                past = inputs[0]
                window_t = (
                    self._sample_window_timesteps(int(past.shape[0]))
                    if self.univariate_micro_batch is not None
                    else None
                )
                for row_index in self._row_slices_for_past(past):
                    n_rows = (
                        int(past.shape[0]) * int(past.shape[1])
                        if row_index is None
                        else int(row_index.numel())
                    )
                    try:
                        slice_loss = self._loss_from_inputs(
                            inputs,
                            scale_for_accumulation=False,
                            token_cache=self.val_token_cache,
                            univariate_row_index=row_index,
                            t=window_t,
                        )
                    except NoVisiblePatchTransitions:
                        n_skipped_slices += 1
                        continue
                    weighted_loss += float(slice_loss.item()) * n_rows
                    n_rows_seen += n_rows
        if n_skipped_slices:
            logger.warning(
                "  [%s] skipped %d val slice(s) with no visible patch_refine GT transitions",
                self.log_prefix, n_skipped_slices,
            )
        if n_rows_seen < 1:
            raise RuntimeError(
                f"{self.log_prefix}: entire val epoch had no visible patch_refine "
                "GT transitions (all slices skipped)"
            )
        return weighted_loss / max(n_rows_seen, 1), time.perf_counter() - started

    def fit(
        self,
        *,
        max_epochs: int,
        start_epoch: int = 0,
        initial_best_val: Optional[float] = None,
        initial_best_epoch: int = 0,
        early_stopping: Optional[Callable[[float], bool]] = None,
        on_best: Optional[Callable[[DiffusionEpochMetrics], None]] = None,
        on_epoch_end: Optional[Callable[[DiffusionEpochMetrics], None]] = None,
    ) -> DiffusionTrainingResult:
        best_val = float("inf") if initial_best_val is None else float(initial_best_val)
        best_epoch = 0 if initial_best_val is None else int(initial_best_epoch)
        history: List[DiffusionEpochMetrics] = []
        started = time.perf_counter()
        start_epoch = int(start_epoch)
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {start_epoch}")
        if start_epoch >= int(max_epochs):
            raise ValueError(
                f"start_epoch={start_epoch} >= max_epochs={max_epochs}; nothing to train"
            )
        for epoch in range(start_epoch, max_epochs):
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
    deterministic_anchor_every_n_batches: int = 1,
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
    deterministic_anchor_every_n_batches = int(deterministic_anchor_every_n_batches)
    if deterministic_anchor_every_n_batches < 1:
        raise ValueError("deterministic_anchor_every_n_batches must be >= 1")

    for batch_idx, (past, future) in enumerate(train_loader):
        past, future = past.to(device), future.to(device)
        include_anchor = (
            not bool(model.config.use_deterministic_anchor_loss)
            or batch_idx % deterministic_anchor_every_n_batches == 0
        )
        with amp_context(bool(model.config.use_amp)):
            loss = model.get_loss(
                past,
                future,
                include_anchor=include_anchor,
            ) / accum_steps
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
