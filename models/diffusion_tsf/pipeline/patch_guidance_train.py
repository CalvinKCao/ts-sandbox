"""Patch-decoder guidance stack construction and real-data HP finetune."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import optuna
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.guidance import PatchDecoderGuidance
from models.diffusion_tsf.ordinal_window_norm import ordinal_encode, ranks_to_unit
from models.diffusion_tsf.patch_guidance_stack import PatchGuidanceStack, PatchGuidanceStackConfig
from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)



def _dataset_yields_ordinal_ranks(dataset) -> bool:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return bool(getattr(dataset, "yields_ordinal_ranks", False))


def _itrans_ar_enabled(state: PipelineState, future_len: int) -> bool:
    from models.diffusion_tsf.train_multivariate_pipeline import _itrans_ar_enabled as _impl
    return _impl(state, future_len)


def _sample_itrans_ar_chunk(state: PipelineState, past: torch.Tensor, future: torch.Tensor):
    from models.diffusion_tsf.train_multivariate_pipeline import _sample_itrans_ar_chunk as _impl
    return _impl(state, past, future)


def _promote_trial_ckpt(study, trial_dir: str, trial_filename: str, dest: str) -> None:
    from models.diffusion_tsf.train_multivariate_pipeline import _promote_trial_ckpt as _impl
    return _impl(study, trial_dir, trial_filename, dest)


def dataset_window_lengths(state: PipelineState, dataset_name: str):
    from models.diffusion_tsf.train_multivariate_pipeline import dataset_window_lengths as _impl
    return _impl(state, dataset_name)


def load_dataset(*args, **kwargs):
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset as _impl
    return _impl(*args, **kwargs)

def _patch_guidance_out_len(state: PipelineState) -> int:
    """Native decoder forecast length (dataset horizon, not diffusion AR chunk)."""
    return int(state.forecast_length)


def _patch_guidance_pred_len(state: PipelineState) -> int:
    return int(state.forecast_length)


def _checkpoint_is_patch_guidance(ckpt: dict) -> bool:
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and cfg.get("in_len") and cfg.get("out_len") and cfg.get("patch_size"):
        return True
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        return False
    return any(k.startswith("decoder.") or k.startswith("mixer.") for k in sd)


def create_patch_guidance_stack(
    state: PipelineState,
    num_vars: int,
    *,
    in_len: Optional[int] = None,
    out_len: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> PatchGuidanceStack:
    cfg = PatchGuidanceStackConfig(
        in_len=int(in_len or state.lookback_length),
        out_len=int(out_len or _patch_guidance_out_len(state)),
        patch_size=int(patch_size or state.mmpd_patch_size),
        data_dim=int(num_vars),
    )
    stack = PatchGuidanceStack(cfg)
    stack.set_channel_dropout_drop_frac(float(getattr(state, "channel_dropout_drop_frac", 0.0)))
    return stack


def wrap_patch_guidance(state: PipelineState, stack: PatchGuidanceStack) -> PatchDecoderGuidance:
    ordinal_ladder = state.ordinal_ladder if state.use_ordinal_window_norm else None
    return PatchDecoderGuidance(
        stack,
        chunk_horizon=_patch_guidance_pred_len(state),
        ordinal_ladder=ordinal_ladder,
    )


def load_patch_guidance_from_checkpoint(
    state: PipelineState,
    path: str,
    num_vars: int,
    device: torch.device,
    ckpt: Optional[dict] = None,
) -> PatchGuidanceStack:
    if ckpt is None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt.get("config") or {})
    cfg_dict.setdefault("data_dim", num_vars)
    cfg_dict.setdefault("in_len", state.lookback_length)
    cfg_dict.setdefault("out_len", _patch_guidance_out_len(state))
    cfg_dict.setdefault("patch_size", state.mmpd_patch_size)
    cfg = PatchGuidanceStackConfig(**cfg_dict)
    stack = PatchGuidanceStack(cfg).to(device)
    stack.set_channel_dropout_drop_frac(float(getattr(state, "channel_dropout_drop_frac", 0.0)))
    stack.load_state_dict(ckpt["model_state_dict"], strict=True)
    stack.eval()
    return stack


def _window_norm_past_future(
    state: PipelineState,
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    apply_ood_shift: bool = False,
    data_is_ranked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if state.use_ordinal_window_norm:
        if state.ordinal_ladder is None:
            raise ValueError("state.ordinal_ladder must be set before ordinal encoding")
        if data_is_ranked:
            return past, future
        past_ord, future_ord, _ladder, _ood_shift = ordinal_encode(
            past,
            future,
            ladder=state.ordinal_ladder,
            apply_ood_shift=apply_ood_shift,
            causal_only=state.ordinal_ood_shift_causal_only,
        )
        return past_ord, future_ord
    if not state.use_window_normalization:
        return past, future
    if state.window_norm_center == "last":
        center = past[..., -1:]
    elif state.window_norm_center == "mean":
        center = past.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {state.window_norm_center!r}")
    past_std = past.std(dim=-1, keepdim=True)
    if state.window_norm_low_var_threshold > 0.0:
        std_floor = past_std.clamp_min(state.window_norm_std_floor)
        unit = torch.full_like(past_std, state.window_norm_low_var_unit_std_by_dataset.get(state.dataset, state.window_norm_low_var_unit_std))
        low_var = past_std < state.window_norm_low_var_threshold
        flat = past_std <= state.window_norm_std_floor
        std = torch.where(flat | low_var, unit, std_floor)
    else:
        std = past_std.clamp_min(state.window_norm_std_floor)
    return (past - center) / std, (future - center) / std


def _patch_guidance_batch(
    state: PipelineState,
    past: torch.Tensor,
    future: torch.Tensor,
    device: torch.device,
    *,
    apply_ood_shift: bool = False,
    data_is_ranked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _itrans_ar_enabled(state, future.shape[-1]):
        past, future = _sample_itrans_ar_chunk(state, past, future)
    past = past.to(device)
    future = future.to(device)
    # Univariate loaders can yield (B, T); decoder expects (B, V, T).
    if past.ndim == 2:
        past = past.unsqueeze(1)
        future = future.unsqueeze(1)
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError(
            f"patch guidance expects (B,V,T), got past={tuple(past.shape)} "
            f"future={tuple(future.shape)}"
        )
    past_norm, future_norm = _window_norm_past_future(
        state, past,
        future,
        apply_ood_shift=apply_ood_shift,
        data_is_ranked=data_is_ranked,
    )
    avail = future_norm.shape[-1] - state.lookback_overlap if state.lookback_overlap > 0 else future_norm.shape[-1]
    pred_len = min(_patch_guidance_out_len(state), avail)
    if state.lookback_overlap > 0:
        target = future_norm[..., state.lookback_overlap : state.lookback_overlap + pred_len]
    else:
        target = future_norm[..., :pred_len]
    if state.use_ordinal_window_norm:
        if state.ordinal_ladder is None:
            raise ValueError("state.ordinal_ladder must be set before ordinal patch guidance")
        ladder_past = state.ordinal_ladder.expand_batch(past_norm.shape[0])
        ladder_target = state.ordinal_ladder.expand_batch(target.shape[0])
        past_norm = ranks_to_unit(past_norm, ladder_past)
        target = ranks_to_unit(target, ladder_target)
    return past_norm, target


def train_patch_guidance_epoch(state: PipelineState, stack, loader, optimizer, device, scheduler=None):
    stack.train()
    total_loss = 0.0
    n_batches = 0
    data_is_ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    from models.diffusion_tsf.train_window_aug import set_train_window_aug_epoch

    # Epoch counter lives on the dataset; bump once per call (one epoch).
    ds = loader.dataset
    while hasattr(ds, "dataset") and not hasattr(ds, "set_epoch"):
        ds = ds.dataset
    if hasattr(ds, "set_epoch"):
        set_train_window_aug_epoch(loader, int(getattr(ds, "_epoch", 0)) + 1)
    for past, future in loader:
        past_norm, y_true = _patch_guidance_batch(
            state, past, future, device, data_is_ranked=data_is_ranked,
        )
        optimizer.zero_grad()
        loss = stack.finetune_loss(past_norm, y_true)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def validate_patch_guidance(state: PipelineState, stack, loader, device):
    stack.eval()
    total_loss = 0.0
    n_batches = 0
    data_is_ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    with torch.no_grad():
        for past, future in loader:
            past_norm, y_true = _patch_guidance_batch(
                state, past,
                future,
                device,
                apply_ood_shift=state.use_ordinal_window_norm,
                data_is_ranked=data_is_ranked,
            )
            loss = stack.finetune_loss(past_norm, y_true)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def patch_guidance_hp_objective(
    state: PipelineState,
    trial,
    train_loader,
    val_loader,
    num_vars: int,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    trial_ckpt_dir: Optional[str] = None,
):
    lr = trial.suggest_categorical("learning_rate", state.itrans_paper_lr_grid)
    batch_size = fixed_batch_size if fixed_batch_size is not None else state.itrans_paper_batch_size
    max_epochs = state.patch_guidance_hp_finetune_max_epochs if max_epochs is None else max_epochs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stack = create_patch_guidance_stack(state, num_vars).to(device)

    from models.diffusion_tsf.pipeline.config import training_value
    from models.diffusion_tsf.pipeline.train.diffusion_loop import (
        EpochGroupBatchSampler,
        log_epoch_shard_contract,
        make_grouped_train_loader,
    )

    n_groups_cfg = int(training_value(state, "train_epoch_groups", 1))
    if n_groups_cfg < 1:
        raise ValueError(
            f"training.train_epoch_groups must be >= 1, got {n_groups_cfg!r}"
        )
    raw_max_bytes = training_value(state, "train_epoch_max_bytes", None)
    max_bytes = None if raw_max_bytes is None else int(raw_max_bytes)
    ds_lb, ds_hz = dataset_window_lengths(state, state.dataset)
    train_loader_local, n_groups, window_nbytes, group_nbytes = make_grouped_train_loader(
        train_loader.dataset,
        batch_size=batch_size,
        n_groups=n_groups_cfg,
        seed=int(state.seed) + 31 * int(getattr(trial, "number", 0)),
        max_bytes=max_bytes,
        n_variates=int(num_vars),
        lookback=int(ds_lb),
        horizon=int(ds_hz),
        overlap=int(state.lookback_overlap),
        smoke_test=bool(smoke_test),
    )
    sampler = getattr(train_loader_local, "batch_sampler", None)
    if not isinstance(sampler, EpochGroupBatchSampler):
        raise TypeError(
            "patch-guidance training requires EpochGroupBatchSampler; "
            f"got {type(sampler)}"
        )
    val_bs = min(batch_size, 32)
    val_loader_local = DataLoader(
        val_loader.dataset, batch_size=val_bs, shuffle=False, num_workers=0,
    )
    logger.info(
        "[Patch guidance HP] groups=%d train_n=%d train_batches=%d "
        "window_bytes=%s group_bytes=%s bs=%d",
        n_groups, len(train_loader.dataset), len(train_loader_local),
        window_nbytes, group_nbytes, batch_size,
    )
    log_epoch_shard_contract(
        name="patch_guidance_hp",
        n_groups=n_groups,
        max_epochs=max_epochs if not smoke_test else 1,
        patience=None,
    )

    optimizer = torch.optim.Adam(stack.parameters(), lr=lr)
    epochs = max_epochs if not smoke_test else 1
    best_val_loss = float("inf")
    trial_ckpt_path = None
    if trial_ckpt_dir is not None:
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_ckpt_path = os.path.join(
            trial_ckpt_dir, f"patch_guidance_hp_trial_{trial.number}.pt",
        )

    try:
        for epoch in range(epochs):
            train_loader_local.batch_sampler.set_epoch(epoch)
            train_patch_guidance_epoch(state, stack, train_loader_local, optimizer, device)
            val_loss = validate_patch_guidance(state, stack, val_loader_local, device)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tuned = {"learning_rate": lr, "batch_size": batch_size}
                if trial_ckpt_path is not None:
                    torch.save(
                        {
                            "model_state_dict": stack.state_dict(),
                            "config": stack.config.to_dict(),
                            "best_params": tuned,
                            "val_loss": val_loss,
                            "ordinal_patch_guidance_unit_ranks": bool(state.use_ordinal_window_norm),
                            "patch_guidance_target_space": (
                                "ordinal_unit_rank"
                                if state.use_ordinal_window_norm
                                else "window_normalized"
                            ),
                        },
                        trial_ckpt_path,
                    )
                    trial.set_user_attr("ckpt_path", trial_ckpt_path)
    except torch.OutOfMemoryError:
        logger.warning(
            "[Patch guidance HP] OOM at batch_size=%s; pruning trial %s.",
            batch_size, trial.number,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned()
    return best_val_loss


def run_patch_guidance_finetune_hp_tuning(
    state: PipelineState,
    dataset_name: str,
    variate_indices: List[int],
    n_trials: int,
    device: torch.device,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    subset_id: Optional[str] = None,
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    parallel_workers: int = 1,
) -> Tuple[Dict, Optional[str]]:
    """HP tune patch decoder + mixer on real data (window-norm MSE)."""
    label = subset_id or dataset_name
    n_vars = len(variate_indices)
    logger.info("=" * 60)
    logger.info(
        "Patch guidance finetune HP: %s (%d trials, %d workers)",
        label, n_trials, parallel_workers,
    )
    logger.info("=" * 60)

    train_ds, val_ds, _, norm_stats = load_dataset(
        state, dataset_name, variate_indices,
        stride=train_stride or state.window_stride,
        test_stride=1 if test_stride is None else test_stride,
    )
    from models.diffusion_tsf.pipeline.data_subset import random_window_subset
    subset_meta = state.data_subset_resolved or {}
    train_ds = random_window_subset(
        train_ds,
        subset_meta.get("train_max_windows"),
        int(state.seed) + 17,
        label="patch_guidance/train",
    )
    val_ds = random_window_subset(
        val_ds,
        subset_meta.get("val_max_windows"),
        int(state.seed) + 29,
        label="patch_guidance/val",
    )
    from models.diffusion_tsf.train_window_aug import maybe_wrap_train_window_aug

    aug_cfg = state.train_window_aug
    train_ds = maybe_wrap_train_window_aug(
        train_ds,
        enabled=bool(aug_cfg.get("enabled", False)),
        apply_prob=float(aug_cfg.get("apply_prob", 0.5)),
        seed=state.seed,
        ladder=norm_stats.get("ordinal_ladder"),
        acf_threshold=float(aug_cfg.get("acf_threshold", 0.35)),
        excluded_names=aug_cfg.get("exclude_names", ()),
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = state.itrans_paper_batch_size
    train_loader = DataLoader(
        train_ds, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=not smoke_test,
    )
    val_loader = DataLoader(val_ds, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or state.checkpoint_dir
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def objective(trial):
            return patch_guidance_hp_objective(
                state, trial, train_loader, val_loader, n_vars, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=state.patch_guidance_hp_finetune_max_epochs,
                trial_ckpt_dir=trial_dir,
            )

        return objective

    study = run_optuna_study(
        study_name=f"patch-guidance-ft-{label}",
        checkpoint_dir=trial_dir,
        n_trials=n_trials,
        parallel_workers=parallel_workers,
        direction="minimize",
        objective_builder=objective_builder,
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        show_progress_bar=not smoke_test,
        sampler_seed=42,
    )

    best_params = dict(study.best_params)
    best_params["batch_size"] = train_bs
    logger.info(
        "Best patch guidance FT params for %s: lr=%.2e → val_loss=%.4f",
        label, best_params["learning_rate"], study.best_value,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"{label}_patch_guidance_hp_best.pt")
        _promote_trial_ckpt(
            study, trial_dir, "patch_guidance_hp_trial_{trial}.pt", ckpt_path,
        )
        logger.info("  Saved best patch guidance FT HP model → %s", ckpt_path)
    return best_params, ckpt_path

