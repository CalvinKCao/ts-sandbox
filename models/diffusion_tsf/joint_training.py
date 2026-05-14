"""End-to-end joint training of iTransformer + diffusion backbone.

This module collapses the legacy 4-phase pipeline
(1A iTrans HP -> 1B Diff HP -> 2A iTrans HP finetune -> 2B Diff HP finetune)
into 2 phases by training the iTransformer guidance jointly with the diffusion
backbone in both pretrain and finetune. See ``architecture.md`` (e2e branch).

Design points:
  * iTrans is unfrozen (``DiffusionTSFConfig.e2e_joint_training=True`` +
    ``iTransformerGuidance(freeze=False)``) so gradients flow back from the
    diffusion noise/EMD loss through the cross-attention token path.
  * Auxiliary forecast MSE on the iTrans horizon prediction anchors the
    iTransformer as a real forecaster — without it, joint training risks
    drifting iTrans into a pure denoising-helper role.
  * Two ghost-image variants (see ``joint_use_ghost_image`` in config):
      - B: keep the ghost image as a U-Net input channel, but ``.detach()`` it
        so the (non-differentiable) ``encode_to_2d`` does not block gradients.
      - C: drop the ghost image; tokens-only conditioning.
  * Optional iTrans-only warmup phase (``itrans_warmup_epochs``): early epochs
    run only the iTransformer + aux MSE loss; the diffusion backbone is
    skipped entirely. Gives the iTransformer a meaningful starting point
    before the diffusion loss starts flowing back into it.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter group helpers
# ---------------------------------------------------------------------------


def _split_param_groups(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split DiffusionTSF parameters into (itrans_params, diffusion_params).

    iTransformer-side: ``guidance_model.model.*`` (the inner iTransformer net)
        plus ``context_encoder.*`` (iTransformerTokenAdapter — projects encoder
        tokens to the cross-attention dim). These are the things the
        auxiliary forecast loss + cross-attn-token gradient should update.

    Diffusion-side: everything else (noise_predictor backbone, blur, etc.).
    """
    itrans_params: List[nn.Parameter] = []
    diff_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("guidance_model.model.") or name.startswith("context_encoder."):
            itrans_params.append(p)
        else:
            diff_params.append(p)
    return itrans_params, diff_params


def _count(params: List[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Loop config
# ---------------------------------------------------------------------------


@dataclass
class JointTrainConfig:
    """Hyperparameters for a single joint training run."""

    diffusion_lr: float = 2e-4
    itrans_lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_epochs: int = 20
    warmup_epochs: int = 1
    patience: int = 5            # early-stop on full-joint val loss
    log_every_n_batches: int = 0  # 0 = only end-of-epoch summary
    use_amp: bool = False         # bfloat16 amp toggle


@dataclass
class JointTrainResult:
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    interrupted: bool = False


# ---------------------------------------------------------------------------
# Single-step helpers
# ---------------------------------------------------------------------------


def _amp_autocast(use_amp: bool, device: torch.device):
    if not use_amp:
        return nullcontext()
    # bfloat16 matches the existing pipeline_config default.
    dtype = torch.bfloat16
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.amp.autocast("cpu", dtype=dtype)


def _aux_forecast_loss(
    model: nn.Module, past: torch.Tensor, future: torch.Tensor
) -> torch.Tensor:
    """Compute the auxiliary MSE forecast loss without running the diffusion backbone.

    Used during the iTrans-only warmup phase. Mirrors what
    DiffusionTSF._forward_factorized does for the aux term but skips
    ``encode_to_2d``, the noise predictor, the EMD term, etc.
    """
    cfg = model.config
    past_norm, future_norm, stats = model._normalize_sequence(past, future)
    mean, std = stats
    K = cfg.lookback_overlap
    W_fut = future_norm.shape[-1]
    H = W_fut - K
    coarse = model.guidance_model.get_forecast(past, H)  # (B, V, H)
    coarse_norm = (coarse - mean) / std
    target = future_norm[..., K:] if K > 0 else future_norm
    ms = cfg.max_scale
    return F.mse_loss(coarse_norm.clamp(-ms, ms), target.clamp(-ms, ms))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def train_joint_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    train_cfg: JointTrainConfig,
    *,
    device: torch.device,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    state_dict_to_cpu: bool = True,
) -> JointTrainResult:
    """Joint pretrain/finetune loop with iTrans-only warmup and dual LRs.

    Args:
        model: DiffusionTSF instance with ``e2e_joint_training=True`` in its
            config and a trainable ``iTransformerGuidance(freeze=False)``.
            DDP/parallel wrappers should be applied before this call (the
            loop uses ``model.parameters()`` directly).
        train_loader / val_loader: yield ``(past, future)`` tensors. ``val_loader``
            may be ``None`` (then we just train for ``num_epochs`` and return
            the final state; no early stopping in that case).
        train_cfg: HPs for this run.
        device: target device.
        on_epoch_end: optional callback for wandb / external logging. Receives
            (epoch_idx_zero_based, metrics_dict).
        state_dict_to_cpu: store the best state on CPU to free GPU memory.
    """
    if not getattr(model.config, "e2e_joint_training", False):
        raise ValueError(
            "train_joint_phase requires DiffusionTSFConfig.e2e_joint_training=True"
        )
    if model.guidance_model is None or not hasattr(model.guidance_model, "get_forecast"):
        raise ValueError(
            "train_joint_phase requires a guidance model with .get_forecast()"
        )

    itrans_params, diff_params = _split_param_groups(model)
    logger.info(
        "Joint training param groups: iTrans=%.2fM, diffusion=%.2fM",
        _count(itrans_params) / 1e6,
        _count(diff_params) / 1e6,
    )
    if not itrans_params:
        raise RuntimeError(
            "No trainable iTransformer parameters found. Did you forget "
            "iTransformerGuidance(freeze=False)?"
        )
    if not diff_params:
        raise RuntimeError("No trainable diffusion parameters found.")

    optimizer = torch.optim.AdamW(
        [
            {"params": itrans_params, "lr": train_cfg.itrans_lr, "name": "itrans"},
            {"params": diff_params, "lr": train_cfg.diffusion_lr, "name": "diffusion"},
        ],
        weight_decay=train_cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(train_cfg.num_epochs, 1),
        eta_min=min(train_cfg.diffusion_lr, train_cfg.itrans_lr) * 0.01,
    )

    result = JointTrainResult()
    epochs_since_best = 0

    try:
        for epoch in range(train_cfg.num_epochs):
            is_warmup = epoch < train_cfg.warmup_epochs
            t0 = time.time()
            metrics = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                train_cfg=train_cfg,
                device=device,
                is_warmup=is_warmup,
            )
            train_time = time.time() - t0

            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = _validate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    train_cfg=train_cfg,
                    is_warmup=is_warmup,
                )

            sched.step()
            epoch_metrics: Dict[str, float] = {
                "epoch": epoch + 1,
                "phase": "warmup" if is_warmup else "joint",
                "train_time_s": train_time,
                "lr_itrans": optimizer.param_groups[0]["lr"],
                "lr_diffusion": optimizer.param_groups[1]["lr"],
                **{f"train_{k}": v for k, v in metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            result.history.append(epoch_metrics)

            tag = "warmup" if is_warmup else "joint"
            val_total = val_metrics.get("loss", float("nan"))
            logger.info(
                "[%s] epoch %d/%d | train loss %.4f (noise %.4f, aux %.4f) | "
                "val %.4f | lr d=%.2e i=%.2e | %.1fs",
                tag, epoch + 1, train_cfg.num_epochs,
                metrics.get("loss", float("nan")),
                metrics.get("noise_loss", 0.0),
                metrics.get("aux_forecast_loss", 0.0),
                val_total,
                optimizer.param_groups[1]["lr"],
                optimizer.param_groups[0]["lr"],
                train_time,
            )

            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_metrics)

            # Early stopping only kicks in once we have full-joint val numbers
            # (warmup-only val isn't comparable to joint-phase val).
            if val_loader is not None and not is_warmup:
                if val_total < result.best_val_loss:
                    result.best_val_loss = val_total
                    result.best_epoch = epoch + 1
                    sd = model.state_dict()
                    if state_dict_to_cpu:
                        sd = {k: v.detach().cpu().clone() for k, v in sd.items()}
                    result.best_state_dict = sd
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                    if epochs_since_best >= train_cfg.patience:
                        logger.info("Early stopping at epoch %d (patience=%d)",
                                    epoch + 1, train_cfg.patience)
                        break
    except KeyboardInterrupt:
        logger.warning("Joint training interrupted by user")
        result.interrupted = True

    if result.best_state_dict is None:
        # No improvement ever happened (e.g. no val loader, or every epoch was
        # warmup). Stash the final state so callers always get something back.
        sd = model.state_dict()
        if state_dict_to_cpu:
            sd = {k: v.detach().cpu().clone() for k, v in sd.items()}
        result.best_state_dict = sd
        result.best_val_loss = float("nan")
        result.best_epoch = train_cfg.num_epochs

    return result


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_cfg: JointTrainConfig,
    device: torch.device,
    is_warmup: bool,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n_batches = 0

    for batch_idx, (past, future) in enumerate(loader):
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_autocast(train_cfg.use_amp, device):
            if is_warmup:
                aux = _aux_forecast_loss(model, past, future)
                loss = aux
                # Match the dict keys produced by full forward so logging is uniform.
                step = {
                    "loss": loss.detach(),
                    "aux_forecast_loss": aux.detach(),
                    "noise_loss": torch.zeros((), device=device),
                    "emd_loss": torch.zeros((), device=device),
                }
            else:
                out = model(past, future)
                loss = out["loss"]
                step = {
                    "loss": loss.detach(),
                    "noise_loss": out["noise_loss"].detach(),
                    "emd_loss": out["emd_loss"].detach(),
                    "aux_forecast_loss": out.get(
                        "aux_forecast_loss", torch.zeros((), device=device)
                    ).detach(),
                }

        loss.backward()
        if train_cfg.grad_clip and train_cfg.grad_clip > 0:
            # Clip diffusion and iTrans groups separately so a huge iTrans
            # gradient does not eat into the diffusion gradient budget.
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], train_cfg.grad_clip)
        optimizer.step()

        for k, v in step.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        n_batches += 1

        if train_cfg.log_every_n_batches > 0 and (batch_idx + 1) % train_cfg.log_every_n_batches == 0:
            logger.info(
                "  batch %d: loss=%.4f", batch_idx + 1, float(loss.detach().item())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Optuna search wrapper
# ---------------------------------------------------------------------------


@dataclass
class JointSearchConfig:
    """Optuna search space and budget for joint training."""

    n_trials: int = 4
    diffusion_lr_min: float = 5e-5
    diffusion_lr_max: float = 5e-4
    itrans_lr_min: float = 5e-5
    itrans_lr_max: float = 5e-4
    num_epochs: int = 15
    warmup_epochs: int = 1
    patience: int = 5
    grad_clip: float = 1.0
    use_amp: bool = False
    seed: int = 42
    median_pruner_warmup: int = 2
    state_dict_to_cpu: bool = True


def optuna_search_joint_phase(
    model_factory: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    search_cfg: JointSearchConfig,
    *,
    device: torch.device,
    study_name: str = "joint_phase",
    on_trial_end: Optional[Callable[[Any, Dict[str, float]], None]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]], "optuna.Study"]:
    """Run an Optuna study over (diffusion_lr, itrans_lr) for joint training.

    Each trial:
        * builds a fresh model from ``model_factory()``;
        * samples ``diffusion_lr`` and ``itrans_lr`` log-uniformly;
        * runs ``train_joint_phase`` with the sampled LRs;
        * reports best joint-phase val loss to Optuna for pruning;
        * if its best is the global best, captures the model state_dict for
          downstream consumers (so we don't have to retrain).

    Returns (best_params, best_state_dict, study). ``best_state_dict`` may be
    ``None`` if no trial ever produced a joint-phase val loss (e.g. all trials
    spent only on warmup, which would indicate a misconfiguration).
    """
    import optuna  # local import — keeps module importable without optuna
    from optuna.samplers import TPESampler

    if val_loader is None:
        raise ValueError(
            "optuna_search_joint_phase requires a val_loader (need val loss to score trials)"
        )

    shared = {
        "best_val_loss": float("inf"),
        "best_state_dict": None,
        "best_params": None,
    }

    def objective(trial: "optuna.Trial") -> float:
        diffusion_lr = trial.suggest_float(
            "diffusion_lr", search_cfg.diffusion_lr_min, search_cfg.diffusion_lr_max, log=True,
        )
        itrans_lr = trial.suggest_float(
            "itrans_lr", search_cfg.itrans_lr_min, search_cfg.itrans_lr_max, log=True,
        )
        logger.info(
            "[%s] Trial %d/%d: diffusion_lr=%.2e, itrans_lr=%.2e",
            study_name, trial.number + 1, search_cfg.n_trials, diffusion_lr, itrans_lr,
        )

        model = model_factory().to(device)
        tcfg = JointTrainConfig(
            diffusion_lr=diffusion_lr,
            itrans_lr=itrans_lr,
            num_epochs=search_cfg.num_epochs,
            warmup_epochs=search_cfg.warmup_epochs,
            patience=search_cfg.patience,
            grad_clip=search_cfg.grad_clip,
            use_amp=search_cfg.use_amp,
        )

        def _report_for_pruning(epoch_idx: int, metrics: Dict[str, float]) -> None:
            # Report joint-phase val loss for median pruning. Warmup epochs
            # carry no comparable signal; skip them.
            if metrics.get("phase") == "joint" and "val_loss" in metrics:
                trial.report(metrics["val_loss"], step=epoch_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        result = train_joint_phase(
            model,
            train_loader,
            val_loader,
            tcfg,
            device=device,
            on_epoch_end=_report_for_pruning,
            state_dict_to_cpu=search_cfg.state_dict_to_cpu,
        )

        # Cross-trial best-state capture
        if result.best_val_loss < shared["best_val_loss"]:
            shared["best_val_loss"] = result.best_val_loss
            shared["best_state_dict"] = result.best_state_dict
            shared["best_params"] = {
                "diffusion_lr": diffusion_lr,
                "itrans_lr": itrans_lr,
                "best_epoch": result.best_epoch,
            }

        if on_trial_end is not None:
            on_trial_end(trial, {
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
            })

        # Free the trial's model before the next factory() call.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if result.best_val_loss == float("inf"):
            # Trial only ran warmup (no joint-phase val numbers). Return a
            # large penalty so Optuna can still rank it.
            return 1e9
        return result.best_val_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=search_cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=search_cfg.median_pruner_warmup),
    )
    study.optimize(objective, n_trials=search_cfg.n_trials, gc_after_trial=True)

    return shared["best_params"] or {}, shared["best_state_dict"], study


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_cfg: JointTrainConfig,
    is_warmup: bool,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n_batches = 0

    for past, future in loader:
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        with _amp_autocast(train_cfg.use_amp, device):
            if is_warmup:
                aux = _aux_forecast_loss(model, past, future)
                step = {
                    "loss": aux,
                    "aux_forecast_loss": aux,
                    "noise_loss": torch.zeros((), device=device),
                    "emd_loss": torch.zeros((), device=device),
                }
            else:
                out = model(past, future)
                step = {
                    "loss": out["loss"],
                    "noise_loss": out["noise_loss"],
                    "emd_loss": out["emd_loss"],
                    "aux_forecast_loss": out.get(
                        "aux_forecast_loss", torch.zeros((), device=device)
                    ),
                }
        for k, v in step.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}
