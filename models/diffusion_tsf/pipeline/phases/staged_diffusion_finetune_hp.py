"""HP tuning for staged coarse/fine diffusion models; best trial checkpoint is final."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from optuna.exceptions import TrialPruned
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _stage_pretrain_ckpt,
    discover_dataset_run_ckpt_dir,
    patch_stage_globals,
)

logger = logging.getLogger(__name__)


TUNED_MODEL_KEYS = (
    "max_scale",
    "binary_noise_schedule",
    "prediction_target",
    "loss_weighting",
    "min_snr_gamma",
    "d3pm_transition_max",
    "dit_dropout",
)


def _stage_subset_dir(state: PipelineState, stage: str) -> str:
    subset_id = state.subset_id or state.dataset
    return os.path.join(state.checkpoint_dir, subset_id, stage)


def _stage_best_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_subset_dir(state, stage), "best.pt")


def _model_kwargs_from_tuned(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    return {key: params[key] for key in TUNED_MODEL_KEYS if key in params}


def _load_reused_stage_params(
    state: PipelineState,
    *,
    stage: str,
    subset_id: str,
    source_config: str,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    source_dir = discover_dataset_run_ckpt_dir(state, source_config)
    meta_path = os.path.join(source_dir, subset_id, stage, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing {stage} metadata for reuse: {meta_path} "
            f"(from *-{state.dataset}-{source_config})"
        )
    with open(meta_path, encoding="utf-8") as f:
        source_meta = json.load(f)
    params = dict(source_meta.get("tuned_params") or {})
    if not params:
        raise ValueError(f"No tuned_params in {meta_path}")
    policy_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    old_ms = params.get("max_scale")
    params["max_scale"] = policy_ms
    params.setdefault("min_snr_gamma", 5.0)
    return params, source_dir, {**source_meta, "reused_max_scale_previous": old_ms}


class _Ema:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, avg in self.shadow.items():
            avg.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

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


def _suggest_staged_params(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    search_space: str = "default",
) -> Dict[str, Any]:
    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    if getattr(state, "max_scale_tuning", False):
        rng = getattr(state, "max_scale_tuning_range", [2.5, 14.0])
        ms = trial.suggest_float("max_scale", float(rng[0]), float(rng[1]))
    else:
        ms = base_ms

    if search_space == "lr_only":
        from models.diffusion_tsf.train_multivariate_pipeline import (
            FINETUNE_HP_LR_MAX,
            FINETUNE_HP_LR_MIN,
        )

        if FINETUNE_HP_LR_MIN == FINETUNE_HP_LR_MAX:
            lr = float(FINETUNE_HP_LR_MIN)
        else:
            lr = trial.suggest_float(
                "learning_rate", FINETUNE_HP_LR_MIN, FINETUNE_HP_LR_MAX, log=True
            )
        return {
            "learning_rate": lr,
            "batch_size": max(1, max_batch_size),
            "ema_decay": float(state.extra.get("diffusion_ema_decay", 0.0)),
            "binary_noise_schedule": state.binary_noise_schedule,
            "loss_weighting": state.loss_weighting,
            "min_snr_gamma": float(state.min_snr_gamma),
            "prediction_target": state.prediction_target,
            "max_scale": ms,
        }

    if smoke_test:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
            "batch_size": min(max(1, max_batch_size), 2),
            "ema_decay": 0.0,
            "binary_noise_schedule": state.binary_noise_schedule,
            "loss_weighting": state.loss_weighting,
            "min_snr_gamma": float(state.min_snr_gamma),
            "prediction_target": state.prediction_target,
            "max_scale": ms,
        }

    batch_grid = [b for b in (4, 8, 16, 32, 48, 64, 96, 128) if b <= max_batch_size]
    if not batch_grid:
        batch_grid = [max(1, max_batch_size)]
    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-6, 8e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", batch_grid),
        "ema_decay": trial.suggest_categorical("ema_decay", [0.0, 0.99, 0.995, 0.999]),
        "binary_noise_schedule": trial.suggest_categorical(
            "binary_noise_schedule", ["linear", "cosine"]
        ),
        "loss_weighting": trial.suggest_categorical("loss_weighting", ["none", "min_snr"]),
        "prediction_target": trial.suggest_categorical("prediction_target", ["x0", "epsilon"]),
        "max_scale": ms,
    }
    params["min_snr_gamma"] = (
        trial.suggest_float("min_snr_gamma", 1.0, 10.0, log=True)
        if params["loss_weighting"] == "min_snr"
        else 5.0
    )
    return params


def _suggest_ordinal_d3pm_params(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
) -> Dict[str, Any]:
    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    batch_size = min(max(1, max_batch_size), 2) if smoke_test else max(1, max_batch_size)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "d3pm_transition_max": trial.suggest_float("d3pm_transition_max", 0.1, 0.4),
        "dit_dropout": trial.suggest_categorical("dit_dropout", [0.0, 0.1]),
        "batch_size": batch_size,
        "max_scale": base_ms,
        "prediction_target": "x0",
        "loss_weighting": "none",
        "min_snr_gamma": float(state.min_snr_gamma),
    }


class _BaseStagedDiffusionFinetuneHPPhase(PipelinePhase):
    stage = ""

    def should_skip(self, state: PipelineState) -> bool:
        if self.stage == "finer" and not getattr(state, "use_triple_scale", False):
            logger.info("  [%s] skipping: use_triple_scale=False", self.name)
            return True
        best_pt = _stage_best_ckpt(state, self.stage)
        meta = os.path.join(_stage_subset_dir(state, self.stage), "metadata.json")
        if os.path.exists(best_pt) and os.path.exists(meta):
            logger.info("  [%s] cached: %s", self.name, best_pt)
            params = None
            try:
                with open(meta) as f:
                    params = json.load(f).get("tuned_params")
            except Exception as e:
                logger.warning("Failed to load tuned params from %s: %s", meta, e)
            if self.stage == "coarse":
                state.diffusion_coarse_finetune_ckpt = best_pt
                state.coarse_finetune_best_params = params
            elif self.stage == "fine":
                state.diffusion_fine_finetune_ckpt = best_pt
                state.fine_finetune_best_params = params
            else:
                state.diffusion_finer_finetune_ckpt = best_pt
                state.finer_finetune_best_params = params
            return True
        return False

    def _pretrained_ckpt(self, state: PipelineState) -> str:
        attr = {
            "coarse": state.diffusion_coarse_pretrain_ckpt,
            "fine": state.diffusion_fine_pretrain_ckpt,
            "finer": state.diffusion_finer_pretrain_ckpt,
        }[self.stage]
        candidates = [
            self.get("pretrained_ckpt"),
            attr,
            _stage_pretrain_ckpt(state, self.stage),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{self.name} requires a staged {self.stage} pretrain checkpoint. "
            f"Expected one of: {', '.join(str(p) for p in candidates if p)}"
        )

    def _build_model(
        self,
        *,
        state: PipelineState,
        n_iv: int,
        itrans_guidance,
        device: torch.device,
        params: Dict[str, Any],
    ):
        from models.diffusion_tsf.train_multivariate_pipeline import (
            anchor_kwargs_from_params,
            create_diffusion_model,
            dataset_window_lengths,
        )

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        model_kwargs = anchor_kwargs_from_params(params)
        model_kwargs.update(_model_kwargs_from_tuned(params))
        return create_diffusion_model(
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=self.stage,
            use_guidance_channel=state.use_guidance_channel,
            **model_kwargs,
        ).to(device)

    def _train_once(
        self,
        *,
        state: PipelineState,
        train_ds,
        val_ds,
        params: Dict[str, Any],
        pretrained_path: str,
        itrans_checkpoint: str,
        device: torch.device,
        variate_indices,
        ckpt_path: Optional[str],
        max_epochs: int,
        patience: int,
        trial=None,
    ) -> Tuple[float, int]:
        from models.diffusion_tsf.guidance import iTransformerGuidance
        from models.diffusion_tsf.train_multivariate_pipeline import (
            EarlyStopping,
            amp_context,
            load_diffusion_state_keep_attached_guidance,
            load_itransformer_from_checkpoint,
            save_checkpoint,
            unwrap_model,
        )

        n_iv = len(variate_indices)
        batch_size = int(params["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, n_iv, device)
        itrans_guidance = iTransformerGuidance(itrans_model)
        model = self._build_model(
            state=state,
            n_iv=n_iv,
            itrans_guidance=itrans_guidance,
            device=device,
            params=params,
        )
        try:
            ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
            load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
            
            lr_scheduler_type = getattr(state, "lr_scheduler_type", "none")
            warmup_epochs = getattr(state, "lr_warmup_epochs", 0)
            warmup_epochs = min(warmup_epochs, max(0, max_epochs - 1))
            
            scheduler = None
            if lr_scheduler_type == "cosine":
                if warmup_epochs > 0:
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[
                            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=float(params["learning_rate"]) * 0.01)
                        ],
                        milestones=[warmup_epochs]
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(params["learning_rate"]) * 0.01)
            elif lr_scheduler_type == "linear":
                if warmup_epochs > 0:
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[
                            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=max_epochs - warmup_epochs)
                        ],
                        milestones=[warmup_epochs]
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=max_epochs)

            early_stop = EarlyStopping(patience=patience)
            ema = _Ema(model, float(params.get("ema_decay", 0.0))) if params.get("ema_decay", 0.0) else None
            best_val = float("inf")
            best_epoch = 0

            for epoch in range(max_epochs):
                model.train()
                train_loss = 0.0
                n_train = 0
                for past, future in train_loader:
                    past, future = past.to(device), future.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_context():
                        loss = model.get_loss(past, future)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if ema is not None:
                        ema.update(model)
                    train_loss += float(loss.item())
                    n_train += 1

                if scheduler is not None:
                    scheduler.step()

                backup = ema.swap_in(model) if ema is not None else None
                model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for past, future in val_loader:
                        past, future = past.to(device), future.to(device)
                        with amp_context():
                            loss = model.get_loss(past, future)
                        val_loss += float(loss.item())
                        n_val += 1
                val_loss /= max(n_val, 1)

                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch + 1
                    config = {
                        "tuned_params": dict(params),
                        "diffusion_stage": self.stage,
                        "best_epoch": best_epoch,
                    }
                    if ckpt_path:
                        save_checkpoint(
                            unwrap_model(model),
                            optimizer,
                            epoch,
                            train_loss / max(n_train, 1),
                            val_loss,
                            config,
                            ckpt_path,
                        )
                if backup is not None:
                    ema.restore(model, backup)

                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        raise TrialPruned()
                if early_stop(val_loss):
                    break

            return best_val, best_epoch
        finally:
            del model, itrans_model, itrans_guidance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            diffusion_probe_max_candidate,
            generate_dataset_job,
            load_dataset,
            load_itransformer_from_checkpoint,
            select_diffusion_batch_size,
        )
        from models.diffusion_tsf.guidance import iTransformerGuidance
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_stage_globals(pipeline_mod, state, self.stage, honor_dataset_windows=True)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        ft_itrans_ckpt = state.itrans_finetune_ckpt
        if not ft_itrans_ckpt or not os.path.exists(ft_itrans_ckpt):
            raise RuntimeError(f"{self.name} requires finetuned iTransformer, got: {ft_itrans_ckpt}")
        if self.stage == "fine" and not state.diffusion_coarse_finetune_ckpt:
            raise RuntimeError("fine staged tuning requires completed coarse best model first")
        if self.stage == "finer":
            if not state.use_triple_scale:
                raise RuntimeError("finer staged tuning requires use_triple_scale=True")
            if not state.diffusion_fine_finetune_ckpt:
                raise RuntimeError("finer staged tuning requires completed fine best model first")
        diff_ckpt = self._pretrained_ckpt(state)

        device = state.resolve_device()
        n_iv = len(variate_indices)
        train_ds, val_ds, _, norm_stats = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
        )
        if state.smoke_test:
            train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
        logger.info(
            "  [%s] train/val windows=%d/%d",
            self.name, len(train_ds), len(val_ds),
        )

        batch_probe_ds = train_ds
        ft_itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, n_iv, device)
        ft_itrans_guidance = iTransformerGuidance(ft_itrans_model)
        max_batch = select_diffusion_batch_size(
            phase_name=f"{self.stage.title()} Diff FT ({subset_id})",
            dataset=batch_probe_ds,
            device=device,
            itrans_guidance=ft_itrans_guidance,
            max_candidate=diffusion_probe_max_candidate(n_iv, state.smoke_test),
            smoke_test=state.smoke_test,
        )
        del ft_itrans_model, ft_itrans_guidance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_trials = int(self.require("n_trials"))
        if state.smoke_test:
            n_trials = 1
        max_epochs = int(self.require("max_epochs"))
        patience = int(self.require("patience"))
        search_space = str(self.require("search_space")).lower()
        if search_space not in {"default", "lr_only", "ordinal_d3pm"}:
            raise ValueError(f"Unknown staged diffusion search_space={search_space!r}")
        if state.smoke_test:
            max_epochs = patience = 1

        subset_dir = _stage_subset_dir(state, self.stage)
        os.makedirs(subset_dir, exist_ok=True)
        final_ckpt = _stage_best_ckpt(state, self.stage)

        reuse_from = self.get("reuse_tuned_params_from")
        reuse_meta: Dict[str, Any] = {}
        hp_best_val_loss: Optional[float] = None
        best_trial_num = -1
        final_val = float("nan")
        final_epoch = 0

        if reuse_from:
            source_dir = discover_dataset_run_ckpt_dir(state, str(reuse_from))
            src_best = os.path.join(source_dir, subset_id, self.stage, "best.pt")
            src_meta = os.path.join(source_dir, subset_id, self.stage, "metadata.json")
            if not os.path.exists(src_best):
                raise FileNotFoundError(f"Missing reused staged checkpoint: {src_best}")
            if not os.path.exists(final_ckpt):
                import shutil
                shutil.copy2(src_best, final_ckpt)
            if os.path.exists(src_meta):
                with open(src_meta, encoding="utf-8") as f:
                    reuse_meta = json.load(f)
            best_params, _, reuse_meta = _load_reused_stage_params(
                state, stage=self.stage, subset_id=subset_id, source_config=str(reuse_from),
            )
            tuned_bs = int(best_params.get("batch_size", max_batch))
            best_params["batch_size"] = min(tuned_bs, max_batch)
            hp_best_val_loss = float(
                reuse_meta.get("best_val_loss")
                or reuse_meta.get("hp_best_val_loss")
                or float("nan")
            )
            final_val = hp_best_val_loss
            final_epoch = int(reuse_meta.get("best_epoch", 0))
            logger.info("  [%s] reused %s from %s", self.name, self.stage, source_dir)
        else:
            from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

            phase = self

            def objective_builder(_worker_id: int):
                dev = state.resolve_device()

                def objective(trial):
                    if search_space == "ordinal_d3pm":
                        params = _suggest_ordinal_d3pm_params(
                            trial, state, max_batch, state.smoke_test,
                        )
                    else:
                        params = _suggest_staged_params(
                            trial, state, max_batch, state.smoke_test, search_space=search_space,
                        )
                    trial.set_user_attr("full_params", dict(params))
                    trial_ckpt = os.path.join(
                        subset_dir, f"_diff_ft_trial_{trial.number}_best.pt",
                    )
                    try:
                        best_val, best_ep = phase._train_once(
                            state=state,
                            train_ds=train_ds,
                            val_ds=val_ds,
                            params=params,
                            pretrained_path=diff_ckpt,
                            itrans_checkpoint=ft_itrans_ckpt,
                            device=dev,
                            variate_indices=variate_indices,
                            ckpt_path=trial_ckpt,
                            max_epochs=max_epochs,
                            patience=patience,
                            trial=trial,
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            "  [%s] trial %d OOM (batch=%s), pruning",
                            phase.name, trial.number, params.get("batch_size"),
                        )
                        raise TrialPruned() from None
                    trial.set_user_attr("best_epoch", best_ep)
                    return best_val

                return objective

            study = run_optuna_study(
                study_name=f"{state.experiment_name}-{self.stage}-hp",
                checkpoint_dir=subset_dir,
                n_trials=n_trials,
                parallel_workers=state.parallel_optuna_workers,
                direction="minimize",
                objective_builder=objective_builder,
                sampler=TPESampler(seed=state.seed, multivariate=True, group=True),
                pruner=HyperbandPruner(
                    min_resource=1, max_resource=max_epochs, reduction_factor=3,
                ),
                sampler_seed=state.seed,
            )
            try:
                best_trial = study.best_trial
            except ValueError as e:
                raise RuntimeError(
                    f"All {self.stage} diffusion HP trials failed for {subset_id}"
                ) from e

            best_params = dict(best_trial.user_attrs.get("full_params") or best_trial.params)
            best_params.setdefault("min_snr_gamma", 5.0)
            best_params.setdefault(
                "max_scale",
                float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
            )
            hp_best_val_loss = float(study.best_value)
            best_trial_num = int(best_trial.number)
            final_epoch = int(best_trial.user_attrs.get("best_epoch", 0))

            import shutil
            src = os.path.join(subset_dir, f"_diff_ft_trial_{best_trial_num}_best.pt")
            if not os.path.exists(src):
                raise RuntimeError(f"Best trial checkpoint missing: {src}")
            shutil.copy2(src, final_ckpt)
            final_val = hp_best_val_loss

            for fn in os.listdir(subset_dir):
                if fn.startswith("_diff_ft_trial_") and fn.endswith("_best.pt"):
                    try:
                        os.remove(os.path.join(subset_dir, fn))
                    except OSError:
                        pass

        meta_out: Dict[str, Any] = {
            "subset_id": subset_id,
            "dataset_name": state.dataset,
            "variate_indices": variate_indices,
            "data_subset": subset_meta,
            "norm_mean": norm_stats["mean"].tolist(),
            "norm_std": norm_stats["std"].tolist(),
            "tuned_params": best_params,
            "best_trial": best_trial_num,
            "hp_best_val_loss": hp_best_val_loss,
            "best_val_loss": float(final_val),
            "best_epoch": int(final_epoch),
            "diffusion_stage": self.stage,
            "search_space": search_space,
            "max_epochs": max_epochs,
            "patience": patience,
        }
        if reuse_from:
            meta_out.update({
                "reuse_tuned_params_from": str(reuse_from),
                "reused_max_scale_policy": best_params.get("max_scale"),
                "reused_max_scale_previous": reuse_meta.get("reused_max_scale_previous"),
            })
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, sort_keys=True)

        if self.stage == "coarse":
            state.diffusion_coarse_finetune_ckpt = final_ckpt
            state.coarse_finetune_best_params = best_params
        elif self.stage == "fine":
            state.diffusion_fine_finetune_ckpt = final_ckpt
            state.fine_finetune_best_params = best_params
        else:
            state.diffusion_finer_finetune_ckpt = final_ckpt
            state.finer_finetune_best_params = best_params

        wandb_utils.log_summary({
            f"hp/{self.stage}_diff_ft_best_val_loss": final_val,
            f"hp/{self.stage}_diff_ft_hp_best_val_loss": hp_best_val_loss,
            f"hp/{self.stage}_diff_ft_best_trial": best_trial_num,
            f"hp/{self.stage}_diff_ft_best_lr": best_params.get("learning_rate"),
            f"hp/{self.stage}_diff_ft_batch_size": best_params.get("batch_size"),
            f"hp/{self.stage}_diff_ft_max_scale": best_params.get("max_scale"),
        })
        return state


class CoarseDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_coarse_finetune_hp"
    stage = "coarse"


class FineDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_fine_finetune_hp"
    stage = "fine"


class FinerDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_finer_finetune_hp"
    stage = "finer"
