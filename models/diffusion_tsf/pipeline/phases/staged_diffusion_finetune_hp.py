"""HP tuning + final full-data training for staged coarse/fine diffusion models."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from optuna import create_study
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


def _fraction_subset(ds, fraction: float, seed: int) -> Subset:
    n = len(ds)
    keep = max(1, int(round(n * float(fraction))))
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:keep].sort().values.tolist()
    return Subset(ds, idx)


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
) -> Dict[str, Any]:
    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    if smoke_test:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
            "batch_size": min(max(1, max_batch_size), 2),
            "ema_decay": 0.0,
            "binary_noise_schedule": "linear",
            "loss_weighting": "none",
            "min_snr_gamma": 5.0,
            "prediction_target": "x0",
            "max_scale": base_ms,
        }

    batch_grid = [b for b in (4, 8, 16, 32, 48, 64, 96, 128) if b <= max_batch_size]
    if not batch_grid:
        batch_grid = [max(1, max_batch_size)]
    scale_low = max(2.5, base_ms * 0.8)
    scale_high = min(14.0, base_ms * 1.25)
    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-6, 8e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", batch_grid),
        "ema_decay": trial.suggest_categorical("ema_decay", [0.0, 0.99, 0.995, 0.999]),
        "binary_noise_schedule": trial.suggest_categorical(
            "binary_noise_schedule", ["linear", "cosine"]
        ),
        "loss_weighting": trial.suggest_categorical("loss_weighting", ["none", "min_snr"]),
        "prediction_target": trial.suggest_categorical("prediction_target", ["x0", "epsilon"]),
        "max_scale": trial.suggest_float("max_scale", scale_low, scale_high),
    }
    params["min_snr_gamma"] = (
        trial.suggest_float("min_snr_gamma", 1.0, 10.0, log=True)
        if params["loss_weighting"] == "min_snr"
        else 5.0
    )
    return params


class _BaseStagedDiffusionFinetuneHPPhase(PipelinePhase):
    stage = ""

    def should_skip(self, state: PipelineState) -> bool:
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
            else:
                state.diffusion_fine_finetune_ckpt = best_pt
                state.fine_finetune_best_params = params
            return True
        return False

    def _pretrained_ckpt(self, state: PipelineState) -> str:
        attr = (
            state.diffusion_coarse_pretrain_ckpt
            if self.stage == "coarse"
            else state.diffusion_fine_pretrain_ckpt
        )
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
        ckpt_path: str,
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
                        "final_full_data_retrain": trial is None,
                    }
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
        tune_fraction = float(self.get("tune_data_fraction", 0.5))
        tune_train_ds = _fraction_subset(train_ds, tune_fraction, state.seed + (11 if self.stage == "coarse" else 17))
        tune_val_ds = _fraction_subset(val_ds, tune_fraction, state.seed + (23 if self.stage == "coarse" else 29))
        logger.info(
            "  [%s] train/val full=%d/%d tune=%d/%d",
            self.name,
            len(train_ds),
            len(val_ds),
            len(tune_train_ds),
            len(tune_val_ds),
        )

        batch_probe_ds = train_ds if self.get("reuse_tuned_params_from") else tune_train_ds
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

        n_trials = int(self.get("n_trials", 20))
        if state.smoke_test:
            n_trials = 1
        hp_epochs = int(self.get("hp_max_epochs", self.get("max_epochs", 12)))
        hp_patience = int(self.get("hp_patience", self.get("patience", 4)))
        final_epochs = int(self.get("final_max_epochs", self.get("max_epochs", 20)))
        final_patience = int(self.get("final_patience", self.get("patience", 8)))
        if state.smoke_test:
            hp_epochs = final_epochs = hp_patience = final_patience = 1

        subset_dir = _stage_subset_dir(state, self.stage)
        os.makedirs(subset_dir, exist_ok=True)

        reuse_from = self.get("reuse_tuned_params_from")
        reuse_meta: Dict[str, Any] = {}
        hp_best_val_loss: Optional[float] = None
        best_trial_num = -1

        if reuse_from:
            best_params, source_dir, reuse_meta = _load_reused_stage_params(
                state,
                stage=self.stage,
                subset_id=subset_id,
                source_config=str(reuse_from),
            )
            tuned_bs = int(best_params.get("batch_size", max_batch))
            best_params["batch_size"] = min(tuned_bs, max_batch)
            hp_best_val_loss = float(
                reuse_meta.get("best_val_loss")
                or reuse_meta.get("hp_best_val_loss")
                or float("nan")
            )
            logger.info(
                "  [%s] reuse tuned_params from %s (%s); policy max_scale=%.4g (was %.4g)",
                self.name,
                source_dir,
                self.stage,
                best_params["max_scale"],
                reuse_meta.get("reused_max_scale_previous"),
            )
        else:
            study = create_study(
                direction="minimize",
                sampler=TPESampler(seed=state.seed, multivariate=True, group=True),
                pruner=HyperbandPruner(min_resource=1, max_resource=hp_epochs, reduction_factor=3),
            )

            def objective(trial):
                params = _suggest_staged_params(trial, state, max_batch, state.smoke_test)
                trial.set_user_attr("full_params", dict(params))
                trial_ckpt = os.path.join(subset_dir, f"_diff_ft_trial_{trial.number}_best.pt")
                try:
                    best_val, _best_epoch = self._train_once(
                        state=state,
                        train_ds=tune_train_ds,
                        val_ds=tune_val_ds,
                        params=params,
                        pretrained_path=diff_ckpt,
                        itrans_checkpoint=ft_itrans_ckpt,
                        device=device,
                        variate_indices=variate_indices,
                        ckpt_path=trial_ckpt,
                        max_epochs=hp_epochs,
                        patience=hp_patience,
                        trial=trial,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "  [%s] trial %d OOM (batch=%s), pruning",
                        self.name,
                        trial.number,
                        params.get("batch_size"),
                    )
                    raise TrialPruned() from None
                return best_val

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            try:
                best_trial = study.best_trial
            except ValueError as e:
                raise RuntimeError(
                    f"All {self.stage} diffusion HP trials failed for {subset_id} "
                    f"({len(study.trials)} trials, none completed)"
                ) from e

            best_params = dict(best_trial.user_attrs.get("full_params") or best_trial.params)
            best_params.setdefault("min_snr_gamma", 5.0)
            best_params.setdefault(
                "max_scale",
                float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
            )
            hp_best_val_loss = float(study.best_value)
            best_trial_num = int(best_trial.number)
        final_ckpt = _stage_best_ckpt(state, self.stage)
        final_val, final_epoch = self._train_once(
            state=state,
            train_ds=train_ds,
            val_ds=val_ds,
            params=best_params,
            pretrained_path=diff_ckpt,
            itrans_checkpoint=ft_itrans_ckpt,
            device=device,
            variate_indices=variate_indices,
            ckpt_path=final_ckpt,
            max_epochs=final_epochs,
            patience=final_patience,
            trial=None,
        )

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
            "final_full_data_retrain": True,
            "tune_data_fraction": tune_fraction,
            "diffusion_stage": self.stage,
        }
        if reuse_from:
            meta_out.update({
                "reuse_tuned_params_from": str(reuse_from),
                "reused_max_scale_policy": best_params.get("max_scale"),
                "reused_max_scale_previous": reuse_meta.get("reused_max_scale_previous"),
            })
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, sort_keys=True)
        if not reuse_from:
            for fn in os.listdir(subset_dir):
                if fn.startswith("_diff_ft_trial_") and fn.endswith("_best.pt"):
                    try:
                        os.remove(os.path.join(subset_dir, fn))
                    except OSError:
                        pass

        if self.stage == "coarse":
            state.diffusion_coarse_finetune_ckpt = final_ckpt
            state.coarse_finetune_best_params = best_params
        else:
            state.diffusion_fine_finetune_ckpt = final_ckpt
            state.fine_finetune_best_params = best_params

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
