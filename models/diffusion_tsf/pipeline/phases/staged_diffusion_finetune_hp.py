"""Phase 2B-style HP tuning for staged coarse/fine diffusion models."""

from __future__ import annotations

import json
import logging
import os

import torch
from optuna import create_study
from optuna.samplers import TPESampler

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _stage_pretrain_ckpt,
    patch_stage_globals,
)

logger = logging.getLogger(__name__)


def _stage_subset_dir(state: PipelineState, stage: str) -> str:
    subset_id = state.subset_id or state.dataset
    return os.path.join(state.checkpoint_dir, subset_id, stage)


def _stage_best_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_subset_dir(state, stage), "best.pt")


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

    def execute(self, state: PipelineState) -> PipelineState:
        import optuna
        from models.diffusion_tsf.train_multivariate_pipeline import (
            _promote_best_trial_to_final,
            diffusion_probe_max_candidate,
            finetune_hp_objective,
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
        diff_ckpt = self._pretrained_ckpt(state)

        device = state.resolve_device()
        n_iv = len(variate_indices)
        train_ds, val_ds, _, norm_stats = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
        )
        logger.info(
            "  [%s] loaded %d train / %d val windows",
            self.name,
            len(train_ds),
            len(val_ds),
        )

        ft_itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, n_iv, device)
        ft_itrans_guidance = iTransformerGuidance(ft_itrans_model)
        ft_diff_bs = select_diffusion_batch_size(
            phase_name=f"{self.stage.title()} Diff FT HP ({subset_id})",
            dataset=train_ds,
            device=device,
            itrans_guidance=ft_itrans_guidance,
            max_candidate=diffusion_probe_max_candidate(n_iv, state.smoke_test),
            smoke_test=state.smoke_test,
        )
        del ft_itrans_model, ft_itrans_guidance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_trials = self.get("n_trials", 5)
        if state.smoke_test:
            n_trials = 1

        subset_dir = _stage_subset_dir(state, self.stage)
        os.makedirs(subset_dir, exist_ok=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.optimize(
            lambda trial: finetune_hp_objective(
                trial,
                state.dataset,
                variate_indices,
                diff_ckpt,
                ft_itrans_ckpt,
                device,
                state.smoke_test,
                fixed_batch_size=ft_diff_bs,
                trial_ckpt_dir=subset_dir,
                train_stride=train_stride,
                test_stride=test_stride,
                train_ds=train_ds,
                val_ds=val_ds,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
            catch=(ValueError, FileNotFoundError, OSError),
        )
        if study.best_trial is None:
            raise RuntimeError(f"All {self.stage} diffusion HP trials failed for {subset_id}")

        subset_info = {
            "subset_id": subset_id,
            "variate_indices": variate_indices,
            "data_subset": subset_meta,
            "diffusion_stage": self.stage,
        }
        ckpt_path, train_metrics = _promote_best_trial_to_final(
            study,
            subset_dir,
            subset_info,
            state.dataset,
            norm_stats,
            ft_diff_bs,
            diff_ckpt,
            ft_itrans_ckpt,
            device,
            state.smoke_test,
        )

        tuned_params = dict(study.best_params)
        tuned_params["batch_size"] = ft_diff_bs
        if self.stage == "coarse":
            state.diffusion_coarse_finetune_ckpt = ckpt_path
            state.coarse_finetune_best_params = tuned_params
        else:
            state.diffusion_fine_finetune_ckpt = ckpt_path
            state.fine_finetune_best_params = tuned_params

        wandb_utils.log_summary({
            f"hp/{self.stage}_diff_ft_best_val_loss": train_metrics.get("best_val_loss"),
            f"hp/{self.stage}_diff_ft_best_trial": train_metrics.get("best_trial"),
            f"hp/{self.stage}_diff_ft_best_lr": tuned_params.get("learning_rate"),
        })
        return state


class CoarseDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_coarse_finetune_hp"
    stage = "coarse"


class FineDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_fine_finetune_hp"
    stage = "fine"
