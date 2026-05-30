"""Phase 2B: Diffusion HP finetune on real data (finetuned iTransformer as guidance).

Runs Optuna HP search and promotes the best trial checkpoint to best.pt
(no separate full retrain step).
"""

from __future__ import annotations

import json
import logging
import os

import torch
from optuna import create_study
from optuna.samplers import TPESampler

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import _patch_globals
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class DiffusionFinetuneHPPhase(PipelinePhase):
    name = "diffusion_finetune_hp"

    def should_skip(self, state: PipelineState) -> bool:
        subset_id = state.subset_id or state.dataset
        best_pt = os.path.join(state.checkpoint_dir, subset_id, "best.pt")
        meta = os.path.join(state.checkpoint_dir, subset_id, "metadata.json")
        if os.path.exists(best_pt) and os.path.exists(meta):
            logger.info(f"  [{self.name}] cached: {best_pt}")
            state.diffusion_finetune_ckpt = best_pt
            try:
                with open(meta) as f:
                    data = json.load(f)
                    if "tuned_params" in data:
                        state.finetune_best_params = data["tuned_params"]
            except Exception as e:
                logger.warning(f"Failed to load tuned_params from metadata.json: {e}")
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        import optuna
        from models.diffusion_tsf.train_multivariate_pipeline import (
            finetune_hp_objective,
            _promote_best_trial_to_final,
            load_dataset,
            load_itransformer_from_checkpoint,
            select_diffusion_batch_size,
            generate_dataset_job,
            diffusion_probe_max_candidate,
        )
        from models.diffusion_tsf.guidance import iTransformerGuidance
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        _patch_globals(pipeline_mod, state)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        device = state.resolve_device()

        # Require predecessors
        ft_itrans_ckpt = state.itrans_finetune_ckpt
        if not ft_itrans_ckpt or not os.path.exists(ft_itrans_ckpt):
            raise RuntimeError(
                f"Phase 2B requires finetuned iTransformer from Phase 2A, "
                f"got: {ft_itrans_ckpt}"
            )
        diff_ckpt = state.diffusion_pretrain_ckpt
        if not diff_ckpt or not os.path.exists(diff_ckpt):
            raise RuntimeError(
                f"Phase 2B requires pretrained diffusion from Phase 1B, "
                f"got: {diff_ckpt}"
            )

        n_iv = len(variate_indices)

        # Probe batch size
        ft_itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, n_iv, device)
        ft_itrans_guidance = iTransformerGuidance(ft_itrans_model)
        probe_ds, _, _, _ = load_dataset(
            state.dataset, variate_indices,
            stride=train_stride, test_stride=test_stride,
        )
        ft_diff_bs = select_diffusion_batch_size(
            phase_name=f"Diff FT HP ({subset_id})",
            dataset=probe_ds,
            device=device,
            itrans_guidance=ft_itrans_guidance,
            max_candidate=diffusion_probe_max_candidate(n_iv, state.smoke_test),
            smoke_test=state.smoke_test,
        )
        del ft_itrans_model, ft_itrans_guidance, probe_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_trials = self.get("n_trials", 5)
        if state.smoke_test:
            n_trials = 1

        subset_dir = os.path.join(state.checkpoint_dir, subset_id)
        os.makedirs(subset_dir, exist_ok=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.optimize(
            lambda trial: finetune_hp_objective(
                trial, state.dataset, variate_indices, diff_ckpt, ft_itrans_ckpt,
                device, state.smoke_test,
                fixed_batch_size=ft_diff_bs, trial_ckpt_dir=subset_dir,
                train_stride=train_stride, test_stride=test_stride,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
            catch=(ValueError,),
        )
        if study.best_trial is None:
            logger.warning(f"All HP trials failed for {subset_id}")
            return state

        tuned_params = study.best_params
        tuned_params["batch_size"] = ft_diff_bs

        _, _, _, norm_stats = load_dataset(
            state.dataset, variate_indices,
            stride=train_stride, test_stride=test_stride,
        )
        subset_info = {
            "subset_id": subset_id,
            "variate_indices": variate_indices,
            "data_subset": subset_meta,
        }
        ckpt_path, train_metrics = _promote_best_trial_to_final(
            study, subset_dir, subset_info, state.dataset, norm_stats,
            ft_diff_bs, diff_ckpt, ft_itrans_ckpt, device, state.smoke_test,
        )

        state.diffusion_finetune_ckpt = ckpt_path
        state.finetune_best_params = tuned_params

        wandb_utils.log_summary({
            "hp/diff_ft_best_val_loss": train_metrics.get("best_val_loss"),
            "hp/diff_ft_best_trial": train_metrics.get("best_trial"),
            "hp/diff_ft_best_lr": tuned_params.get("learning_rate"),
        })

        return state
