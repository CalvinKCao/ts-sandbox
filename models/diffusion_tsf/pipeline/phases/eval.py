"""Eval phase: evaluate finetuned diffusion model + iTransformer baseline.

Runs diffusion evaluation on a random half of the test set (seeded),
then trains + evaluates a standalone iTransformer baseline for comparison.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import _patch_globals
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.visualize_utils import generate_pipeline_visualizations

logger = logging.getLogger(__name__)


class EvalPhase(PipelinePhase):
    name = "eval"

    def should_skip(self, state: PipelineState) -> bool:
        subset_id = state.subset_id or state.dataset
        from models.diffusion_tsf.train_multivariate_pipeline import _load_subset_results
        prior = _load_subset_results(state.results_dir, subset_id)
        if prior.get("eval_metrics"):
            logger.info(f"  [{self.name}] already evaluated: {subset_id}")
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            evaluate_model,
            create_diffusion_model,
            load_diffusion_state_keep_attached_guidance,
            load_itransformer_from_checkpoint,
            load_dataset,
            save_eval_results,
            train_subset_itransformer_full_baseline,
            evaluate_itransformer_baseline,
            generate_dataset_job,
            anchor_kwargs_from_params,
        )
        from models.diffusion_tsf.guidance import iTransformerGuidance
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        _patch_globals(pipeline_mod, state)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]

        device = state.resolve_device()
        n_iv = len(variate_indices)

        # Load finetuned models
        ft_itrans_ckpt = state.itrans_finetune_ckpt
        diff_ckpt = state.diffusion_finetune_ckpt
        if not ft_itrans_ckpt or not os.path.exists(ft_itrans_ckpt):
            raise RuntimeError(f"Eval requires finetuned iTransformer, got: {ft_itrans_ckpt}")
        if not diff_ckpt or not os.path.exists(diff_ckpt):
            raise RuntimeError(f"Eval requires finetuned diffusion model, got: {diff_ckpt}")

        itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, n_iv, device)
        itrans_guidance = iTransformerGuidance(itrans_model)

        tuned_params = state.finetune_best_params or {}
        model = create_diffusion_model(
            n_variates=n_iv, **anchor_kwargs_from_params(tuned_params),
        ).to(device)
        model.set_guidance_model(itrans_guidance)
        ckpt = torch.load(diff_ckpt, map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])

        # Conditionally load Phase 3 residual model
        res_ckpt = state.diffusion_residual_finetune_ckpt
        residual_model = None
        if res_ckpt and os.path.exists(res_ckpt):
            logger.info(f"[{subset_id}] Loading Phase 3 residual model: {res_ckpt}")
            res_params = state.residual_finetune_best_params or {}
            residual_model = create_diffusion_model(
                n_variates=n_iv, **anchor_kwargs_from_params(res_params),
            ).to(device)
            residual_model.set_guidance_model(itrans_guidance)
            residual_model.config.is_residual_model = True
            r_ckpt = torch.load(res_ckpt, map_location=device, weights_only=False)
            load_diffusion_state_keep_attached_guidance(residual_model, r_ckpt["model_state_dict"])
            residual_model.eval()

        # Load test data
        _, _, test_ds, _ = load_dataset(
            state.dataset, variate_indices, stride=1, test_stride=1,
        )
        n_samples = self.get("n_samples", 30)
        if state.smoke_test:
            test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
            n_samples = 1
        else:
            n_full = len(test_ds)
            n_eval = max(1, n_full // 2)
            rng = np.random.default_rng(42)
            eval_idx = sorted(rng.choice(n_full, size=n_eval, replace=False).tolist())
            test_ds = Subset(test_ds, eval_idx)
            logger.info(f"[{subset_id}] eval subset: {n_eval}/{n_full} windows")

        test_loader = DataLoader(test_ds, batch_size=8 if not state.smoke_test else 2, shuffle=False)
        eval_results = evaluate_model(model, test_loader, device, n_samples=n_samples, smoke_test=state.smoke_test, residual_model=residual_model)

        logger.info(
            f"[{subset_id}] Avg MSE={eval_results['averaged']['mse']:.4f}, "
            f"MAE={eval_results['averaged']['mae']:.4f}"
        )

        train_metrics = {"tuned_params": tuned_params}
        save_eval_results(
            subset_id, state.dataset, variate_indices,
            train_metrics, eval_results, state.results_dir,
        )

        # wandb
        wandb_utils.log_summary({
            "eval/mse": eval_results["averaged"]["mse"],
            "eval/mae": eval_results["averaged"]["mae"],
            "eval/trend_accuracy": eval_results["averaged"].get("trend_accuracy"),
            "eval/single_mse": eval_results["single"]["mse"],
            "eval/single_mae": eval_results["single"]["mae"],
        })

        # iTransformer-only baseline
        try:
            full_itrans_ckpt = os.path.join(
                state.checkpoint_dir, f"{subset_id}_itrans_full_dataset.pt",
            )
            if not os.path.exists(full_itrans_ckpt):
                full_itrans_ckpt = train_subset_itransformer_full_baseline(
                    state.dataset, variate_indices, subset_id, device,
                    smoke_test=state.smoke_test,
                )
            eval_test_indices = None
            if not state.smoke_test and isinstance(test_ds, Subset):
                eval_test_indices = list(test_ds.indices)
            baseline_metrics = evaluate_itransformer_baseline(
                subset_id, state.dataset, variate_indices,
                full_itrans_ckpt, state.results_dir, device,
                smoke_test=state.smoke_test,
                test_indices=eval_test_indices,
            )
            wandb_utils.log_summary({
                "eval/itrans_baseline/mse": baseline_metrics.get("mse"),
                "eval/itrans_baseline/mae": baseline_metrics.get("mae"),
            })
        except Exception as e:
            logger.warning(f"iTransformer baseline failed for {subset_id}: {e}")

        # ---------------------------------------------------------
        # Pipeline Visualizations
        # ---------------------------------------------------------
        if wandb_utils._WANDB_AVAILABLE and wandb_utils.wandb.run is not None:
            wandb = wandb_utils.wandb
            logger.info(f"[{subset_id}] Generating and logging pipeline visualizations...")
            viz_output_dir = os.path.join(state.results_dir, "viz", subset_id)
            
            # The test_ds contains windows, we can pass it down. 
            # We already have stats (from load_dataset), wait, we need stats.
            _, _, _, norm_stats = load_dataset(
                state.dataset, variate_indices, stride=1, test_stride=1,
            )
            stats = (norm_stats['mean'], norm_stats['std'])
            
            try:
                viz_paths = generate_pipeline_visualizations(
                    model=model,
                    itrans_model=itrans_model,
                    dataset=test_ds,
                    stats=stats,
                    device=device,
                    output_dir=viz_output_dir,
                    subset_id=subset_id,
                    n_samples=2 if not state.smoke_test else 1,
                    forecast_length=pipeline_mod.FORECAST_LENGTH,
                    lookback_length=pipeline_mod.LOOKBACK_LENGTH,
                )
                
                # Log to wandb with alphanumeric sorting logic natively maintained by the sequential file naming 
                # (e.g., 001_..., 002_...)
                viz_paths.sort()
                wandb_images = [wandb.Image(p, caption=os.path.basename(p)) for p in viz_paths]
                wandb.log({"eval/visualizations": wandb_images})
                logger.info(f"[{subset_id}] Successfully logged {len(viz_paths)} visualizations to W&B.")
            except Exception as e:
                logger.warning(f"Failed to generate pipeline visualizations: {e}", exc_info=True)

        return state
