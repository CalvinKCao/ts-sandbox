"""Phase: patch decoder + mixer HP finetune on real data (window-norm MSE)."""

from __future__ import annotations

import logging
import os
import shutil

import torch

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    discover_dataset_run_ckpt_dir,
)
from models.diffusion_tsf.pipeline.reused_paths import find_reused_guidance_ckpt
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class PatchGuidanceFinetuneHPPhase(PipelinePhase):
    name = "patch_guidance_finetune_hp"

    def _patch_guidance_ckpt_usable(self, state: PipelineState, ckpt_path: str) -> bool:
        if not os.path.exists(ckpt_path):
            return False
        if not state.use_ordinal_window_norm:
            return True
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("  [%s] cannot read patch guidance ckpt %s: %s", self.name, ckpt_path, e)
            return False
        if bool(ckpt.get("ordinal_patch_guidance_unit_ranks", False)):
            return True
        logger.info(
            "  [%s] stale patch guidance (missing unit-rank ordinal targets): %s",
            self.name,
            ckpt_path,
        )
        return False

    def _try_reuse_patch_guidance_ckpt(
        self,
        state: PipelineState,
        *,
        fail_if_missing: bool,
    ) -> bool:
        subset_id = state.subset_id or state.dataset
        ft_ckpt = state.default_guidance_finetune_ckpt_path()
        if self._patch_guidance_ckpt_usable(state, ft_ckpt):
            state.patch_guidance_finetune_ckpt = ft_ckpt
            return True

        reuse_from = self.get("reuse_checkpoint_from_config")
        if not reuse_from:
            return False

        reused = find_reused_guidance_ckpt(str(reuse_from), subset_id)
        if reused and self._patch_guidance_ckpt_usable(state, reused):
            os.makedirs(os.path.dirname(ft_ckpt), exist_ok=True)
            shutil.copy2(reused, ft_ckpt)
            state.patch_guidance_finetune_ckpt = ft_ckpt
            logger.info("  [%s] reused finetuned patch guidance from %s", self.name, reused)
            return True

        ckpt_name = f"{subset_id}_patch_guidance.pt"
        try:
            source_dir = discover_dataset_run_ckpt_dir(
                state, str(reuse_from), required_file=ckpt_name,
            )
        except FileNotFoundError:
            if fail_if_missing:
                raise
            return False

        src_ckpt = os.path.join(source_dir, ckpt_name)
        if not self._patch_guidance_ckpt_usable(state, src_ckpt):
            if fail_if_missing:
                raise FileNotFoundError(
                    f"Missing usable patch guidance finetune to reuse: {src_ckpt} "
                    f"(from *-{state.dataset}-{reuse_from}; need "
                    f"ordinal_patch_guidance_unit_ranks=True for ordinal runs)"
                )
            return False

        os.makedirs(os.path.dirname(ft_ckpt), exist_ok=True)
        shutil.copy2(src_ckpt, ft_ckpt)
        state.patch_guidance_finetune_ckpt = ft_ckpt
        logger.info("  [%s] reused finetuned patch guidance from %s", self.name, source_dir)
        return True

    def should_skip(self, state: PipelineState) -> bool:
        if state.guidance_type != "patch_decoder":
            logger.info("  [%s] skipping: guidance_type=%s", self.name, state.guidance_type)
            return True
        if self._try_reuse_patch_guidance_ckpt(state, fail_if_missing=False):
            ckpt = state.default_guidance_finetune_ckpt_path()
            logger.info("  [%s] cached: %s", self.name, ckpt)
            return True
        return False

    def on_skip(self, state: PipelineState) -> PipelineState:
        self._log_finetune_viz_and_diagnostics(state)
        return state

    def _log_finetune_viz_and_diagnostics(self, state: PipelineState) -> None:
        if state.guidance_type != "patch_decoder":
            return

        ft_ckpt = state.patch_guidance_finetune_ckpt or state.default_guidance_finetune_ckpt_path()
        if not os.path.exists(ft_ckpt):
            logger.warning("  [%s] viz skipped: missing patch guidance ckpt %s", self.name, ft_ckpt)
            return

        # The decoder checkpoint is used only to produce frozen cross-variate
        # tokens. Its forecast overlays are not a diffusion conditioning path.
        try:
            from models.diffusion_tsf.train_multivariate_pipeline import (
                generate_dataset_job,
                load_dataset,
            )
            from models.diffusion_tsf.pipeline.visualize_utils import (
                run_patch_guidance_finetune_diagnostics,
            )

            variate_indices = state.variate_indices
            if not variate_indices:
                raise ValueError(
                    f"[{self.name}] Missing resolved variate_indices in state for dataset {state.dataset!r}. "
                    "Data subset policy must be resolved before running phase."
                )
            subset_meta = state.data_subset_resolved or {}
            train_stride = int(subset_meta.get("train_stride", state.window_stride))
            test_stride = int(subset_meta.get("test_stride", 1))
            train_ds, _, _, _ = load_dataset(
                state, state.dataset,
                variate_indices,
                lookback=state.lookback_length,
                horizon=state.forecast_length,
                lookback_overlap=state.lookback_overlap,
                stride=train_stride,
                test_stride=test_stride,
                ordinal_tie_atol=float(state.ordinal_tie_atol),
                use_ordinal_window_norm=state.use_ordinal_window_norm,
            )
            diag = run_patch_guidance_finetune_diagnostics(
                state, ckpt_path=ft_ckpt, train_ds=train_ds,
            )
            wandb_utils.log_phase_diagnostics_result(diag)
        except Exception as e:
            logger.warning("Patch guidance finetune diagnostics failed: %s", e, exc_info=True)

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            run_patch_guidance_finetune_hp_tuning,
            generate_dataset_job,
        )
        subset_id = state.subset_id or state.dataset
        ft_ckpt = state.default_guidance_finetune_ckpt_path()

        if self.get("reuse_checkpoint_from_config"):
            if self._try_reuse_patch_guidance_ckpt(state, fail_if_missing=False):
                return state
            logger.warning(
                "  [%s] reuse_checkpoint_from_config=%r missing usable ckpt; "
                "training patch guidance instead",
                self.name,
                self.get("reuse_checkpoint_from_config"),
            )
            # fall through to train

        variate_indices = state.variate_indices
        if not variate_indices:
            raise ValueError(
                f"[{self.name}] Missing resolved variate_indices in state for dataset {state.dataset!r}. "
                "Data subset policy must be resolved before running phase."
            )
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        n_trials = int(self.require("n_trials"))
        if state.smoke_test:
            n_trials = 1

        best_params, tune_ckpt_path = run_patch_guidance_finetune_hp_tuning(
            state, dataset_name=state.dataset,
            variate_indices=variate_indices,
            n_trials=n_trials,
            device=state.resolve_device(),
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
            subset_id=subset_id,
            train_stride=train_stride,
            test_stride=test_stride,
            parallel_workers=state.parallel_optuna_workers,
        )

        hp_best = os.path.join(state.checkpoint_dir, f"{subset_id}_patch_guidance_hp_best.pt")
        if os.path.exists(hp_best) and not os.path.exists(ft_ckpt):
            shutil.copy2(hp_best, ft_ckpt)

        state.patch_guidance_finetune_ckpt = ft_ckpt

        wandb_utils.log_summary({
            "hp/patch_guidance_ft_best_lr": best_params.get("learning_rate"),
        })
        if tune_ckpt_path:
            logger.info("  [%s] best HP checkpoint: %s", self.name, tune_ckpt_path)

        if os.path.exists(ft_ckpt):
            # Full forecast viz stays non-smoke; ordinal space alignment runs on smoke too.
            self._log_finetune_viz_and_diagnostics(state)

        return state
