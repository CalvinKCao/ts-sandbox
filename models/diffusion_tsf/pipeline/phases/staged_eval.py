"""Cfg-ablation-compatible eval for chained staged coarse/fine models."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import _stage_best_ckpt
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals

logger = logging.getLogger(__name__)


def _import_summarize_prediction_pack():
    """Import repo eval helper after iTransformer hijacks ``utils`` on sys.path."""
    import os
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[4])
    itrans_dir = str(Path(__file__).resolve().parents[3] / "iTransformer")
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(itrans_dir)]
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    from utils.eval_mmpd_gaussian_anchor import summarize_prediction_pack

    return summarize_prediction_pack


def _load_stage_metadata(state: PipelineState, stage: str) -> Dict:
    meta = os.path.join(os.path.dirname(_stage_best_ckpt(state, stage)), "metadata.json")
    if not os.path.exists(meta):
        return {}
    with open(meta) as f:
        return json.load(f)


def _stage_finetune_ckpt(state: PipelineState, stage: str) -> str:
    value = (
        state.diffusion_coarse_finetune_ckpt
        if stage == "coarse"
        else state.diffusion_fine_finetune_ckpt
    )
    if value and os.path.exists(value):
        return value
    path = _stage_best_ckpt(state, stage)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Missing staged {stage} checkpoint: {path}")


class StagedEvalPhase(PipelinePhase):
    name = "staged_eval"

    def should_skip(self, state: PipelineState) -> bool:
        subset_id = state.subset_id or state.dataset
        partial = os.path.join(state.results_dir, "partials", f"{state.dataset}_staged_anchor.json")
        nested = os.path.join(state.results_dir, subset_id, "staged_results.json")
        if os.path.exists(partial) and os.path.exists(nested):
            try:
                with open(partial) as f:
                    metrics = json.load(f)
                robust_ok = "texture_derivative_motif_jsd" in metrics
                prob_robust_ok = "prob_texture_derivative_motif_jsd" in metrics
            except Exception:
                robust_ok = False
                prob_robust_ok = False
            if robust_ok and prob_robust_ok:
                logger.info("  [%s] already evaluated with robust texture metrics: %s", self.name, partial)
                return True
            logger.info("  [%s] re-evaluating to add robust texture metrics: %s", self.name, partial)
        return False

    def _load_model(self, state: PipelineState, stage: str, itrans_guidance, n_iv: int, device: torch.device):
        from models.diffusion_tsf.train_multivariate_pipeline import (
            anchor_kwargs_from_params,
            create_diffusion_model,
            dataset_window_lengths,
            load_diffusion_state_keep_attached_guidance,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        meta = _load_stage_metadata(state, stage)
        tuned = meta.get("tuned_params") or {}
        model = create_diffusion_model(
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=stage,
            use_guidance_channel=state.use_guidance_channel,
            **anchor_kwargs_from_params(tuned),
        ).to(device)
        ckpt = torch.load(_stage_finetune_ckpt(state, stage), map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
        model.eval()
        return model

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            generate_dataset_job,
            load_dataset,
            load_itransformer_from_checkpoint,
        )
        from models.diffusion_tsf.guidance import iTransformerGuidance
        summarize_prediction_pack = _import_summarize_prediction_pack()

        device = state.resolve_device()
        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(self.get("test_stride", subset_meta.get("test_stride", 1)))
        n_iv = len(variate_indices)

        ft_itrans_ckpt = state.itrans_finetune_ckpt
        if not ft_itrans_ckpt or not os.path.exists(ft_itrans_ckpt):
            ft_itrans_ckpt = os.path.join(state.checkpoint_dir, f"{subset_id}_itransformer_finetuned.pt")
        if not os.path.exists(ft_itrans_ckpt):
            raise FileNotFoundError(f"Missing finetuned iTransformer checkpoint: {ft_itrans_ckpt}")

        itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, n_iv, device)
        itrans_guidance = iTransformerGuidance(itrans_model)
        coarse_model = self._load_model(state, "coarse", itrans_guidance, n_iv, device)
        fine_model = self._load_model(state, "fine", itrans_guidance, n_iv, device)

        _, _, test_ds, _ = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
        )
        if state.smoke_test:
            test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
            prob_samples = 1
            prob_steps = 5
        else:
            eval_fraction = self.get("eval_test_fraction", 0.5)
            n_full = len(test_ds)
            n_eval = max(1, int(round(n_full * float(eval_fraction))))
            rng = np.random.default_rng(state.seed)
            eval_idx = sorted(rng.choice(n_full, size=n_eval, replace=False).tolist())
            test_ds = Subset(test_ds, eval_idx)
            prob_samples = int(self.get("probabilistic_n_samples", 100))
            prob_steps = int(self.get("probabilistic_num_inference_steps", 20))

        batch_size = int(self.get("batch_size", 8 if not state.smoke_test else 2))
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        prob_sampler = self.get("probabilistic_sampler", "dpmpp")
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}

        y_true_all = []
        det_all = []
        sample_all = []
        t0 = time.perf_counter()
        logger.info(
            "[%s] staged eval start: windows=%d batches=%d prob_samples=%d sampler=%s steps=%d",
            subset_id,
            len(test_ds),
            len(loader),
            prob_samples,
            prob_sampler,
            prob_steps,
        )
        with torch.no_grad():
            for batch_idx, (past, future) in enumerate(loader):
                past = past.to(device)
                future = future.to(device)
                K = getattr(coarse_model.config, "lookback_overlap", 0)
                if K > 0:
                    future = future[..., K:]
                y_true_all.append(future.cpu().numpy())

                torch.manual_seed(state.seed + batch_idx)
                coarse_det = coarse_model.generate(past, sampler="anchor")
                fine_det = fine_model.generate(
                    past,
                    sampler="anchor",
                    future_coarse_2d=coarse_det["future_2d_coarse"],
                )
                det_all.append(fine_det["prediction_global_norm"].cpu().numpy())

                batch_samples = []
                for sample_idx in range(prob_samples):
                    seed = state.seed + batch_idx * 1009 + sample_idx * 17
                    torch.manual_seed(seed)
                    coarse_sample = coarse_model.generate(past, **prob_kwargs)
                    torch.manual_seed(seed)
                    fine_sample = fine_model.generate(
                        past,
                        future_coarse_2d=coarse_sample["future_2d_coarse"],
                        **prob_kwargs,
                    )
                    batch_samples.append(fine_sample["prediction_global_norm"].cpu().numpy())
                sample_all.append(np.stack(batch_samples, axis=2))

                if batch_idx < 3 or batch_idx == len(loader) - 1:
                    logger.info(
                        "[%s] staged eval batch %d/%d elapsed=%.1fs",
                        subset_id,
                        batch_idx + 1,
                        len(loader),
                        time.perf_counter() - t0,
                    )

        pack = {
            "y_true": np.concatenate(y_true_all, axis=0),
            "deterministic": np.concatenate(det_all, axis=0),
            "samples": np.concatenate(sample_all, axis=0),
        }
        metrics = summarize_prediction_pack(
            pack,
            gmm_components=int(self.get("gmm_components", 10)),
            seed=state.seed,
            topk_max=int(self.get("topk_max", 3)),
            texture_per_sample=True,
        )

        partial_dir = os.path.join(state.results_dir, "partials")
        raw_dir = os.path.join(state.results_dir, "raw")
        nested_dir = os.path.join(state.results_dir, subset_id)
        os.makedirs(partial_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        with open(os.path.join(partial_dir, f"{state.dataset}_staged_anchor.json"), "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        np.savez_compressed(os.path.join(raw_dir, f"staged_anchor_{state.dataset}.npz"), **pack)
        with open(os.path.join(nested_dir, "staged_results.json"), "w") as f:
            json.dump({
                "dataset": state.dataset,
                "subset_id": subset_id,
                "variate_indices": variate_indices,
                "data_subset": subset_meta,
                "eval_metrics": {"staged_anchor": metrics},
            }, f, indent=2, sort_keys=True)

        wandb_utils.log_summary({
            "eval/staged_mse": metrics.get("mse"),
            "eval/staged_mae": metrics.get("mae"),
            "eval/staged_crps": metrics.get("crps"),
            "eval/staged_top1_mse": metrics.get("top1_mse"),
            "eval/staged_top3_mse": metrics.get("top3_mse"),
        })
        logger.info(
            "[%s] staged eval done: mse=%.4f mae=%.4f crps=%.4f",
            subset_id,
            metrics.get("mse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("crps", float("nan")),
        )
        return state
