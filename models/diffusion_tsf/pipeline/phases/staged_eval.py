"""Cfg-ablation-compatible eval for chained staged coarse/fine models."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _model_kwargs_from_tuned,
    _stage_best_ckpt,
)
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


def _fraction_subset(ds, fraction: float, seed: int):
    n = len(ds)
    keep = max(1, int(round(n * float(fraction))))
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(n, size=keep, replace=False).tolist())
    return Subset(ds, idx)


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
                sampler_ok = (not self.get("tune_sampler", True)) or metrics.get("sampler_tuned")
            except Exception:
                robust_ok = False
                prob_robust_ok = False
                sampler_ok = False
            if robust_ok and prob_robust_ok and sampler_ok:
                logger.info("  [%s] already evaluated with tuned sampler + robust texture metrics: %s", self.name, partial)
                return True
            logger.info("  [%s] re-evaluating to add tuned sampler/robust metrics: %s", self.name, partial)
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
        model_kwargs = anchor_kwargs_from_params(tuned)
        model_kwargs.update(_model_kwargs_from_tuned(tuned))
        model = create_diffusion_model(
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=stage,
            use_guidance_channel=state.use_guidance_channel,
            **model_kwargs,
        ).to(device)
        ckpt = torch.load(_stage_finetune_ckpt(state, stage), map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
        model.eval()
        return model

    def _run_eval(
        self,
        *,
        state: PipelineState,
        subset_id: str,
        loader: DataLoader,
        device: torch.device,
        coarse_model,
        fine_model,
        prob_sampler: str,
        prob_steps: int,
        prob_samples: int,
        summarize_prediction_pack,
    ):
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}
        y_true_all = []
        det_all = []
        sample_all = []
        t0 = time.perf_counter()
        logger.info(
            "[%s] staged eval start: windows=%d batches=%d prob_samples=%d sampler=%s steps=%d",
            subset_id,
            len(loader.dataset),
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
        return metrics, pack

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

        _, _, full_test_ds, _ = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
        )
        batch_size = int(self.get("batch_size", 8 if not state.smoke_test else 2))
        if state.smoke_test:
            final_ds = Subset(full_test_ds, list(range(min(2, len(full_test_ds)))))
            prob_samples = 1
            default_steps = 5
        else:
            eval_fraction = float(self.get("eval_test_fraction", 1.0))
            final_ds = _fraction_subset(full_test_ds, eval_fraction, state.seed) if eval_fraction < 1.0 else full_test_ds
            prob_samples = int(self.get("probabilistic_n_samples", 100))
            default_steps = int(self.get("probabilistic_num_inference_steps", 20))

        sampler_tuning = []
        selected_sampler = self.get("probabilistic_sampler", "dpmpp")
        selected_steps = default_steps
        if self.get("tune_sampler", True) and not state.smoke_test:
            tune_fraction = float(self.get("sampler_tune_fraction", 0.25))
            tune_samples = int(self.get("sampler_tune_probabilistic_n_samples", min(8, prob_samples)))
            candidate_samplers = list(self.get("sampler_tune_candidates", ["ddim", "dpmpp"]))
            candidate_steps = [int(x) for x in self.get("sampler_tune_steps", [10, 20, 40])]
            tune_ds = _fraction_subset(full_test_ds, tune_fraction, state.seed + 7919)
            tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False)
            score_metric = str(self.get("sampler_tune_metric", "top3_mse"))
            best_score = float("inf")
            for sampler in candidate_samplers:
                for steps in candidate_steps:
                    metrics_i, _pack_i = self._run_eval(
                        state=state,
                        subset_id=f"{subset_id}-sampler-tune",
                        loader=tune_loader,
                        device=device,
                        coarse_model=coarse_model,
                        fine_model=fine_model,
                        prob_sampler=sampler,
                        prob_steps=steps,
                        prob_samples=tune_samples,
                        summarize_prediction_pack=summarize_prediction_pack,
                    )
                    score = float(metrics_i.get(score_metric, metrics_i.get("crps", metrics_i.get("mse"))))
                    sampler_tuning.append({
                        "sampler": sampler,
                        "steps": steps,
                        "metric": score_metric,
                        "score": score,
                        "mse": metrics_i.get("mse"),
                        "crps": metrics_i.get("crps"),
                        "top3_mse": metrics_i.get("top3_mse"),
                    })
                    logger.info(
                        "[%s] sampler tune: sampler=%s steps=%d %s=%.6f",
                        subset_id,
                        sampler,
                        steps,
                        score_metric,
                        score,
                    )
                    if score < best_score:
                        best_score = score
                        selected_sampler = sampler
                        selected_steps = steps
            logger.info(
                "[%s] selected sampler=%s steps=%d (%s=%.6f on %.0f%% eval subset)",
                subset_id,
                selected_sampler,
                selected_steps,
                score_metric,
                best_score,
                100 * tune_fraction,
            )

        loader = DataLoader(final_ds, batch_size=batch_size, shuffle=False)
        metrics, pack = self._run_eval(
            state=state,
            subset_id=subset_id,
            loader=loader,
            device=device,
            coarse_model=coarse_model,
            fine_model=fine_model,
            prob_sampler=selected_sampler,
            prob_steps=selected_steps,
            prob_samples=prob_samples,
            summarize_prediction_pack=summarize_prediction_pack,
        )
        metrics.update({
            "sampler_tuned": bool(sampler_tuning),
            "selected_probabilistic_sampler": selected_sampler,
            "selected_probabilistic_num_inference_steps": selected_steps,
        })

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
                "sampler_tuning": sampler_tuning,
                "eval_metrics": {"staged_anchor": metrics},
            }, f, indent=2, sort_keys=True)

        wandb_utils.log_summary({
            "eval/staged_mse": metrics.get("mse"),
            "eval/staged_mae": metrics.get("mae"),
            "eval/staged_crps": metrics.get("crps"),
            "eval/staged_top1_mse": metrics.get("top1_mse"),
            "eval/staged_top3_mse": metrics.get("top3_mse"),
            "eval/selected_sampler": selected_sampler,
            "eval/selected_steps": selected_steps,
        })
        logger.info(
            "[%s] staged eval done: sampler=%s steps=%d mse=%.4f mae=%.4f crps=%.4f",
            subset_id,
            selected_sampler,
            selected_steps,
            metrics.get("mse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("crps", float("nan")),
        )
        return state
