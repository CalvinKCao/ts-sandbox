"""Cfg-ablation-compatible eval for chained staged coarse/fine models."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.config import visualization_settings
from models.diffusion_tsf.pipeline.visualize_utils import (
    decode_staged_anchor_components,
    per_window_anchor_mse,
    per_window_crps,
    run_eval_probabilistic_sample_visualizations,
    run_eval_full_dataset_visualization,
    run_eval_worst_window_visualizations,
    run_ordinal_roundtrip_visualization,
    run_ordinal_coarse_fine_2d_visualization,
    run_real_dataset_phase_diagnostics,
    run_staged_finetune_visualizations,
    run_vertical_dual_repr_visualization,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _model_kwargs_from_tuned,
    _stage_best_ckpt,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals

logger = logging.getLogger(__name__)


def _reshape_parallel_samples(t: torch.Tensor, batch: int, n_samples: int) -> torch.Tensor:
    """``(B*S, V, ...)`` → ``(B, V, S, ...)``."""
    if t.shape[0] != batch * n_samples:
        raise ValueError(
            f"parallel sample reshape expected leading {batch * n_samples}, got {tuple(t.shape)}"
        )
    rest = t.shape[1:]
    return t.view(batch, n_samples, *rest).transpose(1, 2).contiguous()


@torch.no_grad()
def _probe_max_staged_eval_batch_size(
    *,
    coarse_model,
    fine_model,
    lookback: int,
    n_variates: int,
    device: torch.device,
    det_kwargs: Dict[str, Any],
    joint_dual: bool = False,
    min_bs: int = 1,
    max_bs: int = 64,
    headroom: float = 0.85,
) -> int:
    """Largest window batch that fits one coarse→fine anchor generate on this GPU."""
    if device.type != "cuda":
        return max(min_bs, 8)

    def _fits(bs: int) -> bool:
        past = torch.zeros(bs, n_variates, lookback, device=device)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            coarse_out = coarse_model.generate(past, **det_kwargs)
            if not joint_dual:
                fine_model.generate(
                    past,
                    future_coarse_2d=coarse_out["future_2d_coarse"],
                    **det_kwargs,
                )
            torch.cuda.synchronize(device)
            return True
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            return False

    lo = max(1, int(min_bs))
    hi = max(lo, int(max_bs))
    if not _fits(lo):
        logger.warning(
            "staged_eval batch probe: min_bs=%d already OOMs; falling back to 1", lo,
        )
        return 1
    best = lo
    cand = lo
    while cand * 2 <= hi and _fits(cand * 2):
        cand *= 2
        best = cand
    # Binary search (best, next power] for a tighter fit.
    left, right = best, min(hi, best * 2)
    while left < right:
        mid = (left + right + 1) // 2
        if _fits(mid):
            left = mid
        else:
            right = mid - 1
    best = left
    usable = max(1, int(best * float(headroom)))
    logger.info(
        "staged_eval batch probe: max_fit=%d headroom=%.2f -> batch_size=%d",
        best,
        headroom,
        usable,
    )
    torch.cuda.empty_cache()
    return usable


def _deterministic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
    }


def _staged_anchor_global_norm(
    fine_model,
    coarse_out: Dict[str, Any],
    fine_out: Dict[str, Any],
) -> np.ndarray:
    """Decode chained coarse→refine/fine forecast in global norm space."""
    pred = fine_out.get("prediction_global_norm", fine_out.get("prediction"))
    if pred is not None:
        if isinstance(pred, np.ndarray):
            return pred
        return pred.detach().cpu().numpy()
    if getattr(fine_model.config, "diffusion_stage", "") == "patch_refine":
        raise RuntimeError("patch_refine eval output missing prediction_global_norm")
    coarse_2d = coarse_out["future_2d_coarse"]
    fine_2d = fine_out["future_2d_fine"]
    pred = fine_model.decode_dual_from_2d(coarse_2d, fine_2d, from_diffusion=False)
    pred = fine_model._strip_overlap_and_upsample_repr(pred)
    return pred.detach().cpu().numpy()


def _summarize_staged_eval_metrics(
  pack: Dict[str, np.ndarray],
  *,
  gmm_components: int,
  seed: int,
  topk_max: int,
) -> Dict[str, float]:
    from models.diffusion_tsf.metrics import probabilistic_forecast_metrics

    y_true = pack["y_true"]
    samples = pack["samples"]
    sample_mean = samples.mean(axis=2)
    metrics = probabilistic_forecast_metrics(
        y_true,
        samples,
        gmm_components=gmm_components,
        topk_max=topk_max,
        seed=seed,
    )
    sample_mean_metrics = _deterministic_metrics(y_true, sample_mean)
    metrics["sample_mean_mse"] = sample_mean_metrics["mse"]
    metrics["sample_mean_mae"] = sample_mean_metrics["mae"]
    # Keep legacy top-level names pointed at the 100-sample mean forecast.
    metrics["mse"] = sample_mean_metrics["mse"]
    metrics["mae"] = sample_mean_metrics["mae"]

    anchor = _deterministic_metrics(y_true, pack["deterministic"])
    metrics["anchor_mse"] = anchor["mse"]
    metrics["anchor_mae"] = anchor["mae"]
    metrics["anchor_n_samples"] = 1.0
    metrics["metrics_profile"] = "dpmpp_prob_core_plus_anchor"
    return metrics


def _load_stage_metadata(state: PipelineState, stage: str) -> Dict:
    meta = os.path.join(os.path.dirname(_stage_best_ckpt(state, stage)), "metadata.json")
    if not os.path.exists(meta):
        return {}
    with open(meta) as f:
        return json.load(f)


def _stage_finetune_ckpt(state: PipelineState, stage: str) -> str:
    value = {
        "coarse": state.diffusion_coarse_finetune_ckpt,
        "fine": state.diffusion_fine_finetune_ckpt,
        "finer": state.diffusion_finer_finetune_ckpt,
        "vertical_dual": state.diffusion_vertical_dual_finetune_ckpt,
        "channel_dual": state.diffusion_channel_dual_finetune_ckpt,
        "patch_refine": state.diffusion_patch_refine_finetune_ckpt,
    }[stage]
    if value and os.path.exists(value):
        return value
    path = _stage_best_ckpt(state, stage)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Missing staged {stage} checkpoint: {path}")


def _eval_window_indices(ds) -> list:
    if isinstance(ds, Subset):
        return [int(i) for i in ds.indices]
    return list(range(len(ds)))


def _fraction_subset(ds, fraction: float, seed: int):
    n = len(ds)
    keep = max(1, int(round(n * float(fraction))))
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(n, size=keep, replace=False).tolist())
    return Subset(ds, idx)


def _resolve_eval_test_fraction(phase: PipelinePhase, state: PipelineState) -> float:
    by_dataset = phase.get("eval_test_fraction_by_dataset") or {}
    if state.dataset in by_dataset:
        return float(by_dataset[state.dataset])
    return float(phase.require("eval_test_fraction"))


def _ar_eval_enabled(model) -> bool:
    chunk = int(getattr(model.config, "diffusion_chunk_horizon", 0) or 0)
    if chunk <= 0:
        return False
    dataset_h = int(getattr(model.config, "dataset_forecast_length", 0) or 0)
    return dataset_h > chunk


def _staged_generate_once(
    *,
    coarse_model,
    fine_model,
    finer_model,
    past: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    coarse_out = coarse_model.generate(past, **gen_kwargs)
    fine_out = fine_model.generate(
        past,
        future_coarse_2d=coarse_out["future_2d_coarse"],
        **gen_kwargs,
    )
    if finer_model is not None:
        finer_out = finer_model.generate(
            past,
            future_coarse_2d=coarse_out["future_2d_coarse"],
            future_fine_2d=fine_out["future_2d_fine"],
            **gen_kwargs,
        )
        pred = finer_out["prediction_global_norm"]
        return {"coarse": coarse_out, "fine": fine_out, "finer": finer_out, "prediction": pred}
    pred = _staged_anchor_global_norm(fine_model, coarse_out, fine_out)
    pred_t = torch.from_numpy(pred).to(past.device)
    return {"coarse": coarse_out, "fine": fine_out, "prediction": pred_t}


def _staged_generate_autoregressive(
    *,
    coarse_model,
    fine_model,
    finer_model,
    past: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Roll out staged coarse/fine in AR chunks; return global-norm forecast (B,V,H)."""
    K = int(getattr(coarse_model.config, "lookback_overlap", 0))
    dataset_h = int(getattr(coarse_model.config, "dataset_forecast_length", 0) or 0)
    n_chunks = coarse_model._ar_num_chunks(dataset_h)
    pieces = []
    remaining = dataset_h
    for c in range(n_chunks):
        if c == 0:
            past_c = past
        else:
            hist = torch.cat(pieces, dim=-1)
            past_c = torch.cat([past, hist], dim=-1)
        out = _staged_generate_once(
            coarse_model=coarse_model,
            fine_model=fine_model,
            finer_model=finer_model,
            past=past_c,
            gen_kwargs=gen_kwargs,
        )
        chunk = out["prediction"]
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).to(past.device)
        if c > 0:
            chunk = chunk[..., K:]
        if chunk.shape[-1] > remaining:
            chunk = chunk[..., :remaining]
        pieces.append(chunk)
        remaining -= chunk.shape[-1]
        if remaining <= 0:
            break
    return torch.cat(pieces, dim=-1)


def _staged_det_gen_kwargs(state: PipelineState, default_steps: int) -> Dict[str, Any]:
    sampler = str(getattr(state, "eval_sampler", "anchor"))
    if sampler in ("anchor", "deterministic_anchor"):
        return {"sampler": sampler}
    steps = 5 if state.smoke_test else int(default_steps)
    return {"sampler": sampler, "num_inference_steps": steps}


class StagedEvalPhase(PipelinePhase):
    name = "staged_eval"

    def should_skip(self, state: PipelineState) -> bool:
        if bool(self.get("refresh_eval_visualizations", False)):
            logger.info("  [%s] forcing eval refresh for visualizations", self.name)
            return False
        subset_id = state.subset_id or state.dataset
        partial = os.path.join(state.results_dir, "partials", f"{state.dataset}_staged_anchor.json")
        nested = os.path.join(state.results_dir, subset_id, "staged_results.json")
        raw_dir = os.path.join(state.results_dir, "raw")
        anchor_npz = os.path.join(raw_dir, f"staged_anchor_samples_{state.dataset}.npz")
        samples_npz = os.path.join(raw_dir, f"staged_dpmpp_samples_{state.dataset}.npz")
        full_npz = os.path.join(raw_dir, f"staged_anchor_{state.dataset}.npz")
        worst_json = os.path.join(state.results_dir, subset_id, "worst_windows.json")
        if os.path.exists(partial) and os.path.exists(nested):
            try:
                with open(partial) as f:
                    metrics = json.load(f)
                core_ok = "crps" in metrics and "top3_mse" in metrics
                anchor_ok = "anchor_mse" in metrics and "anchor_mae" in metrics
                sample_mean_ok = "sample_mean_mse" in metrics and "sample_mean_mae" in metrics
                raw_ok = os.path.exists(anchor_npz) and os.path.exists(samples_npz)
                diag_ok = True
                if os.path.exists(full_npz):
                    with np.load(full_npz) as z:
                        diag_ok = all(k in z.files for k in ("coarse_anchor", "window_indices"))
                worst_ok = os.path.exists(worst_json)
                sampler_ok = (not self.get("tune_sampler", True)) or metrics.get("sampler_tuned")
            except Exception:
                core_ok = False
                anchor_ok = False
                sample_mean_ok = False
                raw_ok = False
                diag_ok = False
                worst_ok = False
                sampler_ok = False
            if core_ok and anchor_ok and sample_mean_ok and raw_ok and diag_ok and worst_ok and sampler_ok:
                logger.info("  [%s] already evaluated: %s", self.name, partial)
                return True
            logger.info("  [%s] re-evaluating to add missing metrics: %s", self.name, partial)
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
            ordinal_ladder=pipeline_mod.GLOBAL_ORDINAL_LADDER,
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
        finer_model=None,
        prob_sampler: str,
        prob_steps: int,
        prob_samples: int,
        gmm_components: int,
        topk_max: int,
        window_indices: Sequence[int],
        test_stride: int,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        if prob_sampler in {"anchor", "deterministic_anchor"}:
            raise ValueError("staged probabilistic eval must use a regular sampler, not anchor.")
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}
        det_kwargs = _staged_det_gen_kwargs(state, prob_steps)
        vertical_dual = bool(getattr(state, "use_vertical_dual_concat", False))
        channel_dual = bool(getattr(state, "use_channel_dual_concat", False))
        joint_dual = vertical_dual or channel_dual
        dual_stage = "channel_dual" if channel_dual else ("vertical_dual" if vertical_dual else None)
        y_true_all = []
        y_true_with_overlap_all = []
        det_all = []
        det_with_overlap_all = []
        coarse_all = []
        fine_all = []
        sample_all = []
        samples_with_overlap_all = []
        window_idx_all = []
        t0 = time.perf_counter()
        ranked = getattr(loader.dataset, "yields_ordinal_ranks", False)
        if isinstance(loader.dataset, Subset):
            ranked = getattr(loader.dataset.dataset, "yields_ordinal_ranks", False)
        for m in (coarse_model, fine_model, finer_model):
            if m is None:
                continue
            m._ordinal_input_is_ranked = ranked
            m._ordinal_apply_ood_shift = not ranked
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
                batch_n = past.shape[0]
                batch_start = batch_idx * loader.batch_size
                batch_window_indices = window_indices[batch_start:batch_start + batch_n]
                window_idx_all.extend(batch_window_indices)
                K = getattr(coarse_model.config, "lookback_overlap", 0)
                y_true_with_overlap_all.append(future.cpu().numpy())
                if K > 0:
                    future = future[..., K:]
                y_true_all.append(future.cpu().numpy())

                torch.manual_seed(state.seed + batch_idx)
                batch_t0 = time.perf_counter()
                if joint_dual:
                    dual_det = coarse_model.generate(past, **det_kwargs)
                    det_t = dual_det.get("prediction_global_norm", dual_det.get("prediction"))
                    if det_t is None:
                        raise KeyError(
                            f"{dual_stage} generate output missing prediction_global_norm/prediction"
                        )
                    det_all.append(det_t.detach().cpu().numpy())
                    det_with_overlap_all.append(
                        dual_det["prediction_with_overlap"].detach().cpu().numpy()
                    )
                    if channel_dual:
                        coarse_all.append(dual_det["future_2d_coarse"].detach().cpu().numpy())
                        fine_all.append(dual_det["future_2d_fine"].detach().cpu().numpy())
                    else:
                        future_2d = dual_det["future_2d"]
                        coarse_all.append(
                            future_2d[:, :, :int(state.coarse_image_height)].detach().cpu().numpy()
                        )
                        fine_all.append(
                            future_2d[:, :, int(state.coarse_image_height):].detach().cpu().numpy()
                        )
                elif _ar_eval_enabled(coarse_model):
                    det_t = _staged_generate_autoregressive(
                        coarse_model=coarse_model,
                        fine_model=fine_model,
                        finer_model=finer_model,
                        past=past,
                        gen_kwargs=det_kwargs,
                    )
                    det_all.append(det_t.detach().cpu().numpy())
                    coarse_det = coarse_model.generate(past, **det_kwargs)
                    fine_det = fine_model.generate(
                        past,
                        future_coarse_2d=coarse_det["future_2d_coarse"],
                        **det_kwargs,
                    )
                    coarse_np, fine_np, _ = decode_staged_anchor_components(
                        fine_model, coarse_det, fine_det,
                    )
                    coarse_all.append(coarse_np)
                    fine_all.append(fine_np)
                else:
                    coarse_det = coarse_model.generate(past, **det_kwargs)
                    fine_det = fine_model.generate(
                        past,
                        future_coarse_2d=coarse_det["future_2d_coarse"],
                        **det_kwargs,
                    )
                    if finer_model is not None:
                        finer_det = finer_model.generate(
                            past,
                            future_coarse_2d=coarse_det["future_2d_coarse"],
                            future_fine_2d=fine_det["future_2d_fine"],
                            **det_kwargs,
                        )
                        det_t = finer_det["prediction_global_norm"]
                        det_all.append(det_t.detach().cpu().numpy())
                        det_with_overlap_all.append(
                            finer_det["prediction_with_overlap"].detach().cpu().numpy()
                        )
                        coarse_np, fine_np, final_np = decode_staged_anchor_components(
                            finer_model, coarse_det, finer_det,
                        )
                    else:
                        coarse_np, fine_np, final_np = decode_staged_anchor_components(
                            fine_model, coarse_det, fine_det,
                        )
                        det_all.append(fine_det["prediction_global_norm"].detach().cpu().numpy())
                        det_with_overlap_all.append(
                            fine_det["prediction_with_overlap"].detach().cpu().numpy()
                        )
                    coarse_all.append(coarse_np)
                    fine_all.append(fine_np)

                det_s = time.perf_counter() - batch_t0
                prob_t0 = time.perf_counter()
                # Expand window batch across independent MC samples so unique-seg
                # AR (and other generate paths) fill the GPU in one forward chain.
                torch.manual_seed(state.seed + batch_idx * 1009)
                past_exp = past.repeat_interleave(prob_samples, dim=0)
                if joint_dual:
                    dual_sample = coarse_model.generate(past_exp, **prob_kwargs)
                    sample_t = dual_sample.get(
                        "prediction_global_norm", dual_sample.get("prediction"),
                    )
                    if sample_t is None:
                        raise KeyError(
                            f"{dual_stage} generate output missing prediction_global_norm/prediction"
                        )
                    samples_bvs = _reshape_parallel_samples(sample_t, batch_n, prob_samples)
                    overlap_bvs = _reshape_parallel_samples(
                        dual_sample["prediction_with_overlap"], batch_n, prob_samples,
                    )
                    sample_all.append(samples_bvs.detach().cpu().numpy())
                    samples_with_overlap_all.append(overlap_bvs.detach().cpu().numpy())
                elif _ar_eval_enabled(coarse_model):
                    sample_t = _staged_generate_autoregressive(
                        coarse_model=coarse_model,
                        fine_model=fine_model,
                        finer_model=finer_model,
                        past=past_exp,
                        gen_kwargs=prob_kwargs,
                    )
                    samples_bvs = _reshape_parallel_samples(sample_t, batch_n, prob_samples)
                    sample_all.append(samples_bvs.detach().cpu().numpy())
                else:
                    coarse_sample = coarse_model.generate(past_exp, **prob_kwargs)
                    fine_sample = fine_model.generate(
                        past_exp,
                        future_coarse_2d=coarse_sample["future_2d_coarse"],
                        **prob_kwargs,
                    )
                    if finer_model is not None:
                        finer_sample = finer_model.generate(
                            past_exp,
                            future_coarse_2d=coarse_sample["future_2d_coarse"],
                            future_fine_2d=fine_sample["future_2d_fine"],
                            **prob_kwargs,
                        )
                        pred = finer_sample["prediction_global_norm"]
                        overlap = finer_sample["prediction_with_overlap"]
                    else:
                        pred = fine_sample["prediction_global_norm"]
                        overlap = fine_sample["prediction_with_overlap"]
                    samples_bvs = _reshape_parallel_samples(pred, batch_n, prob_samples)
                    overlap_bvs = _reshape_parallel_samples(overlap, batch_n, prob_samples)
                    sample_all.append(samples_bvs.detach().cpu().numpy())
                    samples_with_overlap_all.append(overlap_bvs.detach().cpu().numpy())

                prob_s = time.perf_counter() - prob_t0
                batch_s = time.perf_counter() - batch_t0
                done = batch_idx + 1
                elapsed = time.perf_counter() - t0
                eta_s = (elapsed / done) * (len(loader) - done) if done else 0.0
                logger.info(
                    "[%s] staged eval batch %d/%d n=%d "
                    "det=%.1fs prob=%.1fs (n_samp=%d parallel) batch=%.1fs "
                    "elapsed=%.1fs eta=%.1fs",
                    subset_id,
                    done,
                    len(loader),
                    batch_n,
                    det_s,
                    prob_s,
                    prob_samples,
                    batch_s,
                    elapsed,
                    eta_s,
                )

        pack = {
            "y_true": np.concatenate(y_true_all, axis=0),
            "deterministic": np.concatenate(det_all, axis=0),
            "coarse_anchor": np.concatenate(coarse_all, axis=0),
            "fine_anchor": np.concatenate(fine_all, axis=0),
            "final_anchor": np.concatenate(det_all, axis=0),
            "samples": np.concatenate(sample_all, axis=0),
            "window_indices": np.array(window_idx_all, dtype=np.int64),
            "series_starts": np.array(window_idx_all, dtype=np.int64) * int(test_stride),
        }
        if det_with_overlap_all and len(det_with_overlap_all) == len(det_all):
            pack["y_true_with_overlap"] = np.concatenate(y_true_with_overlap_all, axis=0)
            pack["deterministic_with_overlap"] = np.concatenate(det_with_overlap_all, axis=0)
            pack["final_anchor_with_overlap"] = pack["deterministic_with_overlap"]
        if samples_with_overlap_all and len(samples_with_overlap_all) == len(sample_all):
            pack["samples_with_overlap"] = np.concatenate(samples_with_overlap_all, axis=0)
            pack["sample_mean_with_overlap"] = pack["samples_with_overlap"].mean(axis=2)
        pack["sample_mean"] = pack["samples"].mean(axis=2)
        metrics = _summarize_staged_eval_metrics(
            pack,
            gmm_components=gmm_components,
            seed=state.seed,
            topk_max=topk_max,
        )
        return metrics, pack

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            generate_dataset_job,
            load_dataset,
            load_wrapped_guidance,
            dataset_window_lengths,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
        from models.diffusion_tsf.pipeline.globals_bridge import patch_globals

        device = state.resolve_device()
        gmm_components = int(self.require("gmm_components"))
        topk_max = int(self.require("topk_max"))
        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        phase_test_stride = int(self.require("test_stride"))
        subset_test_stride = int(subset_meta.get("test_stride", 1))
        # Never evaluate denser than the subset policy (e.g. dynamic sample_stride=480).
        test_stride = max(phase_test_stride, subset_test_stride)
        if test_stride != phase_test_stride:
            logger.info(
                "[%s] eval test_stride=%d (phase=%d, subset=%d)",
                subset_id,
                test_stride,
                phase_test_stride,
                subset_test_stride,
            )
        n_iv = len(variate_indices)

        patch_globals(pipeline_mod, state, honor_dataset_windows=True)
        full_train_ds, full_val_ds, full_test_ds, norm_stats = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
            ordinal_tie_atol=float(state.ordinal_tie_atol),
            use_ordinal_window_norm=state.use_ordinal_window_norm,
        )
        if norm_stats.get("ordinal_ladder") is not None:
            state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]

        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = state.default_guidance_finetune_ckpt_path()
        needs_guidance = state.needs_guidance
        if needs_guidance and not os.path.exists(ft_guidance_ckpt):
            raise FileNotFoundError(f"Missing finetuned guidance checkpoint: {ft_guidance_ckpt}")
        if not needs_guidance:
            ft_guidance_ckpt = ""

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        guidance = None
        if needs_guidance:
            guidance = load_wrapped_guidance(
                ft_guidance_ckpt,
                n_iv,
                device,
                guidance_type=state.guidance_type,
                dataset_lookback=ds_lb,
                dataset_horizon=ds_hz,
            )
        vertical_dual = bool(getattr(state, "use_vertical_dual_concat", False))
        channel_dual = bool(getattr(state, "use_channel_dual_concat", False))
        patch_refine = bool(getattr(state, "use_patch_refine_stage", False))
        joint_dual = vertical_dual or channel_dual
        dual_stage = "channel_dual" if channel_dual else ("vertical_dual" if vertical_dual else None)
        if joint_dual:
            coarse_model = self._load_model(state, dual_stage, guidance, n_iv, device)
            fine_model = coarse_model
            finer_model = None
        else:
            coarse_model = self._load_model(state, "coarse", guidance, n_iv, device)
            refine_stage = "patch_refine" if patch_refine else "fine"
            fine_model = self._load_model(state, refine_stage, guidance, n_iv, device)
            finer_model = (
                self._load_model(state, "finer", guidance, n_iv, device)
                if state.use_triple_scale and not patch_refine
                else None
            )

        batch_size = int(self.require("batch_size"))
        if state.smoke_test:
            final_ds = Subset(full_test_ds, list(range(min(2, len(full_test_ds)))))
            prob_samples = 1
            default_steps = 5
        else:
            eval_fraction = _resolve_eval_test_fraction(self, state)
            final_ds = _fraction_subset(full_test_ds, eval_fraction, state.seed) if eval_fraction < 1.0 else full_test_ds
            if eval_fraction < 1.0:
                logger.info(
                    "[%s] eval subset: %d/%d windows (eval_test_fraction=%.3f)",
                    subset_id,
                    len(final_ds),
                    len(full_test_ds),
                    eval_fraction,
                )
            prob_samples = int(self.require("probabilistic_n_samples"))
            default_steps = int(self.require("probabilistic_num_inference_steps"))

        # Probe peak generate batch on this GPU; dataloader batch is smaller so
        # that B_windows * n_prob_samples still fits the parallel MC expand.
        if (
            bool(self.get("probe_eval_batch_size", False))
            and not state.smoke_test
            and device.type == "cuda"
        ):
            probe_kwargs = dict(_staged_det_gen_kwargs(state, default_steps))
            probe_kwargs["num_inference_steps"] = 1
            max_fit = _probe_max_staged_eval_batch_size(
                coarse_model=coarse_model,
                fine_model=fine_model,
                lookback=int(ds_lb),
                n_variates=n_iv,
                device=device,
                det_kwargs=probe_kwargs,
                joint_dual=joint_dual,
                min_bs=1,
                max_bs=int(self.get("probe_eval_batch_size_max", 64)),
            )
            # Parallel samples expand leading dim by prob_samples.
            usable = max(1, max_fit // max(1, int(prob_samples)))
            if usable != batch_size:
                logger.info(
                    "[%s] staged_eval probe: config batch_size=%d -> probed=%d "
                    "(max_fit=%d / n_samples=%d)",
                    subset_id,
                    batch_size,
                    usable,
                    max_fit,
                    prob_samples,
                )
            batch_size = usable

        if isinstance(final_ds, Subset):
            eval_window_indices = [int(i) for i in final_ds.indices]
        else:
            eval_window_indices = list(range(len(final_ds)))

        try:
            from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics

            diagnostic_stages = (
                [(dual_stage, coarse_model, _stage_finetune_ckpt(state, dual_stage), None)]
                if joint_dual
                else [
                    ("coarse", coarse_model, _stage_finetune_ckpt(state, "coarse"), None),
                    (
                        "patch_refine" if patch_refine else "fine",
                        fine_model,
                        _stage_finetune_ckpt(state, "patch_refine" if patch_refine else "fine"),
                        _stage_finetune_ckpt(state, "coarse"),
                    ),
                ]
            )
            ckpt_info = []
            if ft_guidance_ckpt and os.path.exists(ft_guidance_ckpt):
                ckpt_info.append(
                    {
                        "kind": state.guidance_type,
                        "path": ft_guidance_ckpt,
                        "n_variates": n_iv,
                        "lookback": int(ds_lb),
                        "horizon": int(ds_hz),
                    }
                )
            ckpt_info.extend(
                {
                    "kind": f"diffusion_{stage}",
                    "path": ckpt,
                    "n_variates": n_iv,
                    "lookback": int(ds_lb),
                    "horizon": int(ds_hz),
                }
                for stage, _model, ckpt, _coarse_ckpt in diagnostic_stages
            )
            run_phase_start_diagnostics(
                state,
                phase_name=self.name,
                models=[item[1] for item in diagnostic_stages],
                model_labels=[f"diffusion_{item[0]}" for item in diagnostic_stages],
                ckpt_info=ckpt_info,
            )
            _, _, test_ds, _ = load_dataset(
                state.dataset,
                variate_indices,
                stride=train_stride,
                test_stride=test_stride,
                ordinal_tie_atol=float(state.ordinal_tie_atol),
                use_ordinal_window_norm=state.use_ordinal_window_norm,
            )
            for eval_stage, eval_model, eval_ckpt, eval_coarse in diagnostic_stages:
                diag = run_real_dataset_phase_diagnostics(
                    state,
                    train_ds=test_ds,
                    model=eval_model,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    stage=eval_stage,
                    diffusion_ckpt_path=eval_ckpt,
                    coarse_ckpt_path=eval_coarse,
                    tag=f"staged_eval/{eval_stage}",
                    include_phase_start=(eval_stage in {"coarse", "vertical_dual", "channel_dual"}),
                )
                wandb_utils.log_phase_diagnostics_result(diag)
        except Exception as e:
            logger.warning("[%s] eval diagnostics failed: %s", self.name, e, exc_info=True)

        sampler_tuning = []
        selected_sampler = str(self.require("probabilistic_sampler"))
        if selected_sampler in {"anchor", "deterministic_anchor"}:
            raise ValueError(
                "staged probabilistic_sampler must be ddim, quad_t, or ddim_quad, not anchor."
            )
        selected_steps = default_steps
        if bool(self.require("tune_sampler")) and not state.smoke_test:
            tune_fraction = float(self.require("sampler_tune_fraction"))
            tune_samples = int(self.require("sampler_tune_probabilistic_n_samples"))
            candidate_samplers = list(self.require("sampler_tune_candidates"))
            candidate_steps = [int(x) for x in self.require("sampler_tune_steps")]
            tune_ds = _fraction_subset(full_test_ds, tune_fraction, state.seed + 7919)
            tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False)
            score_metric = str(self.require("sampler_tune_metric"))
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
                        finer_model=finer_model,
                        prob_sampler=sampler,
                        prob_steps=steps,
                        prob_samples=tune_samples,
                        gmm_components=gmm_components,
                        topk_max=topk_max,
                        window_indices=_eval_window_indices(tune_ds),
                        test_stride=test_stride,
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
            finer_model=finer_model,
            prob_sampler=selected_sampler,
            prob_steps=selected_steps,
            prob_samples=prob_samples,
            gmm_components=gmm_components,
            topk_max=topk_max,
            window_indices=eval_window_indices,
            test_stride=test_stride,
        )
        metrics.update({
            "sampler_tuned": bool(sampler_tuning),
            "selected_probabilistic_sampler": selected_sampler,
            "selected_probabilistic_num_inference_steps": selected_steps,
        })

        from models.diffusion_tsf.pipeline.phase_diagnostics import select_spaced_top_k

        crps_scores = per_window_crps(pack["y_true"], pack["samples"])
        anchor_scores = per_window_anchor_mse(pack["y_true"], pack["final_anchor"])
        series_starts = pack["series_starts"]
        window_indices_arr = pack["window_indices"]
        worst_manifest: List[Dict[str, Any]] = []
        for metric_name, scores in (("crps", crps_scores), ("anchor_mse", anchor_scores)):
            top_idx = select_spaced_top_k(scores, series_starts, k=10, min_spacing=48)
            for rank, wi in enumerate(top_idx, start=1):
                worst_manifest.append({
                    "metric": metric_name,
                    "rank": rank,
                    "window_index": int(window_indices_arr[wi]),
                    "series_start": int(series_starts[wi]),
                    "score": float(scores[wi]),
                })

        partial_dir = os.path.join(state.results_dir, "partials")
        raw_dir = os.path.join(state.results_dir, "raw")
        nested_dir = os.path.join(state.results_dir, subset_id)
        os.makedirs(partial_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        with open(os.path.join(nested_dir, "worst_windows.json"), "w") as f:
            json.dump(worst_manifest, f, indent=2)
        with open(os.path.join(partial_dir, f"{state.dataset}_staged_anchor.json"), "w") as f:
            payload = dict(metrics)
            payload["seed"] = int(state.seed)
            payload["binary_length_mode"] = getattr(state, "binary_length_mode", "none")
            payload["binary_length_g"] = float(getattr(state, "binary_length_g", 1.0))
            json.dump(payload, f, indent=2, sort_keys=True)
        np.savez_compressed(os.path.join(raw_dir, f"staged_anchor_{state.dataset}.npz"), **pack)
        np.savez_compressed(
            os.path.join(raw_dir, f"staged_anchor_samples_{state.dataset}.npz"),
            y_true=pack["y_true"],
            anchor=pack["deterministic"],
        )
        np.savez_compressed(
            os.path.join(raw_dir, f"staged_dpmpp_samples_{state.dataset}.npz"),
            y_true=pack["y_true"],
            samples=pack["samples"],
            sample_mean=pack["sample_mean"],
        )
        with open(os.path.join(nested_dir, "staged_results.json"), "w") as f:
            json.dump({
                "dataset": state.dataset,
                "subset_id": subset_id,
                "seed": int(state.seed),
                "binary_length_mode": getattr(state, "binary_length_mode", "none"),
                "binary_length_g": float(getattr(state, "binary_length_g", 1.0)),
                "variate_indices": variate_indices,
                "data_subset": subset_meta,
                "sampler_tuning": sampler_tuning,
                "eval_metrics": {"staged_anchor": metrics},
            }, f, indent=2, sort_keys=True)

        wandb_utils.log_eval_metrics({
            "eval/staged_prob_mse": metrics.get("mse"),
            "eval/staged_prob_mae": metrics.get("mae"),
            "eval/staged_sample_mean_mse": metrics.get("sample_mean_mse"),
            "eval/staged_sample_mean_mae": metrics.get("sample_mean_mae"),
            "eval/staged_anchor_mse": metrics.get("anchor_mse"),
            "eval/staged_anchor_mae": metrics.get("anchor_mae"),
            "eval/staged_crps": metrics.get("crps"),
            "eval/staged_top1_mse": metrics.get("top1_mse"),
            "eval/staged_top3_mse": metrics.get("top3_mse"),
            "eval/selected_sampler": selected_sampler,
            "eval/selected_steps": selected_steps,
        })

        skip_viz = bool(
            self.get("skip_eval_visualizations", False)
            or state.extra.get("skip_eval_visualizations", False)
        )
        viz_cfg = visualization_settings(state.merged_config)
        coarse_ft = fine_ft = None
        if not joint_dual:
            coarse_ft = state.diffusion_coarse_finetune_ckpt or _stage_finetune_ckpt(state, "coarse")
            refine_stage = "patch_refine" if patch_refine else "fine"
            fine_ft = (
                state.diffusion_patch_refine_finetune_ckpt
                if patch_refine
                else state.diffusion_fine_finetune_ckpt
            ) or _stage_finetune_ckpt(state, refine_stage)
        if not joint_dual and not skip_viz and viz_cfg.get("enabled", True) and not state.smoke_test:
            try:
                tuned = state.fine_finetune_best_params or state.coarse_finetune_best_params
                viz_paths = run_staged_finetune_visualizations(
                    state,
                    coarse_ckpt_path=coarse_ft,
                    fine_ckpt_path=fine_ft,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    tuned_params=tuned,
                    tag="eval_staged_dual_scale",
                )
                wandb_utils.log_visualization_paths(
                    viz_paths, wandb_key="eval/dual_scale_visualizations",
                )
            except Exception as e:
                logger.warning("Staged eval visualizations failed: %s", e, exc_info=True)

        # Vertical-dual stacked-repr panels: always write locally (including smoke).
        if vertical_dual and not skip_viz and viz_cfg.get("enabled", True):
            try:
                vd_paths = run_vertical_dual_repr_visualization(
                    state,
                    model=coarse_model,
                    device=device,
                    tag="eval_vertical_dual",
                )
                wandb_utils.log_visualization_paths(
                    vd_paths, wandb_key="viz/vertical_dual_repr",
                )
            except Exception as e:
                logger.warning("Vertical-dual repr viz failed: %s", e, exc_info=True)

        if not skip_viz and viz_cfg.get("enabled", True):
            try:
                worst_viz = run_eval_worst_window_visualizations(
                    state,
                    test_ds=full_test_ds,
                    pack=pack,
                    worst_manifest=worst_manifest,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    device=device,
                )
                wandb_utils.log_visualization_paths(worst_viz, wandb_key="eval/worst_samples")
            except Exception as e:
                logger.warning("Worst-window eval viz failed: %s", e, exc_info=True)

            try:
                prob_viz = run_eval_probabilistic_sample_visualizations(
                    state,
                    test_ds=full_test_ds,
                    pack=pack,
                    worst_manifest=worst_manifest,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    device=device,
                    sampler=selected_sampler,
                    num_inference_steps=selected_steps,
                )
                wandb_utils.log_visualization_paths(
                    prob_viz, wandb_key="eval/probabilistic_samples",
                )
            except Exception as e:
                logger.warning("Probabilistic sample eval viz failed: %s", e, exc_info=True)

            try:
                dataset_viz = run_eval_full_dataset_visualization(
                    state, splits={"train": full_train_ds, "val": full_val_ds, "test": full_test_ds},
                )
                wandb_utils.log_visualization_paths(dataset_viz, wandb_key="eval/full_dataset_splits")
            except Exception as e:
                logger.warning("Full-dataset eval viz failed: %s", e, exc_info=True)

            if state.use_ordinal_window_norm:
                try:
                    ord_paths = run_ordinal_roundtrip_visualization(state, split="test")
                    wandb_utils.log_visualization_paths(
                        ord_paths, wandb_key="eval/ordinal_roundtrip",
                    )
                except Exception as e:
                    logger.warning("Ordinal roundtrip viz failed: %s", e, exc_info=True)
                try:
                    repr_paths = run_ordinal_coarse_fine_2d_visualization(state, variate=0)
                    wandb_utils.log_visualization_paths(
                        repr_paths, wandb_key="eval/ordinal_coarse_fine_2d",
                    )
                except Exception as e:
                    logger.warning("Ordinal coarse/fine 2D viz failed: %s", e, exc_info=True)

        logger.info(
            "[%s] staged eval done: sampler=%s steps=%d "
            "prob_mse=%.4f prob_mae=%.4f anchor_mse=%.4f anchor_mae=%.4f crps=%.4f",
            subset_id,
            selected_sampler,
            selected_steps,
            metrics.get("mse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("anchor_mse", float("nan")),
            metrics.get("anchor_mae", float("nan")),
            metrics.get("crps", float("nan")),
        )
        return state
