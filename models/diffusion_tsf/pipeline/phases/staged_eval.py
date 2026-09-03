"""Cfg-ablation-compatible eval for chained staged coarse/fine models."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.config import visualization_settings
from models.diffusion_tsf.pipeline.data_subset import put_subset_record
from models.diffusion_tsf.pipeline.eval_bench import (
    configure as configure_eval_bench,
    dump as dump_eval_bench,
    enabled as eval_bench_enabled,
    reset as reset_eval_bench,
    span as eval_bench_span,
)
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
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _model_kwargs_from_tuned,
    _stage_best_ckpt,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import stage_state

logger = logging.getLogger(__name__)

# Standard forecast prefixes for long-horizon (stitch) eval: score the stitched
# forecast on [:h] without re-generating. Full-H metrics stay un-suffixed.
STAGED_PREFIX_HORIZONS = (96, 192, 336, 720)


def _prefix_horizons_for_length(horizon: int) -> Tuple[int, ...]:
    """Prefixes strictly shorter than H (full-H stays on unsuffixed keys)."""
    h = int(horizon)
    return tuple(p for p in STAGED_PREFIX_HORIZONS if p < h)


def _per_window_prefix_fields(
    y_true: np.ndarray,
    deterministic: np.ndarray,
    samples: np.ndarray | None,
    *,
    anchor_only: bool,
) -> Dict[str, np.ndarray]:
    """Per-window prefix metrics keyed like ``anchor_mse_h96`` / ``mse_h96``."""
    out: Dict[str, np.ndarray] = {}
    for h in _prefix_horizons_for_length(y_true.shape[-1]):
        yt = y_true[..., :h]
        det = deterministic[..., :h]
        out[f"anchor_mse_h{h}"] = per_window_anchor_mse(yt, det)
        out[f"anchor_mae_h{h}"] = _per_window_mae(yt, det)
        if not anchor_only and samples is not None:
            samp = samples[..., :h]
            smean = samp.mean(axis=2)
            out[f"mse_h{h}"] = ((yt - smean) ** 2).mean(axis=(1, 2))
            out[f"mae_h{h}"] = _per_window_mae(yt, smean)
            out[f"sample_mean_mse_h{h}"] = out[f"mse_h{h}"]
            out[f"sample_mean_mae_h{h}"] = out[f"mae_h{h}"]
            out[f"crps_h{h}"] = per_window_crps(yt, samp)
    return out


def _add_prefix_horizon_metrics(
    metrics: Dict[str, float],
    *,
    y_true: np.ndarray,
    deterministic: np.ndarray,
    samples: np.ndarray | None,
    anchor_only: bool,
) -> Dict[str, float]:
    """Attach aggregate prefix metrics (mse/mae/crps) for each standard H."""
    from models.diffusion_tsf.metrics import crps_ensemble

    for h in _prefix_horizons_for_length(y_true.shape[-1]):
        yt = y_true[..., :h]
        det = deterministic[..., :h]
        anchor = _deterministic_metrics(yt, det)
        metrics[f"anchor_mse_h{h}"] = anchor["mse"]
        metrics[f"anchor_mae_h{h}"] = anchor["mae"]
        if not anchor_only and samples is not None:
            samp = samples[..., :h]
            smean = samp.mean(axis=2)
            sm = _deterministic_metrics(yt, smean)
            metrics[f"mse_h{h}"] = sm["mse"]
            metrics[f"mae_h{h}"] = sm["mae"]
            metrics[f"sample_mean_mse_h{h}"] = sm["mse"]
            metrics[f"sample_mean_mae_h{h}"] = sm["mae"]
            metrics[f"crps_h{h}"] = float(crps_ensemble(yt, samp))
    return metrics


def _prefix_wandb_metrics(metrics: Dict[str, float], *, anchor_only: bool) -> Dict[str, float]:
    """Map ``*_h{H}`` metric keys to ``eval/staged_*_h{H}`` wandb names."""
    out: Dict[str, float] = {}
    for h in STAGED_PREFIX_HORIZONS:
        amse = metrics.get(f"anchor_mse_h{h}")
        if amse is not None:
            out[f"eval/staged_anchor_mse_h{h}"] = amse
            out[f"eval/staged_anchor_mae_h{h}"] = metrics.get(f"anchor_mae_h{h}")
        if anchor_only:
            continue
        pmse = metrics.get(f"mse_h{h}")
        if pmse is not None:
            out[f"eval/staged_prob_mse_h{h}"] = pmse
            out[f"eval/staged_prob_mae_h{h}"] = metrics.get(f"mae_h{h}")
            out[f"eval/staged_sample_mean_mse_h{h}"] = metrics.get(
                f"sample_mean_mse_h{h}", pmse,
            )
            out[f"eval/staged_sample_mean_mae_h{h}"] = metrics.get(
                f"sample_mean_mae_h{h}", metrics.get(f"mae_h{h}"),
            )
            out[f"eval/staged_crps_h{h}"] = metrics.get(f"crps_h{h}")
    return out


def _eval_artifact_tag(phase: PipelinePhase) -> str:
    stride = int(phase.require("test_stride"))
    if bool(phase.get("anchor_only", False)):
        return f"s{stride}_anchor"
    return f"s{stride}_prob"


def _eval_checkpoint_dir(state: PipelineState) -> str:
    """Return source weights for an evaluation-only run, or this run's root."""
    raw = state.extra.get("eval_source_checkpoint_dir")
    if not raw:
        return state.checkpoint_dir
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"evaluation source checkpoint root missing: {path}")
    return str(path)


def _reshape_parallel_samples(t: torch.Tensor, batch: int, n_samples: int) -> torch.Tensor:
    """``(B*S, V, ...)`` → ``(B, V, S, ...)``."""
    if t.shape[0] != batch * n_samples:
        raise ValueError(
            f"parallel sample reshape expected leading {batch * n_samples}, got {tuple(t.shape)}"
        )
    rest = t.shape[1:]
    return t.view(batch, n_samples, *rest).transpose(1, 2).contiguous()


def _eval_window_batch_size(phase: PipelinePhase, state: PipelineState) -> int:
    """Dataloader window batch from YAML. No GPU probe, no copy of train batch."""
    leftover = [
        key for key in ("probe_eval_batch_size", "probe_eval_batch_size_max")
        if key in phase.overrides
    ]
    if leftover:
        raise ValueError(
            "staged_eval keys "
            f"{leftover} were removed; set staged_eval.batch_size in YAML"
        )
    by_ds = phase.get("batch_size_by_dataset")
    by_subset = phase.get("batch_size_by_subset_id")
    yaml_bs = phase.get("batch_size")
    set_keys = [
        name
        for name, val in (
            ("batch_size", yaml_bs),
            ("batch_size_by_dataset", by_ds),
            ("batch_size_by_subset_id", by_subset),
        )
        if val is not None
    ]
    if len(set_keys) > 1:
        raise ValueError(
            "staged_eval can set only one of batch_size, "
            f"batch_size_by_dataset, batch_size_by_subset_id; got {set_keys}"
        )
    if by_subset is not None:
        if not isinstance(by_subset, dict) or not by_subset:
            raise ValueError(
                "staged_eval.batch_size_by_subset_id must be a non-empty "
                "subset_id -> int map"
            )
        subset_id = state.subset_id or state.dataset
        if subset_id not in by_subset:
            raise ValueError(
                f"staged_eval.batch_size_by_subset_id missing {subset_id!r}; "
                f"have {sorted(by_subset)}"
            )
        bs = int(by_subset[subset_id])
    elif by_ds is not None:
        if not isinstance(by_ds, dict) or not by_ds:
            raise ValueError(
                "staged_eval.batch_size_by_dataset must be a non-empty "
                "dataset -> int map"
            )
        if state.dataset not in by_ds:
            raise ValueError(
                f"staged_eval.batch_size_by_dataset missing {state.dataset!r}; "
                f"have {sorted(by_ds)}"
            )
        bs = int(by_ds[state.dataset])
    elif yaml_bs is not None:
        bs = int(yaml_bs)
    else:
        bs = 1
    if bs < 1:
        raise ValueError(f"staged_eval.batch_size must be >= 1, got {bs}")
    return bs


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
    with eval_bench_span("sklearn_gmm"):
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
    _add_prefix_horizon_metrics(
        metrics,
        y_true=y_true,
        deterministic=pack["deterministic"],
        samples=samples,
        anchor_only=False,
    )
    return metrics


def _summarize_anchor_only_metrics(pack: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Report deterministic anchor metrics without fabricating sample statistics."""
    y_true = pack["y_true"]
    det = pack["deterministic"]
    anchor = _deterministic_metrics(y_true, det)
    metrics = {
        "anchor_mse": anchor["mse"],
        "anchor_mae": anchor["mae"],
        "anchor_n_samples": 1.0,
        "metrics_profile": "anchor_only",
    }
    _add_prefix_horizon_metrics(
        metrics,
        y_true=y_true,
        deterministic=det,
        samples=None,
        anchor_only=True,
    )
    return metrics


def _load_stage_metadata(state: PipelineState, stage: str) -> Dict:
    meta = os.path.join(
        os.path.dirname(_stage_best_ckpt(state, stage, checkpoint_dir=_eval_checkpoint_dir(state))),
        "metadata.json",
    )
    if not os.path.exists(meta):
        return {}
    with open(meta) as f:
        return json.load(f)


def _stage_finetune_ckpt(
    state: PipelineState,
    stage: str,
    *,
    checkpoint_dir: str | None = None,
) -> str:
    value = {
        "coarse": state.diffusion_coarse_finetune_ckpt,
        "patch_refine": state.diffusion_patch_refine_finetune_ckpt,
    }[stage]
    if value and os.path.exists(value):
        return value
    path = _stage_best_ckpt(state, stage, checkpoint_dir=checkpoint_dir)
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


def _resolve_eval_max_windows(phase: PipelinePhase, state: PipelineState):
    by_dataset = phase.get("eval_max_windows_by_dataset") or {}
    if not by_dataset:
        return None
    if not isinstance(by_dataset, dict):
        raise ValueError("eval_max_windows_by_dataset must be a mapping")
    if state.dataset not in by_dataset:
        return None
    k = int(by_dataset[state.dataset])
    if k < 1:
        raise ValueError(
            f"eval_max_windows_by_dataset[{state.dataset!r}] must be >= 1, got {k}"
        )
    return k


def _eval_progress_dir(
    phase: PipelinePhase, state: PipelineState, subset_id: str,
) -> Path | None:
    """Job-independent resume dir: results/eval_resume/<key>/<subset_id>/."""
    key = phase.get("eval_progress_key")
    if not key:
        return None
    results_root = Path(state.results_dir).resolve()
    store = (
        results_root.parents[1]
        if results_root.parent.name == "datasets"
        else results_root.parent
    )
    return store / "eval_resume" / str(key) / str(subset_id)


def _load_eval_progress(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    jsonl = path / "windows.jsonl"
    records: List[Dict[str, Any]] = []
    done: set = set()
    if not jsonl.is_file():
        return records, done
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            wi = int(rec["window_index"])
            if wi in done:
                continue
            done.add(wi)
            records.append(rec)
    return records, done


def _metrics_from_progress_records(
    records: Sequence[Dict[str, Any]], *, anchor_only: bool,
) -> Dict[str, float]:
    if not records:
        raise ValueError("no eval progress records to summarize")

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    out: Dict[str, float] = {
        "anchor_mse": _mean("anchor_mse"),
        "anchor_mae": _mean("anchor_mae"),
        "anchor_n_samples": 1.0,
        "n_windows": float(len(records)),
    }
    for h in STAGED_PREFIX_HORIZONS:
        for key in (f"anchor_mse_h{h}", f"anchor_mae_h{h}"):
            if any(key in r for r in records):
                out[key] = _mean(key)
    if anchor_only:
        out["metrics_profile"] = "anchor_only"
        return out
    mse = _mean("mse")
    mae = _mean("mae")
    out.update({
        "mse": mse,
        "mae": mae,
        "sample_mean_mse": _mean("sample_mean_mse") if any(
            "sample_mean_mse" in r for r in records
        ) else mse,
        "sample_mean_mae": _mean("sample_mean_mae") if any(
            "sample_mean_mae" in r for r in records
        ) else mae,
        "crps": _mean("crps"),
        "metrics_profile": "dpmpp_prob_core_plus_anchor",
    })
    # Probabilistic prefix keys are optional (older progress jsonl may lack them).
    for h in STAGED_PREFIX_HORIZONS:
        for key in (
            f"mse_h{h}",
            f"mae_h{h}",
            f"sample_mean_mse_h{h}",
            f"sample_mean_mae_h{h}",
            f"crps_h{h}",
        ):
            if any(key in r for r in records):
                out[key] = _mean(key)
    return out


def _write_eval_progress_summary(
    path: Path,
    records: Sequence[Dict[str, Any]],
    *,
    n_planned: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    metrics = (
        _metrics_from_progress_records(
            records, anchor_only=bool(extra.get("anchor_only", False)),
        )
        if records else {}
    )
    payload = {
        "n_planned": int(n_planned),
        "n_done": len(records),
        "complete": bool(n_planned > 0 and len(records) >= int(n_planned)),
        **metrics,
        **{k: v for k, v in extra.items() if k != "anchor_only"},
    }
    tmp = path / "summary.json.tmp"
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path / "summary.json")
    return payload


def _per_window_mae(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - pred).mean(axis=(1, 2))


def _horizon_stitch_enabled(model) -> bool:
    return bool(getattr(model.config, "horizon_stitch", False))


def _staged_generate_once(
    *,
    coarse_model,
    fine_model,
    past: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    coarse_out = coarse_model.generate(past, **gen_kwargs)
    fine_out = fine_model.generate(
        past,
        future_coarse_2d=coarse_out["future_2d_coarse"],
        **gen_kwargs,
    )
    pred = _staged_anchor_global_norm(fine_model, coarse_out, fine_out)
    pred_t = torch.from_numpy(pred).to(past.device)
    return {"coarse": coarse_out, "fine": fine_out, "prediction": pred_t}


def _staged_generate_horizon_stitch(
    *,
    coarse_model,
    fine_model,
    past: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Independent 104-canvas generates + overlap-average stitch to length H.

    Chunks run sequentially (not ``past.repeat_interleave(n_chunks)``) so peak
    rows stay ``batch`` — same as single-canvas eval. Materialising all 8
    chunks at once with MC samples blew past the L40S probed row budget
    (``batch × n_samp × n_chunks``).
    """
    from models.diffusion_tsf.horizon_chunks import chunk_starts, overlap_average_stitch

    k = int(getattr(coarse_model.config, "lookback_overlap", 0))
    inner = int(getattr(coarse_model.config, "horizon_chunk_inner", 96))
    dataset_h = int(getattr(coarse_model.config, "dataset_forecast_length", 0) or 0)
    if dataset_h < inner:
        raise ValueError(
            f"horizon_stitch dataset_forecast_length {dataset_h} < inner {inner}"
        )
    starts = chunk_starts(dataset_h, inner=inner, overlap=k)
    n_chunks = len(starts)
    batch = past.shape[0]
    device = past.device
    expected_w = k + inner
    canvases = []
    diag0 = None
    for start in starts:
        t0 = torch.full((batch,), int(start), device=device, dtype=torch.long)
        chunk_kwargs = dict(gen_kwargs)
        chunk_kwargs["horizon_chunk_t0"] = t0
        out = _staged_generate_once(
            coarse_model=coarse_model,
            fine_model=fine_model,
            past=past,
            gen_kwargs=chunk_kwargs,
        )
        canvas = out["fine"]["prediction_with_overlap"]
        if canvas.shape[-1] != expected_w:
            raise ValueError(
                f"stitch canvas width {canvas.shape[-1]} != overlap+inner {expected_w}"
            )
        canvases.append(canvas)
        if diag0 is None:
            diag0 = {"coarse": out["coarse"], "fine": out["fine"]}
    canvas_bn = torch.stack(canvases, dim=1)
    if canvas_bn.shape[1] != n_chunks:
        raise ValueError(f"stacked n_chunks {canvas_bn.shape[1]} != {n_chunks}")
    stitched = overlap_average_stitch(
        canvas_bn, starts, horizon=dataset_h, inner=inner, overlap=k,
    )
    return stitched, diag0


def _staged_det_gen_kwargs(state: PipelineState, default_steps: int) -> Dict[str, Any]:
    sampler = str(getattr(state, "eval_sampler", "anchor"))
    if sampler in ("anchor", "deterministic_anchor"):
        return {"sampler": sampler}
    return {"sampler": sampler, "num_inference_steps": int(default_steps)}


class StagedEvalPhase(PipelinePhase):
    name = "staged_eval"

    def should_skip(self, state: PipelineState) -> bool:
        if bool(self.get("refresh_eval_visualizations", False)):
            logger.info("  [%s] forcing eval refresh for visualizations", self.name)
            return False
        subset_id = state.subset_id or state.dataset
        tag = _eval_artifact_tag(self)
        partial = os.path.join(
            state.results_dir, "partials", f"{state.dataset}_staged_{tag}.json",
        )
        nested = os.path.join(state.results_dir, subset_id, f"staged_results_{tag}.json")
        raw_dir = os.path.join(state.results_dir, "raw")
        anchor_npz = os.path.join(raw_dir, f"staged_anchor_samples_{state.dataset}_{tag}.npz")
        samples_npz = os.path.join(raw_dir, f"staged_dpmpp_samples_{state.dataset}_{tag}.npz")
        full_npz = os.path.join(raw_dir, f"staged_anchor_{state.dataset}_{tag}.npz")
        worst_json = os.path.join(state.results_dir, subset_id, f"worst_windows_{tag}.json")
        if os.path.exists(partial) and os.path.exists(nested):
            try:
                with open(partial) as f:
                    metrics = json.load(f)
                core_ok = "crps" in metrics and "top3_mse" in metrics
                anchor_ok = "anchor_mse" in metrics and "anchor_mae" in metrics
                sample_mean_ok = "sample_mean_mse" in metrics and "sample_mean_mae" in metrics
                raw_ok = os.path.exists(anchor_npz)
                if not bool(self.get("anchor_only", False)):
                    raw_ok = raw_ok and os.path.exists(samples_npz)
                    if not (core_ok and sample_mean_ok):
                        raw_ok = False
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
            skip_ok = anchor_ok and raw_ok and diag_ok and worst_ok and sampler_ok
            if not bool(self.get("anchor_only", False)):
                skip_ok = skip_ok and core_ok and sample_mean_ok
            if skip_ok:
                logger.info("  [%s] already evaluated (%s): %s", self.name, tag, partial)
                return True
            logger.info("  [%s] re-evaluating %s to add missing metrics: %s", self.name, tag, partial)
        return False

    def _load_model(self, state: PipelineState, stage: str, itrans_guidance, n_iv: int, device: torch.device):
        from models.diffusion_tsf.train_multivariate_pipeline import (
            anchor_kwargs_from_params,
            create_diffusion_model,
            dataset_window_lengths,
            load_diffusion_state_keep_attached_guidance,
        )
        model_state = stage_state(state, stage, honor_dataset_windows=True)
        ds_lb, ds_hz = dataset_window_lengths(state, state.dataset)
        meta = _load_stage_metadata(state, stage)
        tuned = meta.get("tuned_params") or {}
        model_kwargs = anchor_kwargs_from_params(model_state, tuned)
        model_kwargs.update(_model_kwargs_from_tuned(tuned))
        model = create_diffusion_model(
            model_state,
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=stage,
            ordinal_ladder=state.ordinal_ladder,
            **model_kwargs,
        ).to(device)
        ckpt = torch.load(
            _stage_finetune_ckpt(state, stage, checkpoint_dir=_eval_checkpoint_dir(state)),
            map_location=device,
            weights_only=False,
        )
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
        gmm_components: int,
        topk_max: int,
        window_indices: Sequence[int],
        test_stride: int,
        anchor_only: bool = False,
        progress_dir: Path | None = None,
        prior_records: Sequence[Dict[str, Any]] | None = None,
        n_planned: int | None = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        if not anchor_only and prob_sampler in {"anchor", "deterministic_anchor"}:
            raise ValueError("staged probabilistic eval must use a regular sampler, not anchor.")
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}
        det_kwargs = _staged_det_gen_kwargs(state, prob_steps)
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
        for m in (coarse_model, fine_model):
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
        progress_records: List[Dict[str, Any]] = [dict(r) for r in (prior_records or ())]
        progress_fh = None
        planned = int(n_planned) if n_planned is not None else (
            len(progress_records) + len(loader.dataset)
        )
        if progress_dir is not None:
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_fh = (progress_dir / "windows.jsonl").open("a")
        keep_pack = progress_dir is None
        try:
            with torch.no_grad():
                for batch_idx, (past, future) in enumerate(loader):
                    if eval_bench_enabled():
                        reset_eval_bench()
                    with eval_bench_span("to_device"):
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
                    with eval_bench_span("det"):
                        if _horizon_stitch_enabled(coarse_model):
                            det_t, stitch_diag = _staged_generate_horizon_stitch(
                                coarse_model=coarse_model,
                                fine_model=fine_model,
                                past=past,
                                gen_kwargs=det_kwargs,
                            )
                            det_all.append(det_t.detach().cpu().numpy())
                            with eval_bench_span("decode_components"):
                                coarse_np, fine_np, _ = decode_staged_anchor_components(
                                    fine_model, stitch_diag["coarse"], stitch_diag["fine"],
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
                            with eval_bench_span("decode_components"):
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
                    prob_s = 0.0
                    if not anchor_only:
                        prob_t0 = time.perf_counter()
                        with eval_bench_span("prob"):
                            # Expand window batch across independent MC samples so unique-seg
                            # AR (and other generate paths) fill the GPU in one forward chain.
                            torch.manual_seed(state.seed + batch_idx * 1009)
                            with eval_bench_span("mc_expand"):
                                past_exp = past.repeat_interleave(prob_samples, dim=0)
                            if _horizon_stitch_enabled(coarse_model):
                                sample_t, _ = _staged_generate_horizon_stitch(
                                    coarse_model=coarse_model,
                                    fine_model=fine_model,
                                    past=past_exp,
                                    gen_kwargs=prob_kwargs,
                                )
                                with eval_bench_span("reshape_cpu"):
                                    samples_bvs = _reshape_parallel_samples(sample_t, batch_n, prob_samples)
                                    sample_all.append(samples_bvs.detach().cpu().numpy())
                            else:
                                coarse_sample = coarse_model.generate(past_exp, **prob_kwargs)
                                fine_sample = fine_model.generate(
                                    past_exp,
                                    future_coarse_2d=coarse_sample["future_2d_coarse"],
                                    **prob_kwargs,
                                )
                                with eval_bench_span("reshape_cpu"):
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
                        "det=%.1fs prob=%.1fs (n_samp=%d%s) batch=%.1fs "
                        "elapsed=%.1fs eta=%.1fs",
                        subset_id,
                        done,
                        len(loader),
                        batch_n,
                        det_s,
                        prob_s,
                        prob_samples,
                        " anchor-only" if anchor_only else " parallel",
                        batch_s,
                        elapsed,
                        eta_s,
                    )
                    if eval_bench_enabled():
                        dump_eval_bench(
                            logger,
                            title=(
                                f"[{subset_id}] batch {done}/{len(loader)} "
                                f"n={batch_n} n_samp={prob_samples} steps={prob_steps}"
                            ),
                        )
                    if progress_dir is not None:
                        y_np = y_true_all[-1]
                        det_np = det_all[-1]
                        amse = per_window_anchor_mse(y_np, det_np)
                        amae = _per_window_mae(y_np, det_np)
                        samp_np = None if anchor_only else sample_all[-1]
                        if not anchor_only:
                            smean = samp_np.mean(axis=2)
                            pmse = ((y_np - smean) ** 2).mean(axis=(1, 2))
                            pmae = _per_window_mae(y_np, smean)
                            crps = per_window_crps(y_np, samp_np)
                        prefix_fields = _per_window_prefix_fields(
                            y_np, det_np, samp_np, anchor_only=anchor_only,
                        )
                        for j, wi in enumerate(batch_window_indices):
                            rec = {
                                "window_index": int(wi),
                                "anchor_mse": float(amse[j]),
                                "anchor_mae": float(amae[j]),
                            }
                            if not anchor_only:
                                rec.update({
                                    "mse": float(pmse[j]),
                                    "mae": float(pmae[j]),
                                    "sample_mean_mse": float(pmse[j]),
                                    "sample_mean_mae": float(pmae[j]),
                                    "crps": float(crps[j]),
                                })
                            for key, arr in prefix_fields.items():
                                rec[key] = float(arr[j])
                            progress_records.append(rec)
                            if progress_fh is not None:
                                progress_fh.write(json.dumps(rec) + "\n")
                        if progress_fh is not None:
                            progress_fh.flush()
                        summary = _write_eval_progress_summary(
                            progress_dir,
                            progress_records,
                            n_planned=planned,
                            extra={"anchor_only": anchor_only, "subset_id": subset_id},
                        )
                        logger.info(
                            "[%s] eval progress %d/%d anchor_mse=%.4f %s",
                            subset_id,
                            summary.get("n_done", len(progress_records)),
                            planned,
                            summary.get("anchor_mse", float("nan")),
                            "" if anchor_only else f"prob_mse={summary.get('mse', float('nan')):.4f} crps={summary.get('crps', float('nan')):.4f}",
                        )
                        wandb_utils.log_eval_metrics({
                            "eval/progress_n_done": float(summary.get("n_done", 0)),
                            "eval/progress_n_planned": float(planned),
                            "eval/staged_anchor_mse": summary.get("anchor_mse"),
                            **({} if anchor_only else {
                                "eval/staged_prob_mse": summary.get("mse"),
                                "eval/staged_crps": summary.get("crps"),
                            }),
                            **_prefix_wandb_metrics(summary, anchor_only=anchor_only),
                        }, step=int(summary.get("n_done", 0)))
        finally:
            if progress_fh is not None:
                progress_fh.close()

        if progress_dir is not None:
            metrics = _metrics_from_progress_records(
                progress_records, anchor_only=anchor_only,
            )
            return metrics, {}

        pack = {
            "y_true": np.concatenate(y_true_all, axis=0),
            "deterministic": np.concatenate(det_all, axis=0),
            "coarse_anchor": np.concatenate(coarse_all, axis=0),
            "fine_anchor": np.concatenate(fine_all, axis=0),
            "final_anchor": np.concatenate(det_all, axis=0),
            "window_indices": np.array(window_idx_all, dtype=np.int64),
            "series_starts": np.array(window_idx_all, dtype=np.int64) * int(test_stride),
        }
        if det_with_overlap_all and len(det_with_overlap_all) == len(det_all):
            pack["y_true_with_overlap"] = np.concatenate(y_true_with_overlap_all, axis=0)
            pack["deterministic_with_overlap"] = np.concatenate(det_with_overlap_all, axis=0)
            pack["final_anchor_with_overlap"] = pack["deterministic_with_overlap"]
        if not anchor_only:
            pack["samples"] = np.concatenate(sample_all, axis=0)
        if samples_with_overlap_all and len(samples_with_overlap_all) == len(sample_all):
            pack["samples_with_overlap"] = np.concatenate(samples_with_overlap_all, axis=0)
            pack["sample_mean_with_overlap"] = pack["samples_with_overlap"].mean(axis=2)
        if anchor_only:
            metrics = _summarize_anchor_only_metrics(pack)
        else:
            pack["sample_mean"] = pack["samples"].mean(axis=2)
            if eval_bench_enabled():
                reset_eval_bench()
            metrics = _summarize_staged_eval_metrics(
                pack,
                gmm_components=gmm_components,
                seed=state.seed,
                topk_max=topk_max,
            )
            if eval_bench_enabled():
                dump_eval_bench(logger, title="sklearn_gmm_metrics")
        return metrics, pack

    def execute(self, state: PipelineState) -> PipelineState:
        if not getattr(state, "use_patch_refine_stage", False):
            raise ValueError(
                "use_patch_refine_stage must be true; residual fine-as-primary eval was removed"
            )
        from models.diffusion_tsf.train_multivariate_pipeline import (
            generate_dataset_job,
            load_dataset,
            load_wrapped_guidance,
            dataset_window_lengths,
        )

        device = state.resolve_device()
        subset_id = state.subset_id or state.dataset
        bench = configure_eval_bench(bool(getattr(state, "eval_bench", False)))
        if bench:
            logger.info(
                "[%s] eval-bench on (TS_EVAL_BENCH / --eval-bench); skipping viz and diagnostics",
                subset_id,
            )
        anchor_only = bool(self.get("anchor_only", False))
        gmm_components = int(self.require("gmm_components"))
        topk_max = int(self.require("topk_max"))
        variate_indices = state.variate_indices
        if not variate_indices:
            raise ValueError(
                f"[{self.name}] Missing resolved variate_indices in state for dataset {state.dataset!r}. "
                "Data subset policy must be resolved before running phase."
            )
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

        full_train_ds, full_val_ds, full_test_ds, norm_stats = load_dataset(
            state, state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
            ordinal_tie_atol=float(state.ordinal_tie_atol),
            use_ordinal_window_norm=state.use_ordinal_window_norm,
        )
        if norm_stats.get("ordinal_ladder") is not None:
            state.ordinal_ladder = norm_stats["ordinal_ladder"]

        source_checkpoint_dir = _eval_checkpoint_dir(state)
        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = os.path.join(
                source_checkpoint_dir, f"{subset_id}_itransformer_finetuned.pt"
            )
        needs_guidance = state.needs_guidance
        if needs_guidance and not os.path.exists(ft_guidance_ckpt):
            raise FileNotFoundError(f"Missing finetuned guidance checkpoint: {ft_guidance_ckpt}")
        if not needs_guidance:
            ft_guidance_ckpt = ""

        ds_lb, ds_hz = dataset_window_lengths(state, state.dataset)
        guidance = None
        if needs_guidance:
            guidance = load_wrapped_guidance(
                state, ft_guidance_ckpt,
                n_iv,
                device,
                guidance_type=state.guidance_type,
                dataset_lookback=ds_lb,
                dataset_horizon=ds_hz,
            )
        coarse_model = self._load_model(state, "coarse", guidance, n_iv, device)
        fine_model = self._load_model(state, "patch_refine", guidance, n_iv, device)

        batch_size = _eval_window_batch_size(self, state)
        if state.smoke_test:
            final_ds = Subset(full_test_ds, list(range(min(2, len(full_test_ds)))))
            prob_samples = 1
            default_steps = 5
        else:
            eval_fraction = _resolve_eval_test_fraction(self, state)
            eval_k = _resolve_eval_max_windows(self, state)
            if eval_k is not None and eval_fraction < 1.0:
                raise ValueError(
                    f"{subset_id}: eval_max_windows_by_dataset and "
                    "eval_test_fraction<1 cannot both apply"
                )
            if eval_fraction < 1.0:
                final_ds = _fraction_subset(full_test_ds, eval_fraction, state.seed)
                logger.info(
                    "[%s] eval subset: %d/%d windows (eval_test_fraction=%.3f)",
                    subset_id,
                    len(final_ds),
                    len(full_test_ds),
                    eval_fraction,
                )
            elif eval_k is not None:
                from models.diffusion_tsf.pipeline.data_subset import random_window_subset
                final_ds = random_window_subset(
                    full_test_ds,
                    eval_k,
                    int(state.seed),
                    label=f"{subset_id}/eval",
                )
            else:
                final_ds = full_test_ds
            prob_samples = 0 if anchor_only else int(self.require("probabilistic_n_samples"))
            default_steps = 1 if anchor_only else int(self.require("probabilistic_num_inference_steps"))

        max_windows = getattr(state, "eval_max_windows", None)
        if max_windows is not None:
            n_keep = min(int(max_windows), len(final_ds))
            if isinstance(final_ds, Subset):
                final_ds = Subset(full_test_ds, list(final_ds.indices[:n_keep]))
            else:
                final_ds = Subset(final_ds, list(range(n_keep)))
            logger.info("[%s] eval_max_windows=%d -> %d windows", subset_id, int(max_windows), n_keep)
        max_steps = getattr(state, "eval_max_steps", None)
        if max_steps is not None:
            default_steps = int(max_steps)
            logger.info("[%s] eval_max_steps=%d", subset_id, default_steps)
        if bench:
            batch_size = 1

        if isinstance(final_ds, Subset):
            eval_window_indices = [int(i) for i in final_ds.indices]
        else:
            eval_window_indices = list(range(len(final_ds)))
        planned_indices = list(eval_window_indices)
        progress_dir = _eval_progress_dir(self, state, subset_id)
        progress_records: List[Dict[str, Any]] = []
        if progress_dir is not None:
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_records, done = _load_eval_progress(progress_dir)
            remaining = [i for i in planned_indices if i not in done]
            logger.info(
                "[%s] eval progress resume: %d/%d saved, %d remaining (%s)",
                subset_id,
                len(progress_records),
                len(planned_indices),
                len(remaining),
                progress_dir,
            )
            eval_window_indices = remaining
            final_ds = Subset(full_test_ds, remaining)

        if bench:
            logger.info("[%s] skipping eval diagnostics (eval-bench)", subset_id)
        else:
            try:
                from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics

                diagnostic_stages = [
                    (
                        "coarse", coarse_model,
                        _stage_finetune_ckpt(state, "coarse", checkpoint_dir=source_checkpoint_dir),
                        None,
                    ),
                    (
                        "patch_refine",
                        fine_model,
                        _stage_finetune_ckpt(state, "patch_refine", checkpoint_dir=source_checkpoint_dir),
                        _stage_finetune_ckpt(state, "coarse", checkpoint_dir=source_checkpoint_dir),
                    ),
                ]
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
                    state, state.dataset,
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
                        include_phase_start=(eval_stage == "coarse"),
                    )
                    wandb_utils.log_phase_diagnostics_result(diag)
            except Exception as e:
                logger.warning("[%s] eval diagnostics failed: %s", self.name, e, exc_info=True)

        sampler_tuning = []
        selected_sampler = "anchor" if anchor_only else str(self.require("probabilistic_sampler"))
        if not anchor_only and selected_sampler in {"anchor", "deterministic_anchor"}:
            raise ValueError(
                "staged probabilistic_sampler must be ddim, quad_t, or ddim_quad, not anchor."
            )
        selected_steps = default_steps
        if not anchor_only and bool(self.require("tune_sampler")) and not state.smoke_test and not bench:
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

        if progress_dir is not None and not eval_window_indices:
            metrics = _metrics_from_progress_records(
                progress_records, anchor_only=anchor_only,
            )
            pack: Dict[str, np.ndarray] = {}
        else:
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
                gmm_components=gmm_components,
                topk_max=topk_max,
                window_indices=eval_window_indices,
                test_stride=test_stride,
                anchor_only=anchor_only,
                progress_dir=progress_dir,
                prior_records=progress_records,
                n_planned=len(planned_indices),
            )
        metrics.update({
            "sampler_tuned": bool(sampler_tuning),
            "selected_probabilistic_sampler": selected_sampler,
            "selected_probabilistic_num_inference_steps": selected_steps,
        })

        from models.diffusion_tsf.pipeline.phase_diagnostics import select_spaced_top_k

        worst_manifest: List[Dict[str, Any]] = []
        has_pack = bool(pack.get("y_true") is not None and len(pack.get("y_true", [])) > 0)
        if has_pack:
            anchor_scores = per_window_anchor_mse(pack["y_true"], pack["final_anchor"])
            series_starts = pack["series_starts"]
            window_indices_arr = pack["window_indices"]
            worst_metrics = [("anchor_mse", anchor_scores)]
            if not anchor_only:
                worst_metrics.insert(0, ("crps", per_window_crps(pack["y_true"], pack["samples"])))
            for metric_name, scores in worst_metrics:
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
        tag = _eval_artifact_tag(self)
        with open(os.path.join(nested_dir, f"worst_windows_{tag}.json"), "w") as f:
            json.dump(worst_manifest, f, indent=2)
        with open(os.path.join(partial_dir, f"{state.dataset}_staged_{tag}.json"), "w") as f:
            payload = dict(metrics)
            payload["seed"] = int(state.seed)
            payload["binary_length_mode"] = getattr(state, "binary_length_mode", "none")
            payload["binary_length_g"] = float(getattr(state, "binary_length_g", 1.0))
            payload["eval_artifact_tag"] = tag
            payload["test_stride"] = int(test_stride)
            json.dump(payload, f, indent=2, sort_keys=True)
        if has_pack:
            np.savez_compressed(os.path.join(raw_dir, f"staged_anchor_{state.dataset}_{tag}.npz"), **pack)
            np.savez_compressed(
                os.path.join(raw_dir, f"staged_anchor_samples_{state.dataset}_{tag}.npz"),
                y_true=pack["y_true"],
                anchor=pack["deterministic"],
            )
            if not anchor_only:
                np.savez_compressed(
                    os.path.join(raw_dir, f"staged_dpmpp_samples_{state.dataset}_{tag}.npz"),
                    y_true=pack["y_true"],
                    samples=pack["samples"],
                    sample_mean=pack["sample_mean"],
                )
        eval_payload = {
            "dataset": state.dataset,
            "subset_id": subset_id,
            "seed": int(state.seed),
            "binary_length_mode": getattr(state, "binary_length_mode", "none"),
            "binary_length_g": float(getattr(state, "binary_length_g", 1.0)),
            "variate_indices": variate_indices,
            "sampler_tuning": sampler_tuning,
            "eval_artifact_tag": tag,
            "test_stride": int(test_stride),
            "eval_metrics": {"staged_anchor": metrics},
        }
        put_subset_record(eval_payload, state.dataset, subset_meta)
        with open(os.path.join(nested_dir, f"staged_results_{tag}.json"), "w") as f:
            json.dump(eval_payload, f, indent=2, sort_keys=True)

        wandb_metrics = {
            "eval/test_stride": int(test_stride),
        }
        if anchor_only:
            wandb_metrics.update({
                "eval/staged_anchor_mse": metrics.get("anchor_mse"),
                "eval/staged_anchor_mae": metrics.get("anchor_mae"),
            })
        else:
            wandb_metrics.update({
                "eval/staged_prob_mse": metrics.get("mse"),
                "eval/staged_prob_mae": metrics.get("mae"),
                "eval/staged_sample_mean_mse": metrics.get("sample_mean_mse"),
                "eval/staged_sample_mean_mae": metrics.get("sample_mean_mae"),
                "eval/staged_crps": metrics.get("crps"),
                "eval/staged_top1_mse": metrics.get("top1_mse"),
                "eval/staged_top3_mse": metrics.get("top3_mse"),
                "eval/selected_sampler": selected_sampler,
                "eval/selected_steps": selected_steps,
            })
        wandb_metrics.update(_prefix_wandb_metrics(metrics, anchor_only=anchor_only))
        wandb_metrics.update({
            f"eval/{tag}/staged_anchor_mse": metrics.get("anchor_mse"),
            f"eval/{tag}/staged_anchor_mae": metrics.get("anchor_mae"),
        })
        wandb_utils.log_eval_metrics(wandb_metrics, step=int(test_stride))

        skip_viz = bool(
            self.get("skip_eval_visualizations", False)
            or state.extra.get("skip_eval_visualizations", False)
            or bench
        )
        viz_cfg = visualization_settings(state.merged_config)
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

            # Pack-native anchor + probabilistic mean/band panels (same shape as MMPD viz).
            try:
                from utils.mmpd_sample_viz import generate_mmpd_sample_visualizations

                pack_viz_dir = os.path.join(state.results_dir, "viz", "binary_anchor_prob", state.dataset)
                pack_viz = generate_mmpd_sample_visualizations(
                    pack,
                    dataset=state.dataset,
                    out_dir=Path(pack_viz_dir),
                    model_label="binary",
                    n_windows=4 if not state.smoke_test else 1,
                    seed=int(state.seed),
                )
                wandb_utils.log_visualization_paths(
                    pack_viz, wandb_key="eval/binary_anchor_prob_samples",
                )
            except Exception as e:
                logger.warning("Binary pack anchor+prob viz failed: %s", e, exc_info=True)

        # Patch-box / 1d / 2d panels: same skip_eval_visualizations bypass as MMPD
        # gap/redbox. YAML viz_binary_mmpd_redbox is a no-op without MMPD packs;
        # viz_patch_boxes writes refine_boxes from already-loaded coarse/fine.
        if (not bench) and bool(viz_cfg.get("viz_patch_boxes", False)):
            try:
                from utils.staged_eval_sample_viz import (
                    pick_indices,
                    write_staged_sample_panels,
                )

                kind = "patch_refine"
                n_box = int(viz_cfg.get("viz_patch_boxes_n_samples", 1) or 1)
                if state.smoke_test:
                    n_box = min(n_box, 1)
                picks = pick_indices(len(full_test_ds), n_box, int(state.seed), None)
                box_dir = Path(state.results_dir) / "viz" / "staged_eval_samples" / subset_id
                box_paths = write_staged_sample_panels(
                    out_dir=box_dir,
                    run_name=str(subset_id),
                    dataset=str(state.dataset),
                    kind=kind,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    pool=full_test_ds,
                    picks=picks,
                    device=device,
                    sampler=str(selected_sampler),
                    num_sampling_steps=int(selected_steps),
                    seed=int(state.seed),
                    variables_to_plot=int(viz_cfg.get("n_dual_scale_vars", 0) or 0),
                    jpeg_dpi=int(viz_cfg.get("jpeg_dpi", 100) or 100),
                )
                wandb_utils.log_visualization_paths(
                    [str(p) for p in box_paths],
                    wandb_key="eval/staged_eval_sample_panels",
                )
                logger.info(
                    "[%s] viz_patch_boxes wrote %d panels under %s",
                    subset_id, len(box_paths), box_dir,
                )
            except Exception as e:
                logger.warning("Staged patch-box sample viz failed: %s", e, exc_info=True)

        # Point-acc gap/redbox: default ON; runs even when skip_eval_visualizations
        # (earlyjuly leaves). Gated only by viz_binary_mmpd_{gap,redbox} + campaign path.
        # eval-bench skips this too so viz cannot hang the timing run.
        if (not bench) and (
            bool(viz_cfg.get("viz_binary_mmpd_gap", True))
            or bool(viz_cfg.get("viz_binary_mmpd_redbox", True))
        ):
            from utils.staged_point_gap_redbox_viz import (
                run_binary_mmpd_gap_and_redbox_viz,
            )

            merged = state.merged_config or {}
            cfg_path = merged.get("_yaml_path") or state.extra.get("config_path")
            gap_paths = run_binary_mmpd_gap_and_redbox_viz(
                state=state,
                pack=pack,
                coarse_model=coarse_model,
                fine_model=fine_model,
                device=device,
                viz_cfg=viz_cfg,
                patch_refine=True,
                joint_dual=False,
                pack_test_stride=int(test_stride),
                binary_config_path=str(cfg_path) if cfg_path else None,
            )
            wandb_utils.log_visualization_paths(
                gap_paths.get("gap", []),
                wandb_key="eval/point_gap_binary_mmpd",
            )
            wandb_utils.log_visualization_paths(
                gap_paths.get("redbox", []),
                wandb_key="eval/point_gap_redbox",
            )

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
