#!/usr/bin/env python3
"""ETTh1 canvas128 probe: coarse=anchor, fine=quad_t (3-sample mean).

Loads an existing canvas128 ckpt (default job 4571065), runs point-acc eval
with:
  - coarse stage: sampler=anchor (deterministic)
  - fine/patch_refine: conditioned on that coarse ``future_2d_coarse``;
    probabilistic ``quad_t`` with ``n_samples`` draws; final pred = sample mean

Stride knob (fail-fast documented):
  regular canvas128 staged_eval ``test_stride`` is 16 (earlyjuly leaf).
  25% of that density ⇒ ``eval_test_stride = regular_stride * 4`` (default 64).
  Do NOT confuse with MMPD pack lattice stride 4.

Reports: sample_mean MAE/MSE, CRPS (from the n samples), and pure-anchor
MAE/MSE for the same windows.

Example:
  python temp/scripts/eval_etth1_coarse_anchor_fine_prob_avg3.py --smoke-test
  python temp/scripts/eval_etth1_coarse_anchor_fine_prob_avg3.py \\
      --ckpt results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CFG = "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
DEFAULT_CKPT_GLOB = "*-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6"
JOBID_HINT = 4571065
# earlyjuly / canvas128 staged_eval test_stride
REGULAR_TEST_STRIDE = 16
# 25% window density relative to regular → 4× stride
DEFAULT_EVAL_STRIDE = REGULAR_TEST_STRIDE * 4  # 64


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=3, help="Fine-stage MC samples to average.")
    p.add_argument("--prob-sampler", default="quad_t")
    p.add_argument("--prob-steps", type=int, default=20)
    p.add_argument(
        "--eval-test-stride",
        type=int,
        default=DEFAULT_EVAL_STRIDE,
        help=(
            f"Test window stride. Default {DEFAULT_EVAL_STRIDE} = "
            f"{REGULAR_TEST_STRIDE}*4 (25%% of regular canvas128 density). "
            "Fail-fast if < regular or not a multiple."
        ),
    )
    p.add_argument(
        "--regular-test-stride",
        type=int,
        default=REGULAR_TEST_STRIDE,
        help="Baseline staged_eval stride for the 25%% density check.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gmm-components", type=int, default=10)
    p.add_argument("--topk-max", type=int, default=3)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/datasets/<stamp>-etth1-coarse-anchor-fine-prob-avg3",
    )
    return p.parse_args(argv)


def _discover_ckpt(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        ckpt = Path(explicit)
        if not ckpt.is_dir():
            raise FileNotFoundError(f"--ckpt not a directory: {ckpt}")
        return ckpt.resolve()
    base = REPO_ROOT / "results" / "ckpts"
    hits = sorted(base.glob(f"*{JOBID_HINT}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        hits = sorted(base.glob(DEFAULT_CKPT_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(
            f"No ETTh1 canvas128 ckpt under {base} (hint job {JOBID_HINT})"
        )
    return hits[0].resolve()


def _assert_stride_policy(eval_stride: int, regular: int) -> None:
    if eval_stride < regular:
        raise ValueError(
            f"eval_test_stride={eval_stride} < regular_test_stride={regular}; "
            "25% density requires a *larger* stride (fewer windows)"
        )
    if eval_stride % regular != 0:
        raise ValueError(
            f"eval_test_stride={eval_stride} must be a multiple of "
            f"regular_test_stride={regular} (got remainder {eval_stride % regular})"
        )
    factor = eval_stride // regular
    density = 1.0 / float(factor)
    if abs(density - 0.25) > 1e-9 and factor != 4:
        # Allow non-25% only if user explicitly chose a different multiple —
        # still document the effective density.
        print(
            f"[stride] WARN density={density:.4f} (factor={factor}); "
            f"default 25% uses factor=4 → stride={regular * 4}",
            flush=True,
        )
    print(
        f"[stride] regular={regular} eval={eval_stride} "
        f"(factor={factor}, density≈{density:.2%} of regular windows)",
        flush=True,
    )


def _reshape_samples(pred: torch.Tensor, batch_n: int, n_samples: int) -> torch.Tensor:
    # pred: (B*S, V, H) → (B, V, S, H)
    if pred.dim() != 3:
        raise ValueError(f"expected (B*S,V,H), got {tuple(pred.shape)}")
    if pred.shape[0] != batch_n * n_samples:
        raise ValueError(
            f"leading {pred.shape[0]} != batch_n*{n_samples}={batch_n * n_samples}"
        )
    b, v, h = batch_n, pred.shape[1], pred.shape[2]
    return pred.view(b, n_samples, v, h).permute(0, 2, 1, 3).contiguous()


@torch.no_grad()
def run_eval(
    *,
    coarse_model: Any,
    fine_model: Any,
    loader: DataLoader,
    window_indices: Sequence[int],
    device: torch.device,
    seed: int,
    n_samples: int,
    prob_sampler: str,
    prob_steps: int,
    test_stride: int,
) -> Dict[str, np.ndarray]:
    if prob_sampler in {"anchor", "deterministic_anchor"}:
        raise ValueError("fine-stage sampler must be probabilistic (e.g. quad_t), not anchor")
    det_kwargs = {"sampler": "anchor", "num_inference_steps": 1}
    prob_kwargs = {"sampler": str(prob_sampler), "num_inference_steps": int(prob_steps)}

    y_true_all: List[np.ndarray] = []
    anchor_all: List[np.ndarray] = []
    sample_all: List[np.ndarray] = []
    window_idx_all: List[int] = []
    t0 = time.perf_counter()

    for batch_idx, (past, future) in enumerate(loader):
        past = past.to(device)
        future = future.to(device)
        batch_n = int(past.shape[0])
        batch_start = batch_idx * int(loader.batch_size)
        batch_wis = list(window_indices[batch_start : batch_start + batch_n])
        window_idx_all.extend(int(i) for i in batch_wis)

        K = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
        if K > 0:
            future_core = future[..., K:]
        else:
            future_core = future
        y_true_all.append(future_core.detach().cpu().numpy())

        # --- pure anchor (coarse+fine) for reference metrics ---
        torch.manual_seed(seed + batch_idx)
        coarse_det = coarse_model.generate(past, **det_kwargs)
        fine_det = fine_model.generate(
            past,
            future_coarse_2d=coarse_det["future_2d_coarse"],
            **det_kwargs,
        )
        anchor_pred = fine_det.get("prediction_global_norm", fine_det.get("prediction"))
        if anchor_pred is None:
            raise KeyError("fine anchor generate missing prediction_global_norm")
        anchor_all.append(anchor_pred.detach().cpu().numpy())

        # --- hybrid: coarse anchor map → fine quad_t × n_samples ---
        torch.manual_seed(seed + batch_idx * 1009)
        past_exp = past.repeat_interleave(n_samples, dim=0)
        coarse_map = coarse_det["future_2d_coarse"].repeat_interleave(n_samples, dim=0)
        fine_prob = fine_model.generate(
            past_exp,
            future_coarse_2d=coarse_map,
            **prob_kwargs,
        )
        prob_pred = fine_prob.get("prediction_global_norm", fine_prob.get("prediction"))
        if prob_pred is None:
            raise KeyError("fine prob generate missing prediction_global_norm")
        samples_bvs = _reshape_samples(prob_pred, batch_n, n_samples)
        sample_all.append(samples_bvs.detach().cpu().numpy())

        elapsed = time.perf_counter() - t0
        done = batch_idx + 1
        eta = (elapsed / done) * (len(loader) - done) if done else 0.0
        print(
            f"[eval] batch {done}/{len(loader)} n={batch_n} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    pack = {
        "y_true": np.concatenate(y_true_all, axis=0),
        "deterministic": np.concatenate(anchor_all, axis=0),
        "final_anchor": np.concatenate(anchor_all, axis=0),
        "samples": np.concatenate(sample_all, axis=0),
        "window_indices": np.asarray(window_idx_all, dtype=np.int64),
        "series_starts": np.asarray(window_idx_all, dtype=np.int64) * int(test_stride),
    }
    pack["sample_mean"] = pack["samples"].mean(axis=2)
    return pack


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _assert_stride_policy(int(args.eval_test_stride), int(args.regular_test_stride))

    if args.smoke_test:
        args.n_samples = min(int(args.n_samples), 2)
        args.prob_steps = min(int(args.prob_steps), 5)
        args.batch_size = 1

    ckpt = _discover_ckpt(args.ckpt)
    cfg_path = str((REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute()
                   else Path(args.config))
    if not Path(cfg_path).is_file():
        raise FileNotFoundError(cfg_path)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[setup] ckpt={ckpt}", flush=True)
    print(f"[setup] config={cfg_path} device={device}", flush=True)
    print(
        f"[setup] coarse=anchor | fine={args.prob_sampler} "
        f"n_samples={args.n_samples} steps={args.prob_steps}",
        flush=True,
    )

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.pipeline.data_subset import resolve_pipeline_data_subset
    from models.diffusion_tsf.train_multivariate_pipeline import (
        LOOKBACK_LENGTH,
        PREDICTION_LENGTH,
        load_dataset,
        load_wrapped_guidance,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from utils.visualize_staged_eval_2d_preds import (
        _load_stage_model,
        _resolve_guidance_ckpt,
    )
    from models.diffusion_tsf.pipeline.phases.staged_eval import (
        _deterministic_metrics,
        _summarize_staged_eval_metrics,
    )

    cfg = load_experiment_config(cfg_path, cli_overrides={"dataset": args.dataset})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(ckpt)
    state.dataset = args.dataset
    state.seed = int(args.seed)
    resolve_pipeline_data_subset(state)
    subset_id = state.subset_id or args.dataset
    state.subset_id = subset_id
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    lookback = int(getattr(pipeline_mod, "LOOKBACK_LENGTH", LOOKBACK_LENGTH) or LOOKBACK_LENGTH)
    horizon = int(getattr(pipeline_mod, "PREDICTION_LENGTH", PREDICTION_LENGTH) or PREDICTION_LENGTH)
    variate_indices = list(state.variate_indices or [])
    if not variate_indices:
        raise RuntimeError(f"{args.dataset}: empty variate_indices")

    train_stride = int(
        (state.data_subset_resolved or {}).get("train_stride", state.window_stride) or 1
    )
    _, _, test_ds, norm_stats = load_dataset(
        args.dataset,
        variate_indices,
        stride=train_stride,
        test_stride=int(args.eval_test_stride),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=bool(state.use_ordinal_window_norm),
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    if args.smoke_test:
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
        window_indices = list(range(len(test_ds)))
    else:
        window_indices = list(range(len(test_ds)))

    print(f"[data] subset={subset_id} test_windows={len(test_ds)} "
          f"stride={args.eval_test_stride} lookback={lookback} horizon={horizon}",
          flush=True)

    guidance_path, guidance_type = _resolve_guidance_ckpt(ckpt, subset_id, "auto")
    guidance = load_wrapped_guidance(
        str(guidance_path),
        len(variate_indices),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    patch_refine = bool(getattr(state, "use_patch_refine_stage", False))
    refine_stage = "patch_refine" if patch_refine else "fine"
    # Stage ckpts live under subset_id/
    coarse_pt = ckpt / subset_id / "coarse" / "best.pt"
    fine_pt = ckpt / subset_id / refine_stage / "best.pt"
    if not coarse_pt.is_file():
        # legacy dataset-named dir
        coarse_pt = ckpt / args.dataset / "coarse" / "best.pt"
        fine_pt = ckpt / args.dataset / refine_stage / "best.pt"
    if not coarse_pt.is_file() or not fine_pt.is_file():
        raise FileNotFoundError(
            f"missing stage best.pt under {ckpt}/{subset_id} "
            f"(coarse={coarse_pt.is_file()} fine={fine_pt.is_file()})"
        )
    coarse_model = _load_stage_model(
        state, "coarse", coarse_pt, guidance, len(variate_indices), device,
    )
    fine_model = _load_stage_model(
        state, refine_stage, fine_pt, guidance, len(variate_indices), device,
    )
    ranked = bool(getattr(test_ds if not isinstance(test_ds, Subset) else test_ds.dataset,
                          "yields_ordinal_ranks", False))
    for m in (coarse_model, fine_model):
        m._ordinal_input_is_ranked = ranked
        m._ordinal_apply_ood_shift = not ranked

    loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    pack = run_eval(
        coarse_model=coarse_model,
        fine_model=fine_model,
        loader=loader,
        window_indices=window_indices,
        device=device,
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        prob_sampler=str(args.prob_sampler),
        prob_steps=int(args.prob_steps),
        test_stride=int(args.eval_test_stride),
    )
    metrics = _summarize_staged_eval_metrics(
        pack,
        gmm_components=int(args.gmm_components),
        seed=int(args.seed),
        topk_max=int(args.topk_max),
    )
    # Primary "final" forecast for this probe = fine sample mean (conditioned on coarse anchor).
    sm = _deterministic_metrics(pack["y_true"], pack["sample_mean"])
    metrics["hybrid_sample_mean_mse"] = sm["mse"]
    metrics["hybrid_sample_mean_mae"] = sm["mae"]

    stamp = time.strftime("%m-%d-%H%M")
    out = args.output_dir
    if out is None:
        tag = "smoke" if args.smoke_test else "full"
        out = REPO_ROOT / "results" / "datasets" / (
            f"{stamp}-etth1-coarse-anchor-fine-prob-avg3-{tag}"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw / f"hybrid_{args.dataset}.npz",
        y_true=pack["y_true"],
        final_anchor=pack["final_anchor"],
        samples=pack["samples"],
        sample_mean=pack["sample_mean"],
        window_indices=pack["window_indices"],
        series_starts=pack["series_starts"],
    )
    summary = {
        "dataset": args.dataset,
        "subset_id": subset_id,
        "ckpt": str(ckpt),
        "config": cfg_path,
        "coarse_sampler": "anchor",
        "fine_sampler": args.prob_sampler,
        "n_samples": int(args.n_samples),
        "prob_steps": int(args.prob_steps),
        "regular_test_stride": int(args.regular_test_stride),
        "eval_test_stride": int(args.eval_test_stride),
        "stride_density": float(args.regular_test_stride) / float(args.eval_test_stride),
        "n_windows": int(pack["y_true"].shape[0]),
        "smoke_test": bool(args.smoke_test),
        "metrics": {k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("========== METRICS ==========", flush=True)
    print(
        f"hybrid sample_mean  MAE={metrics['hybrid_sample_mean_mae']:.6f}  "
        f"MSE={metrics['hybrid_sample_mean_mse']:.6f}",
        flush=True,
    )
    print(
        f"CRPS (n={args.n_samples})     {metrics.get('crps', float('nan')):.6f}",
        flush=True,
    )
    print(
        f"pure anchor         MAE={metrics['anchor_mae']:.6f}  "
        f"MSE={metrics['anchor_mse']:.6f}",
        flush=True,
    )
    if "sample_mean_mae" in metrics:
        print(
            f"legacy sample_mean  MAE={metrics['sample_mean_mae']:.6f}  "
            f"MSE={metrics['sample_mean_mse']:.6f}",
            flush=True,
        )
    print(f"wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
