#!/usr/bin/env python3
"""Compare anchor-only MSE: window-norm patch_refine vs ordinal guided_p8.

Loads (or regenerates) staged_anchor packs, scores per-window anchor MSE for
both runs, greedily picks the ``--top-k`` windows with largest
``ordinal_mse - window_mse`` whose forecast horizons do not overlap
(lookback may still share history), and plots lookback | GT | both anchors.

Defaults point at the ETTh1 ablation pair on Killarney scratch:
  - window-norm: 08-01-4524397-...-earlyjuly_norm
  - guided_p8:   08-01-4519745-...-guided_p8

Example (Killarney login or WSL with synced results):

  python temp/scripts/compare_anchor_mse_delta_plots.py \\
    --output-dir temp/viz_anchor_mse_delta_wn_vs_guided_p8

  # Force re-eval anchors from ckpts (ignore existing npz):
  python temp/scripts/compare_anchor_mse_delta_plots.py --force-reeval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualize_staged_eval_2d_preds import (  # noqa: E402
    _build_state,
    _load_stage_model,
    _load_staged_bundle,
    _resolve_guidance_ckpt,
)

DEFAULT_WINDOW = (
    "results/ckpts/08-01-4524397-ETTh1-binary_window_norm_patch_refine_earlyjuly_norm"
)
DEFAULT_ORDINAL = (
    "results/ckpts/08-01-4519745-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8"
)
DEFAULT_WINDOW_CFG = "configs/binary_window_norm_patch_refine_earlyjuly_norm.yaml"
DEFAULT_ORDINAL_CFG = (
    "configs/binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8.yaml"
)
DEFAULT_WINDOW_RESULTS = (
    "results/datasets/08-01-4524397-ETTh1-binary_window_norm_patch_refine_earlyjuly_norm"
)
DEFAULT_ORDINAL_RESULTS = (
    "results/datasets/08-01-4519745-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8"
)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _pack_path(results_dir: Path, dataset: str) -> Path:
    return results_dir / "raw" / f"staged_anchor_{dataset}.npz"


def _per_window_mse(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((pred - y_true) ** 2).mean(axis=(1, 2)).astype(np.float64)


def _greedy_nonoverlapping(
    order: Sequence[int],
    interval_starts: np.ndarray,
    *,
    span: int,
    top_k: int,
) -> List[int]:
    """Greedy pick by ``order``; reject windows whose [start, start+span) overlaps."""
    picked: List[int] = []
    intervals: List[Tuple[int, int]] = []
    for i in order:
        s = int(interval_starts[i])
        e = s + int(span)
        if any(not (e <= a or s >= b) for a, b in intervals):
            continue
        picked.append(int(i))
        intervals.append((s, e))
        if len(picked) >= top_k:
            break
    return picked


@torch.no_grad()
def _eval_anchor_pack(
    *,
    ckpt_dir: Path,
    config_path: Path,
    dataset: str,
    window_indices: Sequence[int],
    test_stride: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Anchor-only staged generate on fixed window indices (global-norm space)."""
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
        patch_stage_globals,
    )
    from models.diffusion_tsf.train_multivariate_pipeline import (
        load_dataset,
        load_wrapped_guidance,
        resolve_pipeline_data_subset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    bundle = _load_staged_bundle(ckpt_dir, dataset)
    subset_id = str(bundle["subset_id"])
    variate_indices = [int(i) for i in bundle["variate_indices"]]
    state = _build_state(ckpt_dir, dataset, subset_id, str(config_path))
    state.variate_indices = list(variate_indices)
    resolve_pipeline_data_subset(state)
    variate_indices = list(state.variate_indices)

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    train_stride = int((state.data_subset_resolved or {}).get("train_stride", state.window_stride))
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    guidance = None
    if state.needs_guidance:
        gpath = _resolve_guidance_ckpt(ckpt_dir, subset_id)
        guidance = load_wrapped_guidance(str(gpath), len(variate_indices), device)

    refine_stage = "patch_refine" if bool(state.use_patch_refine_stage) else "fine"
    coarse = _load_stage_model(
        state, "coarse", Path(bundle["coarse_pt"]), guidance, len(variate_indices), device,
        strict_non_guidance_shapes=True,
    )
    refine = _load_stage_model(
        state, refine_stage, Path(bundle["refine_pt"]), guidance, len(variate_indices), device,
        strict_non_guidance_shapes=True,
    )
    ranked = bool(getattr(test_ds, "yields_ordinal_ranks", False))
    for m in (coarse, refine):
        m._ordinal_input_is_ranked = ranked
        m._ordinal_apply_ood_shift = not ranked

    subset = Subset(test_ds, list(window_indices))
    loader = DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    det_kwargs = {"sampler": "anchor"}
    y_all: List[np.ndarray] = []
    det_all: List[np.ndarray] = []
    k = int(getattr(coarse.config, "lookback_overlap", 0) or 0)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            past = past.to(device)
            future = future.to(device)
            yt = future[..., k:] if k > 0 else future
            torch.manual_seed(seed + batch_idx)
            coarse_out = coarse.generate(past, **det_kwargs)
            fine_out = refine.generate(
                past,
                future_coarse_2d=coarse_out["future_2d_coarse"],
                **det_kwargs,
            )
            pred = fine_out["prediction_global_norm"]
            y_all.append(yt.detach().cpu().numpy())
            det_all.append(pred.detach().cpu().numpy())
            print(
                f"  [{config_path.stem}] batch {batch_idx + 1}/{len(loader)} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    y_true = np.concatenate(y_all, axis=0)
    det = np.concatenate(det_all, axis=0)
    wi = np.asarray(list(window_indices), dtype=np.int64)
    return {
        "y_true": y_true,
        "final_anchor": det,
        "deterministic": det,
        "window_indices": wi,
        "series_starts": wi * int(test_stride),
    }


def _load_or_eval_pack(
    *,
    label: str,
    results_dir: Path,
    ckpt_dir: Path,
    config_path: Path,
    dataset: str,
    window_indices: Optional[Sequence[int]],
    test_stride: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    force_reeval: bool,
) -> Dict[str, np.ndarray]:
    pack_file = _pack_path(results_dir, dataset)
    if pack_file.is_file() and not force_reeval:
        print(f"[{label}] reuse pack {pack_file}", flush=True)
        with np.load(pack_file) as z:
            out = {k: z[k] for k in z.files}
        if window_indices is not None:
            wi = np.asarray(out["window_indices"], dtype=np.int64)
            want = np.asarray(list(window_indices), dtype=np.int64)
            if not np.array_equal(wi, want):
                raise RuntimeError(
                    f"[{label}] pack window_indices mismatch vs requested set"
                )
        return out
    if window_indices is None:
        raise FileNotFoundError(
            f"[{label}] missing pack {pack_file} and no window_indices to eval"
        )
    print(f"[{label}] anchor-only re-eval from {ckpt_dir}", flush=True)
    return _eval_anchor_pack(
        ckpt_dir=ckpt_dir,
        config_path=config_path,
        dataset=dataset,
        window_indices=window_indices,
        test_stride=test_stride,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )


def _fetch_past_future(
    *,
    config_path: Path,
    dataset: str,
    window_index: int,
    test_stride: int,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (past, future_horizon) in test-loader space (global z-score)."""
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.train_multivariate_pipeline import (
        generate_dataset_job,
        load_dataset,
        resolve_pipeline_data_subset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    cfg = load_experiment_config(str(config_path))
    state = PipelineState.from_config(cfg)
    state.dataset = dataset
    state.subset_id = dataset
    job = generate_dataset_job(dataset)
    state.variate_indices = list(state.variate_indices or job["variate_indices"])
    resolve_pipeline_data_subset(state)
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    train_stride = int((state.data_subset_resolved or {}).get("train_stride", state.window_stride))
    _, _, test_ds, _ = load_dataset(
        dataset,
        list(state.variate_indices),
        stride=train_stride,
        test_stride=test_stride,
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    past, future = test_ds[int(window_index)]
    past_np = past.numpy() if torch.is_tensor(past) else np.asarray(past)
    fut_np = future.numpy() if torch.is_tensor(future) else np.asarray(future)
    k = int(getattr(test_ds, "lookback_overlap", 0) or 0)
    if k > 0:
        fut_np = fut_np[..., k:]
    return past_np, fut_np


def _plot_window(
    *,
    out_path: Path,
    past: np.ndarray,
    y_true: np.ndarray,
    pred_win: np.ndarray,
    pred_ord: np.ndarray,
    title: str,
    max_variates: int,
) -> None:
    V = min(int(past.shape[0]), int(max_variates))
    lb = past.shape[-1]
    hz = y_true.shape[-1]
    t_past = np.arange(lb)
    t_fut = np.arange(lb, lb + hz)
    ncols = 1
    nrows = V
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 2.2 * V), sharex=True, squeeze=False,
    )
    for v in range(V):
        ax = axes[v, 0]
        ax.plot(t_past, past[v], color="0.45", lw=1.0, label="lookback" if v == 0 else None)
        ax.plot(t_fut, y_true[v], color="black", lw=1.4, label="GT" if v == 0 else None)
        ax.plot(
            t_fut, pred_win[v], color="#1b9e77", lw=1.2, alpha=0.9,
            label="window-norm PR" if v == 0 else None,
        )
        ax.plot(
            t_fut, pred_ord[v], color="#d95f02", lw=1.2, alpha=0.9,
            label="ordinal guided_p8" if v == 0 else None,
        )
        ax.axvline(lb - 0.5, color="0.7", ls="--", lw=0.8)
        ax.set_ylabel(f"v{v}")
        if v == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=4)
    axes[0, 0].set_title(title, fontsize=10)
    axes[-1, 0].set_xlabel("t (lookback | horizon)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--window-ckpt-dir", default=DEFAULT_WINDOW)
    p.add_argument("--ordinal-ckpt-dir", default=DEFAULT_ORDINAL)
    p.add_argument("--window-config", default=DEFAULT_WINDOW_CFG)
    p.add_argument("--ordinal-config", default=DEFAULT_ORDINAL_CFG)
    p.add_argument("--window-results-dir", default=DEFAULT_WINDOW_RESULTS)
    p.add_argument("--ordinal-results-dir", default=DEFAULT_ORDINAL_RESULTS)
    p.add_argument("--test-stride", type=int, default=16)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--delta-mode",
        choices=["ordinal_minus_window", "abs"],
        default="ordinal_minus_window",
        help="Ranking score: ordinal_mse-window_mse (default) or |delta|",
    )
    p.add_argument(
        "--overlap-span",
        type=int,
        default=None,
        help="Non-overlap span in series steps (default=horizon; use lookback+horizon for full-window)",
    )
    p.add_argument("--max-variates", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--force-reeval", action="store_true")
    p.add_argument(
        "--output-dir",
        default="temp/viz_anchor_mse_delta_wn_vs_guided_p8",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    win_ckpt = _resolve(args.window_ckpt_dir)
    ord_ckpt = _resolve(args.ordinal_ckpt_dir)
    win_cfg = _resolve(args.window_config)
    ord_cfg = _resolve(args.ordinal_config)
    win_res = _resolve(args.window_results_dir)
    ord_res = _resolve(args.ordinal_results_dir)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Prefer guided pack as the index source (same seed/fraction historically).
    seed_pack_path = _pack_path(ord_res, dataset)
    if not seed_pack_path.is_file():
        seed_pack_path = _pack_path(win_res, dataset)
    if seed_pack_path.is_file() and not args.force_reeval:
        with np.load(seed_pack_path) as z:
            window_indices = z["window_indices"].astype(np.int64).tolist()
        print(f"[indices] from {seed_pack_path} n={len(window_indices)}", flush=True)
    else:
        raise FileNotFoundError(
            f"Need an existing staged_anchor pack to fix the eval subset, missing {seed_pack_path}. "
            "Run staged_eval once, or pass synced results dirs."
        )

    win_pack = _load_or_eval_pack(
        label="window",
        results_dir=win_res,
        ckpt_dir=win_ckpt,
        config_path=win_cfg,
        dataset=dataset,
        window_indices=window_indices,
        test_stride=args.test_stride,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        force_reeval=args.force_reeval,
    )
    ord_pack = _load_or_eval_pack(
        label="ordinal",
        results_dir=ord_res,
        ckpt_dir=ord_ckpt,
        config_path=ord_cfg,
        dataset=dataset,
        window_indices=window_indices,
        test_stride=args.test_stride,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        force_reeval=args.force_reeval,
    )

    if not np.array_equal(win_pack["window_indices"], ord_pack["window_indices"]):
        raise RuntimeError("window_indices differ between packs")
    if not np.allclose(win_pack["y_true"], ord_pack["y_true"], rtol=0, atol=0):
        # Should be identical for same loader space; warn but continue if tiny drift.
        maxabs = float(np.max(np.abs(win_pack["y_true"] - ord_pack["y_true"])))
        if maxabs > 1e-5:
            raise RuntimeError(f"y_true mismatch between packs (maxabs={maxabs})")

    y_true = win_pack["y_true"]
    pred_w = win_pack["final_anchor"]
    pred_o = ord_pack["final_anchor"]
    mse_w = _per_window_mse(y_true, pred_w)
    mse_o = _per_window_mse(y_true, pred_o)
    delta = mse_o - mse_w
    score = np.abs(delta) if args.delta_mode == "abs" else delta
    series_starts = win_pack["series_starts"].astype(np.int64)
    # Default: non-overlapping forecast horizons only (lookbacks may share history).
    # Interval starts at lookback offset so span=horizon blocks forecast overlap.
    if args.overlap_span is not None:
        span = int(args.overlap_span)
        interval_starts = series_starts
    else:
        span = int(args.horizon)
        interval_starts = series_starts + int(args.lookback)
    order = np.argsort(-score)  # largest score first
    picked = _greedy_nonoverlapping(
        order, interval_starts, span=span, top_k=args.top_k,
    )

    summary = {
        "dataset": dataset,
        "n_windows": int(len(mse_w)),
        "mean_anchor_mse_window": float(mse_w.mean()),
        "mean_anchor_mse_ordinal": float(mse_o.mean()),
        "mean_delta_ordinal_minus_window": float(delta.mean()),
        "delta_mode": args.delta_mode,
        "overlap_span": span,
        "top_k_requested": args.top_k,
        "top_k_selected": len(picked),
        "window_ckpt": str(win_ckpt),
        "ordinal_ckpt": str(ord_ckpt),
        "selected": [],
    }
    print(
        f"mean anchor_mse window={summary['mean_anchor_mse_window']:.4f} "
        f"ordinal={summary['mean_anchor_mse_ordinal']:.4f} "
        f"delta={summary['mean_delta_ordinal_minus_window']:.4f}",
        flush=True,
    )
    print(
        f"selected {len(picked)}/{args.top_k} non-overlapping (span={span}) "
        f"by {args.delta_mode}",
        flush=True,
    )

    # Prefer window-norm config for lookback fetch (test yields global z either way).
    for rank, i in enumerate(picked):
        wi = int(win_pack["window_indices"][i])
        ss = int(series_starts[i])
        row = {
            "rank": rank,
            "pack_row": int(i),
            "window_index": wi,
            "series_start": ss,
            "mse_window": float(mse_w[i]),
            "mse_ordinal": float(mse_o[i]),
            "delta": float(delta[i]),
        }
        summary["selected"].append(row)
        print(
            f"  #{rank:02d} wi={wi} start={ss} "
            f"mse_w={mse_w[i]:.4f} mse_o={mse_o[i]:.4f} delta={delta[i]:+.4f}",
            flush=True,
        )
        past, _fut = _fetch_past_future(
            config_path=win_cfg,
            dataset=dataset,
            window_index=wi,
            test_stride=args.test_stride,
            lookback=args.lookback,
            horizon=args.horizon,
        )
        title = (
            f"#{rank:02d} wi={wi} start={ss}  "
            f"mse_w={mse_w[i]:.3f} mse_o={mse_o[i]:.3f} Δ(o−w)={delta[i]:+.3f}"
        )
        _plot_window(
            out_path=out_dir / f"rank{rank:02d}_wi{wi}_start{ss}.png",
            past=past,
            y_true=y_true[i],
            pred_win=pred_w[i],
            pred_ord=pred_o[i],
            title=title,
            max_variates=args.max_variates,
        )

    # Also dump full per-window table.
    table = {
        "window_indices": win_pack["window_indices"].tolist(),
        "series_starts": series_starts.tolist(),
        "mse_window": mse_w.tolist(),
        "mse_ordinal": mse_o.tolist(),
        "delta_ordinal_minus_window": delta.tolist(),
    }
    (out_dir / "per_window_mse.json").write_text(json.dumps(table, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
