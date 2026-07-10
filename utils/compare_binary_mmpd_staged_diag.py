#!/usr/bin/env python3
"""Compare binary staged anchor vs MMPD per-window, plot worst gaps, cache eval.

Ranks test windows by |anchor MSE gap| (largest disagreement). Picks top-K with
optional min spacing on series start, then adds N random windows from the rest.
For each pick, saves binary coarse/fine 2D maps + 1D reps vs GT with MMPD overlay.

Caches per-dataset npz under --output-dir/eval_cache/ so plots can be regenerated
without re-running eval.

Example:
  python utils/compare_binary_mmpd_staged_diag.py \\
    --mmpd-dir results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720 \\
    --binary-config configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml \\
    --mmpd-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml \\
    --datasets ETTh1,weather,electricity,exchange_rate,traffic \\
    --output-dir reports/binary_vs_mmpd_ordinal_lb336_hz720
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.haar_frequency_calibration import ensure_haar_frequency_calibration
from models.diffusion_tsf.pipeline.fourier_frequency_calibration import ensure_fourier_frequency_calibration
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phase_diagnostics import select_spaced_top_k
from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.visualize_utils import per_window_anchor_mse, save_figure_jpg
from models.diffusion_tsf.train_multivariate_pipeline import (
    generate_dataset_job,
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from utils.eval_mmpd_gaussian_anchor import (
    build_anchor_runs_from_subset_config,
    eval_test_stride,
    get_or_create_indices,
    indices_path,
    indices_root,
    load_indices,
    make_eval_indices,
    mmpd_checkpoint_data_names,
    run_mmpd_eval,
    stable_dataset_seed,
    subsample_eval_indices,
    summarize_anchor_prob_core_metrics,
)
from models.diffusion_tsf.ordinal_window_norm import ordinal_decode
from models.diffusion_tsf.pipeline.visualize_utils import _dataset_window_z_scores
from utils.visualize_staged_eval_2d_preds import (
    _anchor_maps,
    _build_state,
    _load_stage_model,
    _load_staged_bundle,
    _resolve_guidance_ckpt,
)
from utils.visualize_staged_forecast import _window_lengths


DEFAULT_BINARY_CONFIG = "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml"
DEFAULT_MMPD_CONFIG = "configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml"
DEFAULT_MMPD_DIR = "results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720"
DEFAULT_BINARY_CKPT_STEM = "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm"
# Campaign dirs from submit_mmpd_decoder_flat_subsets_paper_lb336_hz720*.sh
MMPD_SERIES_GLOBS = (
    "*mmpd-decoder-ordinal-norm-lb336-hz720*",
    "*mmpd-decoder-paper-lb336-hz720*",
)


def discover_latest_mmpd_output_root(
    datasets_root: Path,
    dataset: str,
    *,
    data_names: Sequence[str],
    backbone: str = "Decoder",
    series_globs: Sequence[str] = MMPD_SERIES_GLOBS,
) -> Path:
    """Newest campaign dir under datasets_root that has a Decoder-MMPD ckpt for dataset."""
    campaigns: List[Path] = []
    for glob in series_globs:
        campaigns.extend(
            p for p in datasets_root.glob(glob) if p.is_dir() and not p.name.startswith("_smoke")
        )
    # de-dupe while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in campaigns:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    best: Optional[Tuple[float, Path, Path]] = None
    prefixes = [f"data{name}_" for name in data_names]
    for camp in uniq:
        base = camp / "mmpd_out" / "checkpoints" / f"{backbone}-MMPD"
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            if not any(d.name.startswith(pref) for pref in prefixes):
                continue
            ckpt = d / "model_checkpoint.pth"
            if not ckpt.is_file():
                continue
            mt = ckpt.stat().st_mtime
            if best is None or mt > best[0]:
                best = (mt, camp.resolve(), ckpt)

    if best is None:
        raise FileNotFoundError(
            f"No {backbone}-MMPD checkpoint for dataset={dataset} "
            f"(tried data names {list(data_names)}) under {datasets_root} "
            f"globs={list(series_globs)}"
        )
    print(
        f"[mmpd-ckpt] {dataset}: using {best[2]} (campaign={best[1].name})",
        flush=True,
    )
    return best[1]


def _staged_eval_overrides(config_path: str, dataset: str) -> Dict[str, Any]:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    for phase in cfg.get("phases") or []:
        if phase.get("phase") == "staged_eval":
            return {k: v for k, v in phase.items() if k != "phase"}
    raise KeyError(f"No staged_eval phase in {config_path}")


def discover_binary_ckpt(ckpt_base: Path, dataset: str, config_stem: str) -> Path:
    suffix = f"-{dataset}-{config_stem}"
    matches = sorted(
        [p for p in ckpt_base.iterdir() if p.is_dir() and p.name.endswith(suffix)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    usable: List[Path] = []
    for p in matches:
        # Prefer dirs that actually have staged coarse/fine weights (skip empty cluster stubs).
        try:
            _load_staged_bundle(p, dataset)
        except FileNotFoundError:
            continue
        usable.append(p)
    if not usable:
        raise FileNotFoundError(
            f"No usable checkpoint dir *{suffix} under {ckpt_base} "
            f"(need {{subset}}/coarse|fine/best.pt)"
        )
    return usable[0]


def build_mmpd_args(
    *,
    mmpd_dir: Path,
    mmpd_config: Path,
    repo: Path,
    force_mmpd_eval: bool,
    smoke_test: bool = False,
) -> argparse.Namespace:
    args = argparse.Namespace(
        datasets=[],
        output_dir=mmpd_dir.resolve(),
        ckpt_base=(repo / "results" / "ckpts").resolve(),
        mmpd_repo=(repo / "temp" / "MMPD").resolve(),
        mmpd_data_dir=(repo / "temp" / "mmpd_datasets").resolve(),
        mmpd_run_config=mmpd_config.resolve(),
        seed=2026,
        skip_mmpd_train=True,
        force_mmpd_eval=force_mmpd_eval,
        force_indices=False,
        # Always keep MMPD campaign index files at full coverage; diag fraction
        # is applied in-memory in run_or_load_dataset_eval.
        test_fraction=1.0,
        metrics_profile="anchor-compat",
        mmpd_instance_norm=False,
        no_update_mmpd=True,
        sample_num=20,
        num_sampling_steps=20,
        gmm_components=10,
        gmm_iterations=10,
        topk_max=3,
        mmpd_eval_batch_size=16,
        num_workers=0,
        gpu=0,
        cpu=False,
        indices_dir=None,
        mmpd_output_root=None,
        test_max_items=None,
        eval_test_stride=None,
        use_ordinal_window_norm=False,
        ordinal_tie_atol=1.0e-6,
        mmpd_backbone="Decoder",
        lookback=336,
        horizon=720,
        patch_size=None,
        subset_config=None,
    )
    with mmpd_config.open(encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    mmpd_block = full_cfg.get("mmpd")
    if not isinstance(mmpd_block, dict):
        raise ValueError(f"{mmpd_config} missing mmpd: block")
    from utils.mmpd_run_config import apply_mmpd_run_config

    apply_mmpd_run_config(args, mmpd_block)
    exp = full_cfg.get("experiment") or {}
    if exp.get("use_ordinal_window_norm"):
        args.use_ordinal_window_norm = True
        args.mmpd_instance_norm = False
    if "ordinal_tie_atol" in exp:
        args.ordinal_tie_atol = float(exp["ordinal_tie_atol"])
    if args.subset_config is None:
        raise ValueError(
            f"{mmpd_config}: mmpd.subset_config required (e.g. binary_anchor_ar.yaml)"
        )
    if smoke_test:
        args.test_max_items = 8
        args.sample_num = 2
        args.num_sampling_steps = 2
        args.mmpd_eval_batch_size = 4
        args.force_mmpd_eval = True
    return args


def cache_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "eval_cache" / f"{dataset}.npz"


def summary_path(cache_dir: Path, dataset: str) -> Path:
    return cache_dir / f"{dataset}_summary.json"


def load_eval_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_eval_cache(
    path: Path,
    *,
    window_indices: np.ndarray,
    series_starts: np.ndarray,
    binary_mse: np.ndarray,
    mmpd_mse: np.ndarray,
    diff: np.ndarray,
    binary_metrics: Dict[str, float],
    mmpd_metrics: Dict[str, float],
    test_stride: int,
    binary_ckpt: str,
    mmpd_dir: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_indices=window_indices,
        series_starts=series_starts,
        binary_anchor_mse=binary_mse,
        mmpd_anchor_mse=mmpd_mse,
        error_diff=diff,
        test_stride=np.array([test_stride], dtype=np.int64),
    )
    summary = {
        "dataset": path.stem,
        "binary_ckpt": binary_ckpt,
        "mmpd_dir": mmpd_dir,
        "test_stride": test_stride,
        "n_windows": int(len(window_indices)),
        "binary_metrics": binary_metrics,
        "mmpd_metrics": mmpd_metrics,
        "mean_binary_anchor_mse": float(binary_mse.mean()),
        "mean_mmpd_anchor_mse": float(mmpd_mse.mean()),
        "mean_error_diff": float(diff.mean()),
    }
    with summary_path(path.parent, path.stem).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def align_mmpd_pack(pack: Dict[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
    pack_indices = pack.get("indices")
    if pack_indices is None:
        if pack["y_true"].shape[0] != len(indices):
            raise ValueError("MMPD pack missing indices and row count mismatch")
        pack_indices = np.asarray(indices, dtype=np.int64)
    order = {int(wi): row for row, wi in enumerate(pack_indices)}
    rows = []
    for wi in indices:
        if int(wi) not in order:
            raise KeyError(f"window {wi} missing from MMPD eval pack")
        rows.append(order[int(wi)])
    idx = np.asarray(rows, dtype=np.int64)
    out = {
        "y_true": pack["y_true"][idx],
        "deterministic": pack["deterministic"][idx],
        "indices": np.asarray(indices, dtype=np.int64),
    }
    if "samples" in pack:
        out["samples"] = pack["samples"][idx]
    return out


@torch.no_grad()
def run_binary_staged_eval(
    *,
    checkpoint_dir: Path,
    dataset: str,
    config_path: str,
    window_indices: Sequence[int],
    test_stride: int,
    device: torch.device,
    prob_samples: Optional[int] = None,
    prob_steps: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    overrides = _staged_eval_overrides(config_path, dataset)
    phase = StagedEvalPhase(**overrides)
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(checkpoint_dir.resolve())
    state.dataset = dataset
    resolve_pipeline_data_subset(state)
    subset_id = state.subset_id or dataset
    state.subset_id = subset_id

    ensure_haar_frequency_calibration(state)
    ensure_fourier_frequency_calibration(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    variate_indices = state.variate_indices
    if variate_indices is None:
        variate_indices = generate_dataset_job(dataset)["variate_indices"]
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))

    lookback, horizon = _window_lengths(dataset, state)
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]

    guidance_path, guidance_type = _resolve_guidance_ckpt(checkpoint_dir, subset_id, "auto")
    guidance = load_wrapped_guidance(
        str(guidance_path),
        len(variate_indices),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    coarse_model = phase._load_model(state, "coarse", guidance, len(variate_indices), device)
    fine_model = phase._load_model(state, "fine", guidance, len(variate_indices), device)
    finer_model = (
        phase._load_model(state, "finer", guidance, len(variate_indices), device)
        if state.use_triple_scale
        else None
    )

    subset = Subset(test_ds, list(window_indices))
    # 336/720 maps are ~5.5× larger than 96/96; keep micro-batch small to avoid OOM kills.
    batch_size = min(2, int(overrides.get("batch_size", 8)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    # Diag ranks on anchor MSE only — skip the 20-sample CRPS path (huge VRAM/time).
    n_prob = int(prob_samples if prob_samples is not None else 1)
    n_steps = int(prob_steps if prob_steps is not None else overrides.get("probabilistic_num_inference_steps", 20))
    print(
        f"[eval] {dataset}: binary loader batch={batch_size} "
        f"prob_samples={n_prob} windows={len(window_indices)}",
        flush=True,
    )
    metrics, pack = phase._run_eval(
        state=state,
        subset_id=subset_id,
        loader=loader,
        device=device,
        coarse_model=coarse_model,
        fine_model=fine_model,
        finer_model=finer_model,
        prob_sampler=str(overrides.get("probabilistic_sampler", "dpmpp")),
        prob_steps=n_steps,
        prob_samples=n_prob,
        gmm_components=int(overrides.get("gmm_components", 10)),
        topk_max=int(overrides.get("topk_max", 3)),
        window_indices=list(window_indices),
        test_stride=test_stride,
    )
    return metrics, pack


def run_or_load_dataset_eval(
    *,
    dataset: str,
    mmpd_args: argparse.Namespace,
    binary_ckpt: Path,
    binary_config: str,
    output_dir: Path,
    device: torch.device,
    force_eval: bool,
    test_fraction: float = 1.0,
    datasets_root: Optional[Path] = None,
    auto_mmpd_ckpt: bool = True,
) -> Dict[str, np.ndarray]:
    cache = cache_path(output_dir, dataset)
    if cache.is_file() and not force_eval:
        print(f"[cache] {dataset}: loading {cache}")
        return load_eval_cache(cache)

    if mmpd_args.subset_config is None:
        raise ValueError("mmpd.subset_config not resolved; check mmpd run YAML")
    subset_runs = build_anchor_runs_from_subset_config(
        Path(mmpd_args.subset_config),
        [dataset],
        int(mmpd_args.seed),
    )
    run = subset_runs[dataset]
    if auto_mmpd_ckpt:
        root = (datasets_root or (REPO_ROOT / "results" / "datasets")).resolve()
        mmpd_args.mmpd_output_root = discover_latest_mmpd_output_root(
            root,
            dataset,
            data_names=mmpd_checkpoint_data_names(run),
            backbone=str(mmpd_args.mmpd_backbone),
        )
    idx_root = indices_root(mmpd_args)
    indices_file = indices_path(idx_root, dataset)
    if indices_file.is_file() and not mmpd_args.force_indices:
        window_indices = load_indices(idx_root, dataset)
    else:
        window_indices = get_or_create_indices(mmpd_args, run)
    # Fraction subsample in-memory only — do not rewrite MMPD campaign index files.
    frac = float(test_fraction)
    if frac < 1.0:
        n_full = len(window_indices)
        keep = make_eval_indices(
            n_full,
            frac,
            stable_dataset_seed(int(mmpd_args.seed), dataset),
            None,
        )
        window_indices = [int(window_indices[i]) for i in keep]
        print(
            f"[subset] {dataset}: test_fraction={frac:g} -> {len(window_indices)}/{n_full} windows",
            flush=True,
        )
    if mmpd_args.test_max_items is not None:
        window_indices = subsample_eval_indices(
            window_indices,
            mmpd_args.test_max_items,
            seed=int(mmpd_args.seed),
            dataset=dataset,
        )
    test_stride = eval_test_stride(mmpd_args, run)

    smoke_prob = None
    smoke_steps = None
    if mmpd_args.test_max_items is not None and mmpd_args.sample_num <= 4:
        smoke_prob = int(mmpd_args.sample_num)
        smoke_steps = int(mmpd_args.num_sampling_steps)

    print(f"[eval] {dataset}: binary staged anchor ({len(window_indices)} windows, stride={test_stride})")
    binary_metrics, binary_pack = run_binary_staged_eval(
        checkpoint_dir=binary_ckpt,
        dataset=dataset,
        config_path=binary_config,
        window_indices=window_indices,
        test_stride=test_stride,
        device=device,
        prob_samples=smoke_prob,
        prob_steps=smoke_steps,
    )

    print(f"[eval] {dataset}: MMPD anchor")
    mmpd_pack_raw = run_mmpd_eval(mmpd_args, run, window_indices)
    mmpd_pack = align_mmpd_pack(mmpd_pack_raw, window_indices)
    mmpd_metrics = summarize_anchor_prob_core_metrics(mmpd_pack)

    y_true_bin = binary_pack["y_true"]
    y_true_mmpd = mmpd_pack["y_true"]
    if y_true_bin.shape != y_true_mmpd.shape:
        raise RuntimeError(
            f"{dataset}: shape mismatch binary {y_true_bin.shape} vs mmpd {y_true_mmpd.shape}"
        )
    if not np.allclose(y_true_bin, y_true_mmpd, rtol=1e-4, atol=1e-4):
        row = int(np.argmax(np.abs(y_true_bin - y_true_mmpd).reshape(len(y_true_bin), -1).mean(axis=1)))
        wi = int(window_indices[row]) if row < len(window_indices) else row
        raise RuntimeError(
            f"{dataset}: y_true mismatch at row {row} (window {wi}); "
            "check eval_test_stride / ordinal norm alignment"
        )
    y_true = y_true_mmpd

    binary_mse = per_window_anchor_mse(y_true, binary_pack["final_anchor"])
    mmpd_mse = per_window_anchor_mse(y_true, mmpd_pack["deterministic"])
    series_starts = np.asarray(window_indices, dtype=np.int64) * int(test_stride)
    diff = binary_mse - mmpd_mse

    save_eval_cache(
        cache,
        window_indices=np.asarray(window_indices, dtype=np.int64),
        series_starts=series_starts,
        binary_mse=binary_mse,
        mmpd_mse=mmpd_mse,
        diff=diff,
        binary_metrics=binary_metrics,
        mmpd_metrics=mmpd_metrics,
        test_stride=test_stride,
        binary_ckpt=str(binary_ckpt),
        mmpd_dir=str(mmpd_args.output_dir),
    )
    print(f"[cache] {dataset}: wrote {cache}")
    return load_eval_cache(cache)


def rank_scores(cache: Dict[str, np.ndarray], diff_mode: str) -> np.ndarray:
    if diff_mode == "mmpd_minus_binary":
        return cache["mmpd_anchor_mse"] - cache["binary_anchor_mse"]
    if diff_mode == "binary_minus_mmpd":
        return cache["binary_anchor_mse"] - cache["mmpd_anchor_mse"]
    if diff_mode == "abs_diff":
        return np.abs(cache["binary_anchor_mse"] - cache["mmpd_anchor_mse"])
    raise ValueError(f"unknown diff_mode: {diff_mode}")


def _window_manifest_entry(
    cache: Dict[str, np.ndarray],
    *,
    row: int,
    rank: int,
    pick_kind: str,
    scores: np.ndarray,
) -> Dict[str, Any]:
    wi = int(cache["window_indices"][row])
    return {
        "rank": rank,
        "pick_kind": pick_kind,
        "row": int(row),
        "window_index": wi,
        "series_start": int(cache["series_starts"][row]),
        "binary_anchor_mse": float(cache["binary_anchor_mse"][row]),
        "mmpd_anchor_mse": float(cache["mmpd_anchor_mse"][row]),
        "error_diff_mmpd_minus_binary": float(
            cache["mmpd_anchor_mse"][row] - cache["binary_anchor_mse"][row]
        ),
        "error_diff_binary_minus_mmpd": float(
            cache["binary_anchor_mse"][row] - cache["mmpd_anchor_mse"][row]
        ),
        "rank_score": float(scores[row]),
    }


def select_top_windows(
    cache: Dict[str, np.ndarray],
    *,
    top_k: int,
    random_k: int,
    min_spacing: int,
    diff_mode: str,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    scores = rank_scores(cache, diff_mode)
    n = int(len(scores))
    top_rows = select_spaced_top_k(
        scores,
        cache["series_starts"],
        k=min(top_k, n),
        min_spacing=min_spacing,
    )
    used = set(top_rows)
    remaining = [i for i in range(n) if i not in used]
    rng = np.random.default_rng(seed)
    n_rand = min(int(random_k), len(remaining))
    rand_rows: List[int] = []
    if n_rand > 0:
        chosen = rng.choice(np.asarray(remaining, dtype=np.int64), size=n_rand, replace=False)
        rand_rows = [int(i) for i in chosen]

    manifest: List[Dict[str, Any]] = []
    for rank, row in enumerate(top_rows, start=1):
        manifest.append(
            _window_manifest_entry(
                cache, row=row, rank=rank, pick_kind="top_diff", scores=scores
            )
        )
    for j, row in enumerate(rand_rows, start=1):
        manifest.append(
            _window_manifest_entry(
                cache,
                row=row,
                rank=len(top_rows) + j,
                pick_kind="random",
                scores=scores,
            )
        )
    return manifest


def _ensure_bvt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


@torch.no_grad()
def _ordinal_or_window_to_global_z(
    model: torch.nn.Module,
    past_model: torch.Tensor,
    future_model: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Map model-space 1D (ordinal ranks or window-norm) to global z."""
    past_model = _ensure_bvt(past_model)
    if future_model is not None:
        future_model = _ensure_bvt(future_model)
    if model._uses_global_ordinal_encoding():
        ladder = model.config.ordinal_ladder.expand_batch(past_model.shape[0])
        return ordinal_decode(past_model, future_model, ladder)
    # Window-norm path: past_model already window-normed; recover via stored stats if present.
    # For diag plots we pass global-z GT separately; treat model space as already comparable.
    return past_model, future_model


@torch.no_grad()
def _decode_maps_to_plot_1d(
    fine_model: torch.nn.Module,
    coarse_map: np.ndarray,
    fine_map: np.ndarray,
    *,
    lookback: int,
    future_raw_len: int,
    upsample_to_raw: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode past|future CDF maps (future half still includes overlap prefix).

    Returns (coarse_global_z, fine_Δz, final_global_z) with time length:
      lookback + future_raw_len   if upsample_to_raw
      repr(lookback)+repr(future_raw_len) otherwise

    Fine residual = final_z - coarse_z after middle-bin fine decode in rank space.
    """
    device = next(fine_model.parameters()).device
    coarse_t = torch.from_numpy(np.asarray(coarse_map)).unsqueeze(0).to(device=device, dtype=torch.float32)
    fine_t = torch.from_numpy(np.asarray(fine_map)).unsqueeze(0).to(device=device, dtype=torch.float32)

    coarse_1d = _ensure_bvt(fine_model._decode_coarse_1d_from_map(coarse_t, cdf_decoder="mean"))
    if fine_model._uses_global_ordinal_encoding():
        vmax = fine_model._ordinal_rank_max_tensor(device, dtype=coarse_t.dtype)
        fine_vals = []
        for vi in range(fine_t.shape[1]):
            span = float(vmax[vi].item())
            fine_range = span / float(coarse_t.shape[2]) * 0.5 if span > 0.0 else 0.0
            fine_vals.append(
                fine_model.to_2d._decode_occupancy_bounded(
                    fine_t[:, vi : vi + 1],
                    value_min=-fine_range,
                    value_max=fine_range,
                    cdf_decoder="mean",
                )
            )
        fine_rank = torch.cat(fine_vals, dim=1)
    else:
        fine_rank = _ensure_bvt(
            fine_model._decode_fine_1d_from_map(
                fine_t,
                coarse_height=int(coarse_t.shape[2]),
                cdf_decoder="mean",
            )
        )
    final_1d = coarse_1d + fine_rank

    w_past = int(fine_model._repr_time_len(lookback))
    if coarse_1d.shape[-1] < w_past:
        raise ValueError(f"map width {coarse_1d.shape[-1]} < past repr {w_past}")
    w_fut = int(coarse_1d.shape[-1] - w_past)
    w_fut_expected = int(fine_model._repr_time_len(future_raw_len))
    if w_fut != w_fut_expected:
        raise ValueError(
            f"future map width {w_fut} != repr({future_raw_len})={w_fut_expected} "
            f"(lookback={lookback}, total_map={coarse_1d.shape[-1]})"
        )

    if upsample_to_raw:
        past_len = lookback

        def _split_up(x: torch.Tensor) -> torch.Tensor:
            past = fine_model._resample_1d_time_series(x[..., :w_past], lookback)
            fut = fine_model._resample_1d_time_series(x[..., w_past:], future_raw_len)
            return torch.cat([past, fut], dim=-1)

        coarse_1d = _split_up(coarse_1d)
        final_1d = _split_up(final_1d)
    else:
        past_len = w_past

    past_c, fut_c = coarse_1d[..., :past_len], coarse_1d[..., past_len:]
    past_f, fut_f = final_1d[..., :past_len], final_1d[..., past_len:]
    past_c_z, fut_c_z = _ordinal_or_window_to_global_z(fine_model, past_c, fut_c)
    past_f_z, fut_f_z = _ordinal_or_window_to_global_z(fine_model, past_f, fut_f)
    coarse_z = torch.cat([past_c_z, fut_c_z], dim=-1)
    final_z = torch.cat([past_f_z, fut_f_z], dim=-1)
    fine_z = final_z - coarse_z
    return (
        coarse_z[0].detach().cpu().numpy(),
        fine_z[0].detach().cpu().numpy(),
        final_z[0].detach().cpu().numpy(),
    )


def _strip_overlap_raw(
    past_future: np.ndarray,
    *,
    lookback: int,
    lookback_overlap: int,
) -> np.ndarray:
    """past | future(with K) → past | future_core (length lookback + horizon)."""
    k = int(lookback_overlap)
    if past_future.shape[-1] < lookback + max(k, 0) + 1 and k > 0:
        raise ValueError(
            f"canvas length {past_future.shape[-1]} too short for lookback={lookback} K={k}"
        )
    past = past_future[..., :lookback]
    fut = past_future[..., lookback:]
    if k > 0:
        if fut.shape[-1] <= k:
            raise ValueError(f"future length {fut.shape[-1]} <= overlap {k}")
        fut = fut[..., k:]
    return np.concatenate([past, fut], axis=-1)


def _paint_value_ridge(
    values: np.ndarray,
    *,
    y_min: float,
    y_max: float,
    n_rows: int = 128,
    band: int = 1,
) -> np.ndarray:
    """Paint decoded 1D values as black filled pixels on white (n_rows, T)."""
    t = int(values.shape[-1])
    # 1 = white (empty), 0 = black (filled)
    img = np.ones((n_rows, t), dtype=np.float32)
    if y_max <= y_min:
        return img
    span = y_max - y_min
    rows = np.rint((values - y_min) / span * (n_rows - 1)).astype(np.int64)
    rows = np.clip(rows, 0, n_rows - 1)
    for b in range(-band, band + 1):
        rr = np.clip(rows + b, 0, n_rows - 1)
        img[rr, np.arange(t)] = 0.0
    return img


def _symmetric_lim(values: np.ndarray, *, pad: float = 1.1, floor: float = 1e-3) -> float:
    m = float(np.nanmax(np.abs(values))) if values.size else 0.0
    return max(m * pad, floor)


def _future_core_global_z(future_z: np.ndarray, lookback_overlap: int) -> np.ndarray:
    """Strip overlap prefix so t=0 matches MMPD / staged_eval y_true."""
    k = int(lookback_overlap)
    if k <= 0:
        return future_z
    if future_z.shape[-1] <= k:
        raise ValueError(f"future length {future_z.shape[-1]} <= overlap {k}")
    return future_z[..., k:]


def _lag0_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 8:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _assert_horizon_aligned(
    name: str,
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    min_corr: float = 0.85,
) -> None:
    if pred.shape != ref.shape:
        raise ValueError(f"{name}: shape {pred.shape} != ref {ref.shape}")
    corr = _lag0_corr(pred, ref)
    if not np.isfinite(corr) or corr < min_corr:
        raise ValueError(f"{name}: lag-0 corr={corr:.4f} < {min_corr} (likely time shift)")


def _plot_compare_panel(
    *,
    maps: Dict[str, Any],
    fine_model: torch.nn.Module,
    mmpd_1d: np.ndarray,
    past_z: np.ndarray,
    future_z: np.ndarray,
    dataset: str,
    window_index: int,
    meta: Dict[str, Any],
    output_path: Path,
    variables_to_plot: int,
    jpeg_dpi: int,
) -> List[Path]:
    """Plot GT / binary / MMPD — one wide figure per variate.

    Shared raw-time axis t ∈ [-LB, H) on every panel. Overlap K is decoded with
    the future half, then stripped in raw time. Horizon binary curves come from
    generate()'s prediction_global_norm.
    """
    k = int(getattr(fine_model.config, "lookback_overlap", 0) or 0)
    lookback = int(past_z.shape[-1])
    future_raw_len = int(future_z.shape[-1])  # hz + K
    if k > 0 and future_raw_len <= k:
        raise ValueError(f"future_z length {future_raw_len} <= lookback_overlap {k}")
    future_core_z = _future_core_global_z(future_z, k)
    horizon_core = int(future_core_z.shape[-1])
    if k > 0 and future_raw_len != horizon_core + k:
        raise ValueError(
            f"future_z length {future_raw_len} != horizon_core {horizon_core} + K {k}"
        )

    gt_1d = np.concatenate([past_z, future_core_z], axis=-1)
    canvas_len = int(gt_1d.shape[-1])
    if canvas_len != lookback + horizon_core:
        raise ValueError(f"gt canvas {canvas_len} != LB+H {lookback + horizon_core}")

    gt_c_full, gt_f_full, gt_final_full = _decode_maps_to_plot_1d(
        fine_model,
        maps["gt_coarse"],
        maps["gt_fine"],
        lookback=lookback,
        future_raw_len=future_raw_len,
    )
    pr_c_full, pr_f_full, pr_final_full = _decode_maps_to_plot_1d(
        fine_model,
        maps["pred_coarse"],
        maps["pred_fine"],
        lookback=lookback,
        future_raw_len=future_raw_len,
    )
    del pr_c_full, pr_f_full
    gt_coarse_np = _strip_overlap_raw(gt_c_full, lookback=lookback, lookback_overlap=k)
    gt_fine_np = _strip_overlap_raw(gt_f_full, lookback=lookback, lookback_overlap=k)
    gt_final_np = _strip_overlap_raw(gt_final_full, lookback=lookback, lookback_overlap=k)
    pr_final_map = _strip_overlap_raw(pr_final_full, lookback=lookback, lookback_overlap=k)

    fine_out = maps["fine_out"]
    coarse_out = maps["coarse_out"]
    bin_final_h = fine_out["prediction_global_norm"][0].detach().cpu().numpy()
    bin_coarse_h = coarse_out["prediction_global_norm"][0].detach().cpu().numpy()
    if bin_final_h.shape[-1] != horizon_core or bin_coarse_h.shape[-1] != horizon_core:
        raise ValueError(
            f"prediction_global_norm width fine={bin_final_h.shape[-1]} "
            f"coarse={bin_coarse_h.shape[-1]} != horizon_core {horizon_core}"
        )

    _assert_horizon_aligned(
        f"{dataset} win {window_index}: map-decode final vs prediction_global_norm",
        pr_final_map[..., lookback:],
        bin_final_h,
        min_corr=0.95,
    )
    _assert_horizon_aligned(
        f"{dataset} win {window_index}: GT encode→decode vs GT future",
        gt_final_np[..., lookback:],
        future_core_z,
        min_corr=0.85,
    )

    mmpd_plot = np.asarray(mmpd_1d)
    if mmpd_plot.ndim != 2:
        raise ValueError(f"MMPD expected (V,H), got {mmpd_plot.shape}")
    if mmpd_plot.shape[-1] != horizon_core:
        raise ValueError(
            f"MMPD future width {mmpd_plot.shape[-1]} != horizon_core {horizon_core} "
            f"(lookback_overlap={k}, raw_future={future_raw_len})"
        )
    if mmpd_plot.shape[0] != gt_1d.shape[0]:
        raise ValueError(
            f"MMPD V={mmpd_plot.shape[0]} != GT V={gt_1d.shape[0]} "
            "(variate subset mismatch — wrong pack or ckpt)"
        )
    # Time base must match GT horizon exactly (length already checked). Fail if
    # someone accidentally passed y_true instead of deterministic with a shift.
    if mmpd_plot.shape != future_core_z.shape:
        raise ValueError(
            f"MMPD shape {mmpd_plot.shape} != future_core_z {future_core_z.shape}"
        )

    coarse_np = np.concatenate([gt_coarse_np[..., :lookback], bin_coarse_h], axis=-1)
    final_np = np.concatenate([gt_final_np[..., :lookback], bin_final_h], axis=-1)
    fine_np = np.concatenate(
        [gt_fine_np[..., :lookback], bin_final_h - bin_coarse_h],
        axis=-1,
    )

    n_vars = min(variables_to_plot, int(gt_1d.shape[0]))
    t_axis = np.arange(-lookback, horizon_core, dtype=np.float64)
    if t_axis.shape[0] != canvas_len:
        raise ValueError(f"t_axis len {t_axis.shape[0]} != canvas {canvas_len}")
    t_h = t_axis[lookback:]
    span_label = f"LB={lookback}, K={k}, H={horizon_core}"

    kind = str(meta.get("pick_kind", "top_diff"))
    stem = output_path.stem
    suffix = output_path.suffix or ".jpg"
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for col in range(n_vars):
        # Per-variate y-limits so one bad channel doesn't squash the rest.
        coarse_lim = _symmetric_lim(
            np.concatenate(
                [
                    gt_1d[col],
                    gt_coarse_np[col],
                    coarse_np[col],
                    final_np[col],
                    mmpd_plot[col],
                ],
                axis=0,
            )
        )
        fine_lim = _symmetric_lim(np.concatenate([gt_fine_np[col], fine_np[col]], axis=0))
        fine_lim = max(fine_lim, 0.05 * coarse_lim)
        fine_lim = min(fine_lim, 0.5 * coarse_lim)

        # One variate per file: very wide + tall so H=720 is readable.
        fig = plt.figure(figsize=(28.0, 18.0), constrained_layout=True)
        gs = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.0, 0.85, 0.85, 1.8])

        ridge_rows = (
            (0, "GT coarse (global z)", gt_coarse_np[col], coarse_lim, False),
            (1, "Binary coarse (global z)", coarse_np[col], coarse_lim, False),
            (2, "GT fine residual (Δz)", gt_fine_np[col], fine_lim, True),
            (3, "Binary fine residual (Δz)", fine_np[col], fine_lim, True),
        )
        for row, label, series, lim, is_fine in ridge_rows:
            if series.shape[-1] != canvas_len:
                raise ValueError(f"{label} ridge T={series.shape[-1]} != canvas {canvas_len}")
            ax = fig.add_subplot(gs[row, 0])
            ridge = _paint_value_ridge(series, y_min=-lim, y_max=lim, n_rows=128, band=1)
            ax.imshow(
                ridge,
                aspect="auto",
                origin="lower",
                extent=[-lookback, horizon_core, -lim, lim],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9, alpha=0.9)
            ax.axvline(x=0.0, color="#888888", linestyle="-", linewidth=1.1, alpha=0.95)
            ax.set_xlim(-lookback, horizon_core)
            ax.set_ylabel("global z" if not is_fine else "Δz", fontsize=11)
            ax.set_title(f"{label} | {span_label}", fontsize=12)
            ax.tick_params(labelsize=10)

        ax = fig.add_subplot(gs[4, 0])
        ax.plot(t_axis, gt_1d[col], color="#1565C0", linewidth=2.2, label="GT (global z)", zorder=4)
        ax.plot(
            t_axis,
            gt_final_np[col],
            color="#90CAF9",
            linewidth=1.4,
            linestyle=":",
            alpha=0.9,
            label="GT encode→decode",
            zorder=2,
        )
        ax.plot(
            t_h,
            coarse_np[col, lookback:],
            color="#EF6C00",
            linewidth=2.0,
            alpha=0.95,
            label="Binary coarse (H)",
            zorder=3,
        )
        ax.plot(
            t_h,
            final_np[col, lookback:],
            color="#C2185B",
            linewidth=2.2,
            label="Binary final (H)",
            zorder=3,
        )
        ax.plot(
            t_h,
            mmpd_plot[col],
            color="#6A1B9A",
            linewidth=2.2,
            linestyle="--",
            label="MMPD (H)",
            zorder=3,
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.45)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1.2, alpha=0.45)
        ax.set_ylim(-coarse_lim, coarse_lim)
        ax.set_xlim(-lookback, horizon_core)
        ax.set_xlabel("t (0 = forecast start)", fontsize=11)
        ax.set_ylabel("global z", fontsize=11)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=10)
        ax.set_title(f"1D global-z | {span_label}", fontsize=12)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.92)

        title = (
            f"{dataset} win {window_index} var {col} | {kind} #{meta['rank']:02d} | "
            f"bin_mse={meta['binary_anchor_mse']:.4f} mmpd_mse={meta['mmpd_anchor_mse']:.4f} "
            f"(mmpd-bin)={meta['error_diff_mmpd_minus_binary']:.4f}"
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        var_path = out_dir / f"{stem}_var{col}{suffix}"
        save_figure_jpg(fig, str(var_path), dpi=jpeg_dpi)
        saved.append(var_path)

    return saved


@torch.no_grad()
def plot_dataset_windows(
    *,
    dataset: str,
    binary_ckpt: Path,
    binary_config: str,
    mmpd_pack_path: Path,
    cache: Dict[str, np.ndarray],
    top_manifest: Sequence[Dict[str, Any]],
    output_dir: Path,
    test_stride: int,
    device: torch.device,
    variables_to_plot: int,
    jpeg_dpi: int,
) -> List[Path]:
    bundle = _load_staged_bundle(binary_ckpt, dataset)
    subset_id = bundle["subset_id"]
    variate_indices = bundle["variate_indices"]
    state = _build_state(binary_ckpt, dataset, subset_id, binary_config)
    lookback, horizon = _window_lengths(dataset, state)
    data_subset = bundle["fine_metadata"].get("data_subset") or {}
    _, _, test_ds, _ = load_dataset(
        dataset,
        variate_indices,
        stride=int(data_subset.get("train_stride", state.window_stride)),
        test_stride=int(test_stride),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )

    guidance_path, guidance_type = _resolve_guidance_ckpt(binary_ckpt, subset_id, "auto")
    guidance_model = load_wrapped_guidance(
        str(guidance_path),
        len(variate_indices),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    coarse_model = _load_stage_model(state, "coarse", bundle["coarse_pt"], guidance_model, len(variate_indices), device)
    fine_model = _load_stage_model(state, "fine", bundle["fine_pt"], guidance_model, len(variate_indices), device)
    ranked = bool(getattr(test_ds, "yields_ordinal_ranks", False))
    for m in (coarse_model, fine_model):
        m._ordinal_input_is_ranked = ranked
        m._ordinal_apply_ood_shift = bool(not ranked)

    with np.load(mmpd_pack_path) as mmpd_data:
        mmpd_det = mmpd_data["deterministic"]
        mmpd_indices = mmpd_data["indices"] if "indices" in mmpd_data.files else cache["window_indices"]

    plot_dir = output_dir / "plots" / dataset
    saved: List[Path] = []
    for entry in top_manifest:
        wi = int(entry["window_index"])
        row = int(entry["row"])
        past, future = test_ds[wi]
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)
        maps = _anchor_maps(coarse_model, fine_model, past_b, future_b)
        # Global-z GT from ds.data (ordinal datasets yield ranks in __getitem__).
        past_z_t, future_z_t = _dataset_window_z_scores(test_ds, wi)
        past_z = past_z_t.numpy()
        future_z = future_z_t.numpy()
        del past_b, future_b
        if device.type == "cuda":
            torch.cuda.empty_cache()
        mmpd_rows = np.where(mmpd_indices == wi)[0]
        if mmpd_rows.size == 0:
            raise KeyError(
                f"{dataset} win {wi}: missing from MMPD pack "
                f"({mmpd_pack_path}); cannot plot aligned future"
            )
        mmpd_1d = mmpd_det[int(mmpd_rows[0])]
        kind = str(entry.get("pick_kind", "top_diff"))
        out_path = (
            plot_dir
            / f"{kind}_r{entry['rank']:02d}_win{wi}_mmpd_minus_bin{entry['error_diff_mmpd_minus_binary']:+.4f}.jpg"
        )
        saved.extend(
            _plot_compare_panel(
                maps=maps,
                fine_model=fine_model,
                mmpd_1d=mmpd_1d,
                past_z=past_z,
                future_z=future_z,
                dataset=dataset,
                window_index=wi,
                meta=entry,
                output_path=out_path,
                variables_to_plot=variables_to_plot,
                jpeg_dpi=jpeg_dpi,
            )
        )
        print(f"[plot] {dataset} win {wi} -> {variables_to_plot} variate figs under {plot_dir}", flush=True)
    return saved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mmpd-dir", type=Path, default=REPO_ROOT / DEFAULT_MMPD_DIR)
    p.add_argument(
        "--mmpd-output-root",
        type=Path,
        default=None,
        help="MMPD train output root with mmpd_out/checkpoints (default: auto-pick newest "
        "lb336/hz720 paper/ordinal campaign ckpt per dataset).",
    )
    p.add_argument(
        "--no-auto-mmpd-ckpt",
        action="store_true",
        help="Disable newest-campaign ckpt discovery; use --mmpd-output-root or --mmpd-dir.",
    )
    p.add_argument("--mmpd-config", type=Path, default=REPO_ROOT / DEFAULT_MMPD_CONFIG)
    p.add_argument("--binary-config", default=DEFAULT_BINARY_CONFIG)
    p.add_argument("--binary-ckpt-base", type=Path, default=REPO_ROOT / "results" / "ckpts")
    p.add_argument("--binary-ckpt-stem", default=DEFAULT_BINARY_CKPT_STEM)
    p.add_argument("--datasets", default="ETTh1,weather,electricity,exchange_rate,traffic")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "binary_vs_mmpd_ordinal_lb336_hz720")
    p.add_argument("--top-k", type=int, default=10, help="Largest |error-diff| windows to plot")
    p.add_argument("--random-k", type=int, default=10, help="Extra random windows (any error)")
    p.add_argument(
        "--min-spacing",
        type=int,
        default=48,
        help="Min series-start spacing among top-diff picks (not applied to random)",
    )
    p.add_argument(
        "--diff-mode",
        choices=("abs_diff", "binary_minus_mmpd", "mmpd_minus_binary"),
        default="abs_diff",
        help="Top-k rank score (default: |binary-mmpd| disagreement).",
    )
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--force-mmpd-eval", action="store_true")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="8 windows, 2 prob samples, small top-k; writes under output-dir_smoke",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.125,
        help="Random fraction of test windows per dataset (default 1/8). Use 1.0 for full set.",
    )
    p.add_argument("--test-max-items", type=int, default=None, help="Cap eval windows per dataset")
    p.add_argument("--plots-only", action="store_true", help="Skip eval; require eval_cache/*.npz")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--variables-to-plot", type=int, default=3)
    p.add_argument("--jpeg-dpi", type=int, default=180)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    output_dir = args.output_dir.resolve()
    if args.smoke_test:
        if not str(output_dir).endswith("_smoke"):
            output_dir = output_dir.parent / f"{output_dir.name}_smoke"
        args.top_k = min(int(args.top_k), 2)
        args.random_k = min(int(args.random_k), 2)
        args.min_spacing = min(int(args.min_spacing), 48)
        args.force_eval = True
        args.test_fraction = 1.0
    elif float(args.test_fraction) < 1.0:
        # Keep full-set caches separate from fraction runs.
        frac_tag = f"_f{args.test_fraction:g}".replace(".", "p")
        if frac_tag not in output_dir.name and not str(output_dir).endswith("_smoke"):
            output_dir = output_dir.parent / f"{output_dir.name}{frac_tag}"
        args.force_eval = True
        args.force_mmpd_eval = True
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    mmpd_campaign_dir = args.mmpd_dir.resolve()
    mmpd_args = build_mmpd_args(
        mmpd_dir=mmpd_campaign_dir,
        mmpd_config=args.mmpd_config.resolve(),
        repo=REPO_ROOT,
        force_mmpd_eval=args.force_mmpd_eval or args.smoke_test,
        smoke_test=args.smoke_test,
    )
    # Indices from --mmpd-dir; ckpts from mmpd_output_root (auto or explicit).
    # Fraction/smoke packs write under the diag output dir only.
    mmpd_args.indices_dir = mmpd_campaign_dir
    if args.mmpd_output_root is not None:
        mmpd_args.mmpd_output_root = args.mmpd_output_root.resolve()
        auto_mmpd_ckpt = False
    else:
        mmpd_args.mmpd_output_root = mmpd_campaign_dir
        auto_mmpd_ckpt = not bool(args.no_auto_mmpd_ckpt)
    if float(args.test_fraction) < 1.0 or args.smoke_test:
        mmpd_args.output_dir = output_dir
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    if args.test_max_items is not None:
        mmpd_args.test_max_items = int(args.test_max_items)

    all_top: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    datasets_root = (REPO_ROOT / "results" / "datasets").resolve()

    for dataset in datasets:
        binary_ckpt = discover_binary_ckpt(args.binary_ckpt_base, dataset, args.binary_ckpt_stem)
        if args.plots_only:
            cache_file = cache_path(output_dir, dataset)
            if not cache_file.is_file():
                raise FileNotFoundError(f"--plots-only but missing {cache_file}")
            cache = load_eval_cache(cache_file)
        else:
            cache = run_or_load_dataset_eval(
                dataset=dataset,
                mmpd_args=mmpd_args,
                binary_ckpt=binary_ckpt,
                binary_config=args.binary_config,
                output_dir=output_dir,
                device=device,
                force_eval=args.force_eval,
                test_fraction=float(args.test_fraction),
                datasets_root=datasets_root,
                auto_mmpd_ckpt=auto_mmpd_ckpt,
            )

        top = select_top_windows(
            cache,
            top_k=args.top_k,
            random_k=args.random_k,
            min_spacing=args.min_spacing,
            diff_mode=args.diff_mode,
            seed=int(mmpd_args.seed) + 17 * (zlib.crc32(dataset.encode("utf-8")) % 10_000),
        )
        all_top[dataset] = top
        top_json = output_dir / "top_windows" / f"{dataset}.json"
        top_json.parent.mkdir(parents=True, exist_ok=True)
        with top_json.open("w", encoding="utf-8") as f:
            json.dump(top, f, indent=2)
        print(f"[rank] {dataset}: wrote {len(top)} picks -> {top_json}")

        summary_file = summary_path(output_dir / "eval_cache", dataset)
        if summary_file.is_file():
            with summary_file.open(encoding="utf-8") as f:
                sm = json.load(f)
            summary_rows.append({
                "dataset": dataset,
                "binary_ckpt": sm.get("binary_ckpt", str(binary_ckpt)),
                "mean_binary_anchor_mse": sm.get("mean_binary_anchor_mse"),
                "mean_mmpd_anchor_mse": sm.get("mean_mmpd_anchor_mse"),
                "mean_error_diff_binary_minus_mmpd": sm.get("mean_error_diff"),
                "n_windows": sm.get("n_windows"),
            })

        if args.skip_plots:
            continue

        mmpd_npz = Path(mmpd_args.output_dir) / "raw" / f"mmpd_{dataset}.npz"
        if not mmpd_npz.is_file():
            raise FileNotFoundError(f"Missing MMPD raw pack for plots: {mmpd_npz}")
        test_stride = int(cache["test_stride"][0]) if "test_stride" in cache else 1
        plot_dataset_windows(
            dataset=dataset,
            binary_ckpt=binary_ckpt,
            binary_config=args.binary_config,
            mmpd_pack_path=mmpd_npz,
            cache=cache,
            top_manifest=top,
            output_dir=output_dir,
            test_stride=test_stride,
            device=device,
            variables_to_plot=args.variables_to_plot,
            jpeg_dpi=args.jpeg_dpi,
        )

    if summary_rows:
        csv_path = output_dir / "dataset_summary.csv"
        fields = list(summary_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[summary] {csv_path}")


if __name__ == "__main__":
    main()
