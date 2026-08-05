#!/usr/bin/env python3
"""L8/L16 candidate-only disc for patch_refine / residual-fine ablation ckpts.

Shared fair protocol (matches eval_univariate_disc_two_ablations_vs_gt):
  - generate final forecasts (sample0) in **global dataset-z**
    (``use_ordinal_window_norm=False`` pack pool; ``prediction_global_norm``)
  - snap GT / binary / MMPD onto the **training lattice** for that leaf:
      * canvas128 / window-norm leaves → finite H-row window-norm grid
        (``legal_window_norm_patch_refine_levels_dataset_z``, H from
        ``patch_refine_canvas_height``, ``max_scale`` from ckpt metadata)
      * ordinal leaves → absolute ordinal patch-refine ladder
        (``legal_patch_refine_levels_dataset_z``)
    Do **not** instance-norm / window-zscore the disc series; rungs may be
    window-specific but values stay in dataset-z.
  - disc preprocess: **bin-center shift only** (zscore hard-off when BC on)

Supports both checkpoint layouts:
  - coarse + patch_refine  (window-norm / canvas128 leaves)
  - coarse + fine          (ordinal residual)

``--viz-only``: skip disc train; write zoomed L8/L16 disc-input panels so the
ladder snap is visually checkable before a Killarney submit.

By default also writes full-horizon staged_eval red-box panels
(``viz/staged_eval_samples/<run>/``) via ``viz_ablation_staged_eval_samples``
helpers (1d hz96 + 2d coarse/fine + per-variate refine_boxes). Opt out with
``--no-redbox-viz``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from temp.eval_univariate_patch_refine_vs_gt import load_patch_refine_run
from utils.disc_bin_center_shift import bin_center_shift, nearest_bin_indices
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    apply_disc_pack_protocol,
    binary_mmpd_train_scaler_map,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    AnchorRun,
    ensure_mmpd_repo,
    load_tsf_pack_pool,
    parse_pack_splits,
    run_mmpd_eval,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.forecast_pack_reduce import reduce_pack_forecast
from utils.patch_refine_ordinal_ladder import (
    assert_on_patch_refine_levels,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.patch_refine_value_grid import (
    legal_window_norm_patch_refine_levels_dataset_z,
)
from utils.staged_binary_forecast import generate_staged_forecast
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)
from utils.visualize_staged_forecast import _load_staged_bundle


DEFAULT_MMPD = (
    REPO_ROOT / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
)
# Default: canvas128 coarser ladder leaf (override --runs after train / for legacy 256 ckpts).
DEFAULT_RUNS = (
    "window_norm_c128:results/ckpts/PLACEHOLDER-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6:"
    "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="name:ckpt_root:config triples (coarse+patch_refine or coarse+fine layouts)",
    )
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    # Pack pool: paper-test windows at stride 4 by default (MMPD-aligned).
    # Use --pack-splits val,test for paper val+test combined pool (80/20 disc carve).
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument(
        "--pack-splits",
        default="test",
        help="TSF splits forming the disc pool (comma or space). "
        "val,test → combined pool; defaults train/val frac to 0.8/0 if still 0.7/0.15.",
    )
    p.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    p.add_argument("--mmpd-backbone", default="Decoder")
    p.add_argument(
        "--mmpd-sample-num",
        type=int,
        default=1,
        help="MMPD stochastic draws when rematerializing non-test packs (sample0 only needs 1).",
    )
    p.add_argument("--force-mmpd-eval", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--probabilistic-sampler", default="quad_t")
    p.add_argument("--raw-binary-batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-raw-eval", action="store_true")
    # Shared forecast packs (binary + MMPD) so later disc runs skip regeneration.
    # Default dir: results/datasets/disc_forecast_cache/<keyed>.npz
    p.add_argument(
        "--forecast-cache-dir",
        type=Path,
        default=REPO_ROOT / "results" / "datasets" / "disc_forecast_cache",
        help="Shared keyed forecast packs (ckpt+pack+protocol). "
        "Auto-load when compatible; always write after generate.",
    )
    p.add_argument(
        "--require-forecast-cache",
        "--reuse-forecast-cache",
        action="store_true",
        help="Fail fast if a compatible shared/local forecast pack is missing "
        "(no regenerate). Pass as --reuse-forecast-cache or --require-forecast-cache.",
    )
    p.add_argument(
        "--no-forecast-cache",
        action="store_true",
        help="Disable shared forecast cache (only use output_dir/raw).",
    )
    p.add_argument(
        "--disc-disagreement-max",
        type=int,
        default=12,
        help="Max panels per disagreement direction (mmpd_wrong_binary_right / "
        "binary_wrong_mmpd_right). Default 12.",
    )
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=1.0)
    p.add_argument("--disc-index-stride", type=int, default=1)
    # candidate_only: disc sees L-slice only (no lookback continuity).
    p.add_argument("--candidate-only", action="store_true", default=True)
    p.add_argument("--no-candidate-only", action="store_false", dest="candidate_only")
    # Per-slice integer bin mean-centering (no zscore) — campaign default.
    p.add_argument("--disc-bin-center-shift", action="store_true", default=True)
    p.add_argument("--no-disc-bin-center-shift", action="store_false", dest="disc_bin_center_shift")
    p.add_argument("--disc-bin-center-reduce", default="per_variate")
    p.add_argument(
        "--disc-apply-zscore",
        action="store_true",
        default=False,
        help="Legacy per-slice zscore_time. Mutually exclusive with bin-center; "
        "default off. Enabling this with --disc-bin-center-shift fails fast.",
    )
    p.add_argument("--nonoverlapping-patches", action="store_true", default=False)
    p.add_argument("--no-offset-embedding", action="store_true", default=False)
    p.add_argument("--offset-stride", type=int, default=1)
    p.add_argument(
        "--unique-absolute-slices",
        action="store_true",
        default=True,
        help="One random (window,offset) per absolute L-block across overlapping "
        "96-horizons (UniquePatchSegmentDataset-style). Default on.",
    )
    p.add_argument(
        "--no-unique-absolute-slices",
        action="store_false",
        dest="unique_absolute_slices",
        help="Dense Cartesian product of windows × in-horizon offsets (old slow path).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=256)
    # Disc fractions *inside* the pack (not paper borders). Default 70/15/15.
    # For val+test pool 80/20: --pack-splits val,test --train-fraction 0.8 --val-fraction 0
    # (val carved from last 10% of purged train pool; see split_windows).
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of pack for val (default 0.15). <=0 → carve early-stop val "
        "from train pool (use with --train-fraction 0.8 for 80/20 test).",
    )
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument("--force-train", action="store_true", default=True)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--canvas-height",
        type=int,
        default=None,
        help="Absolute patch-refine ladder rows. Default: read from each run's config "
        "(patch_refine_canvas_height). Fail if missing.",
    )
    p.add_argument(
        "--viz-only",
        action="store_true",
        help="Generate zoomed L8/L16 disc-input lattice panels; skip disc training.",
    )
    p.add_argument("--viz-n-windows", type=int, default=2)
    p.add_argument("--viz-variate", type=int, default=0)
    p.add_argument("--viz-zoom-steps", type=int, default=12)
    p.add_argument(
        "--redbox-viz",
        action="store_true",
        default=True,
        help="Write full-horizon staged_eval red-box panels (default on).",
    )
    p.add_argument(
        "--no-redbox-viz",
        action="store_false",
        dest="redbox_viz",
        help="Skip full-horizon red-box staged_eval panels.",
    )
    p.add_argument(
        "--redbox-n-samples",
        type=int,
        default=10,
        help="Pool windows for red-box / 1d hz96 staged_eval panels.",
    )
    p.add_argument(
        "--redbox-sampler",
        default="quad_t",
        choices=("anchor", "quad_t", "ddim", "ddim_quad"),
    )
    p.add_argument("--redbox-num-sampling-steps", type=int, default=20)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    if bool(getattr(args, "disc_bin_center_shift", False)) and bool(
        getattr(args, "disc_apply_zscore", False)
    ):
        raise ValueError(
            "disc_bin_center_shift and disc_apply_zscore are mutually exclusive "
            "(campaign path is BC-only; leave --disc-apply-zscore off)"
        )
    if not args.smoke_test:
        return
    args.max_windows = min(int(args.max_windows or 4), 4)
    args.num_sampling_steps = min(int(args.num_sampling_steps), 2)
    args.epochs = min(int(args.epochs), 2)
    args.viz_n_windows = min(int(args.viz_n_windows), 2)
    args.raw_binary_batch_size = 1
    args.redbox_n_samples = min(int(args.redbox_n_samples), 1)


def _parse_run_specs(specs: Sequence[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for spec in specs:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(f"bad --runs entry (want name:ckpt:config): {spec}")
        out.append({"name": parts[0], "ckpt": parts[1], "config": parts[2]})
    return out


def _load_fine_run(dataset: str, checkpoint_dir: Path) -> Tuple[AnchorRun, Dict[str, Path]]:
    bundle = _load_staged_bundle(checkpoint_dir, dataset)
    if str(bundle.get("stage")) == "vertical_dual":
        raise ValueError(f"{checkpoint_dir}: vertical_dual not supported here")
    meta = dict(bundle["fine_metadata"])
    meta["dataset_name"] = dataset
    meta["dataset"] = dataset
    run = AnchorRun(
        variant="binary_coarse_fine",
        dataset=dataset,
        root=checkpoint_dir,
        subset_dir=Path(bundle["coarse_pt"]).parent.parent,
        best_pt=Path(bundle["fine_pt"]),
        itrans_pt=None,
        metadata=meta,
    )
    return run, {
        "coarse_pt": Path(bundle["coarse_pt"]),
        "refine_pt": Path(bundle["fine_pt"]),
        "stage": "fine",
    }


def load_ablation_run(
    dataset: str,
    checkpoint_dir: Path,
) -> Tuple[AnchorRun, Dict[str, Path], str]:
    """Return (run, stages, kind) where kind is patch_refine|fine."""
    try:
        run, stages = load_patch_refine_run(dataset, checkpoint_dir, test_stride=None)
        stages = dict(stages)
        stages["stage"] = "patch_refine"
        return run, stages, "patch_refine"
    except FileNotFoundError:
        run, stages = _load_fine_run(dataset, checkpoint_dir)
        return run, stages, "fine"


def write_redbox_forecast_viz(
    *,
    args: argparse.Namespace,
    run_name: str,
    ckpt_root: Path,
    config_path: str,
    device: torch.device,
) -> List[str]:
    """Full-horizon 1d + 2d + per-variate red-box panels via staged_eval helpers."""
    # Lazy import: viz module imports load_ablation_run from this file.
    from temp.viz_ablation_staged_eval_samples import _pick_indices, viz_run

    run, _, _ = load_ablation_run(str(args.dataset), ckpt_root)
    pool_vars = run_variate_indices(run)
    out_root = Path(args.output_dir) / "viz" / "staged_eval_samples"
    out_root.mkdir(parents=True, exist_ok=True)
    n_samples = int(args.redbox_n_samples)
    if n_samples < 1:
        raise ValueError(f"--redbox-n-samples must be >= 1, got {n_samples}")

    print(
        f"[{run_name}] redbox-viz: loading pack pool vars={pool_vars} "
        f"n_samples={n_samples} sampler={args.redbox_sampler}",
        flush=True,
    )
    pool, _starts, _splits, _, _ = load_tsf_pack_pool(
        str(args.dataset),
        pool_vars,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        train_stride=1,
        test_stride=int(args.pack_test_stride),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    picks = _pick_indices(len(pool), n_samples, int(args.seed), None)
    viz_args = SimpleNamespace(
        dataset=str(args.dataset),
        output_root=out_root,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        pack_test_stride=int(args.pack_test_stride),
        pack_splits=str(args.pack_splits),
        n_samples=n_samples,
        seed=int(args.seed),
        pool_indices=None,
        variables_to_plot=0,  # all variates
        jpeg_dpi=120,
        num_sampling_steps=int(args.redbox_num_sampling_steps),
        sampler=str(args.redbox_sampler),
        device=str(device),
        code_root=None,
        skip_existing_runs=False,
    )
    paths = viz_run(
        viz_args,
        run_name=run_name,
        ckpt_root=ckpt_root,
        config_path=config_path,
        device=device,
        picks=picks,
        pool=pool,
    )
    if not paths:
        raise RuntimeError(f"{run_name}: redbox-viz wrote zero panels under {out_root}")
    print(f"[{run_name}] redbox-viz wrote {len(paths)} panels under {out_root / run_name}", flush=True)
    return [str(p) for p in paths]


def _mmpd_pack(root: Path, dataset: str) -> Dict[str, np.ndarray]:
    path = root / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing MMPD pack: {path}")
    with np.load(path) as data:
        pack = {key: data[key] for key in data.files}
    for key in ("y_true", "samples", "indices"):
        if key not in pack:
            raise KeyError(f"{path} missing {key}")
    return pack


def _subset_aligned(
    indices: Sequence[int],
    pack: Mapping[str, np.ndarray],
    pick: np.ndarray,
) -> Tuple[List[int], Dict[str, np.ndarray]]:
    pick = np.asarray(pick, dtype=np.int64)
    thinned_indices = [int(indices[int(i)]) for i in pick.tolist()]
    n_full = len(indices)
    thinned = {
        key: (
            value[pick]
            if isinstance(value, np.ndarray) and value.shape[:1] == (n_full,)
            else value
        )
        for key, value in pack.items()
    }
    return thinned_indices, thinned


def _thin_windows(
    indices: Sequence[int],
    pack: Mapping[str, np.ndarray],
    *,
    max_windows: Optional[int],
    seed: int,
) -> Tuple[List[int], Dict[str, np.ndarray]]:
    n = len(indices)
    if max_windows is None or max_windows >= n:
        return list(indices), dict(pack)
    rng = np.random.default_rng(int(seed))
    pick = np.sort(rng.choice(n, size=int(max_windows), replace=False))
    return _subset_aligned(indices, pack, pick)


def _pack_strides(
    args: argparse.Namespace,
    run: AnchorRun,
    pack_splits: Sequence[str],
) -> Tuple[int, int]:
    """Train/test strides for the TSF pack pool.

    Multi-split packs (e.g. val,test) use ``pack_test_stride`` for *both* so
    val density matches the MMPD-aligned test grid (fail-fast if ambiguous).
    """
    test_stride = int(args.pack_test_stride)
    if list(pack_splits) == ["test"]:
        return int(run_train_stride(run)), test_stride
    return test_stride, test_stride


def _pack_tag(pack_splits: Sequence[str]) -> str:
    return "-".join(pack_splits)


def _forecast_protocol_tag(args: argparse.Namespace) -> str:
    """Stable protocol fingerprint for shared forecast packs."""
    return (
        f"lb{int(args.lookback)}_hz{int(args.horizon)}"
        f"_stride{int(args.pack_test_stride)}"
        f"_steps{int(args.num_sampling_steps)}"
        f"_{args.probabilistic_sampler}"
        f"_agg{args.fake_agg}"
    )


def _binary_forecast_cache_name(
    *,
    run_name: str,
    dataset: str,
    ckpt_root: Path,
    pack_splits: Sequence[str],
    args: argparse.Namespace,
) -> str:
    return (
        f"binary_{run_name}_{dataset}_{_pack_tag(pack_splits)}"
        f"__{ckpt_root.name}__{_forecast_protocol_tag(args)}.npz"
    )


def _mmpd_forecast_cache_name(
    *,
    dataset: str,
    pack_splits: Sequence[str],
    args: argparse.Namespace,
) -> str:
    # MMPD donor is mmpd-output-root; key by dataset + pack + stride (not binary ckpt).
    return (
        f"mmpd_{dataset}_{_pack_tag(pack_splits)}"
        f"__stride{int(args.pack_test_stride)}"
        f"__samples{int(getattr(args, 'mmpd_sample_num', 1) or 1)}.npz"
    )


def _shared_cache_dir(args: argparse.Namespace) -> Optional[Path]:
    if bool(getattr(args, "no_forecast_cache", False)):
        return None
    path = Path(getattr(args, "forecast_cache_dir"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _try_load_pack(
    path: Path,
    *,
    indices: Optional[Sequence[int]] = None,
    label: str,
) -> Optional[Dict[str, np.ndarray]]:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as data:
        pack = {key: data[key] for key in data.files}
    if indices is not None and not np.array_equal(
        pack.get("indices"), np.asarray(indices, dtype=np.int64)
    ):
        print(
            f"[{label}] cache index mismatch at {path} "
            f"(have {len(pack.get('indices', []))} want {len(indices)}); ignoring",
            flush=True,
        )
        return None
    print(f"[{label}] reusing cached {path}", flush=True)
    return pack


def _write_pack(path: Path, pack: Mapping[str, np.ndarray], *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dict(pack))
    print(f"[{label}] wrote {path}", flush=True)


def _rematerialize_mmpd_pack(
    args: argparse.Namespace,
    *,
    run: AnchorRun,
    indices: Sequence[int],
    pack_splits: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Run MMPD eval on ``indices`` into output_dir/raw (uses donor ckpts via mmpd-output-root)."""
    dataset = run.dataset
    local_cache = args.output_dir / "raw" / f"mmpd_{dataset}_{_pack_tag(pack_splits)}.npz"
    shared_dir = _shared_cache_dir(args)
    shared_cache = (
        None
        if shared_dir is None
        else shared_dir / _mmpd_forecast_cache_name(
            dataset=dataset, pack_splits=pack_splits, args=args,
        )
    )

    if not args.force_mmpd_eval and not args.force_raw_eval:
        pack = _try_load_pack(local_cache, indices=indices, label="mmpd")
        if pack is not None:
            return pack
        if shared_cache is not None:
            pack = _try_load_pack(shared_cache, indices=indices, label="mmpd-shared")
            if pack is not None:
                _write_pack(local_cache, pack, label="mmpd")
                return pack

    if bool(getattr(args, "require_forecast_cache", False)):
        raise FileNotFoundError(
            f"require-forecast-cache: missing MMPD pack "
            f"(local={local_cache}, shared={shared_cache})"
        )

    mmpd_repo = Path(args.mmpd_repo)
    if not mmpd_repo.is_dir():
        # Killarney ordinal-fine worktree often lacks a local clone; fall back to main scratch.
        alt = Path(os.environ.get("SCRATCH", "/scratch")) / "ts-sandbox" / "temp" / "MMPD"
        if alt.is_dir():
            mmpd_repo = alt
            args.mmpd_repo = alt
            print(f"[mmpd] using fallback repo {alt}", flush=True)
        else:
            raise FileNotFoundError(
                f"MMPD repo missing at {args.mmpd_repo} (and {alt}); "
                "clone to temp/MMPD or pass --mmpd-repo"
            )
    ensure_mmpd_repo(mmpd_repo, update=False)

    # run_mmpd_eval writes output_dir/raw/mmpd_{dataset}.npz — point output_dir
    # at a temp subdir then copy/rename to the pack-tagged cache.
    eval_ns = SimpleNamespace(**vars(args))
    eval_ns.mmpd_repo = mmpd_repo
    eval_ns.pack_splits = ",".join(pack_splits)
    eval_ns.sample_num = int(getattr(args, "mmpd_sample_num", 1))
    eval_ns.gmm_components = int(getattr(args, "gmm_components", 1) or 1)
    eval_ns.gmm_iterations = int(getattr(args, "gmm_iterations", 1) or 1)
    eval_ns.force_mmpd_eval = True
    eval_ns.no_update_mmpd = True
    eval_ns.mmpd_output_root = Path(args.mmpd_output_root)
    eval_ns.eval_test_stride = int(args.pack_test_stride)
    eval_ns.test_stride = int(args.pack_test_stride)
    eval_ns.mmpd_eval_batch_size = int(getattr(args, "mmpd_eval_batch_size", 16) or 16)
    eval_ns.mmpd_batch_size = int(getattr(args, "mmpd_batch_size", 32) or 32)
    eval_ns.mmpd_config_suffix = getattr(args, "mmpd_config_suffix", None)
    eval_ns.force_mmpd_train = False
    eval_ns.patch_size = getattr(args, "patch_size", None)
    eval_ns.cpu = bool(getattr(args, "cpu", False))
    eval_ns.gpu = int(getattr(args, "gpu", 0) or 0)

    print(
        f"[mmpd] rematerializing pack_splits={pack_splits} windows={len(indices)} "
        f"-> {local_cache}",
        flush=True,
    )
    pack = run_mmpd_eval(eval_ns, run, indices)
    # Persist under pack-tagged name (and keep canonical name for run_mmpd_eval reuse).
    _write_pack(local_cache, pack, label="mmpd")
    if shared_cache is not None:
        _write_pack(shared_cache, pack, label="mmpd-shared")
    return pack


def _binary_lattice_atol(legal_levels: np.ndarray) -> float:
    gaps = np.diff(np.sort(np.asarray(legal_levels, dtype=np.float64), axis=-1), axis=-1)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return 1e-4
    return float(max(1e-4, 0.25 * float(np.min(positive))))


def _ladder_only(
    *,
    dataset: str,
    run: AnchorRun,
    lookback: int,
    horizon: int,
) -> Any:
    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=1e-6,
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{dataset}: ordinal ladder missing (needed for patch-refine snap)")
    return ladder


def _canvas_height_from_state(state: Any, override: Optional[int]) -> int:
    # Not guessed: read patch_refine_canvas_height (256 legacy / 128 coarser leaf)
    # or pass --canvas-height.
    if override is not None and int(override) > 0:
        return int(override)
    h = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    if h <= 0:
        raise RuntimeError(
            "patch_refine_canvas_height missing/invalid; pass --canvas-height or set it in YAML"
        )
    return h


def _max_scale_from_ckpt_metadata(ckpt_root: Path, dataset: str) -> float:
    """Fail-fast: max_scale from patch_refine (or fine) tuned_params metadata."""
    root = Path(ckpt_root)
    candidates: List[Path] = []
    # subset_id dirs under ckpt root
    for subset in sorted(root.iterdir()) if root.is_dir() else []:
        if not subset.is_dir():
            continue
        for stage in ("patch_refine", "fine", "coarse"):
            meta = subset / stage / "metadata.json"
            if meta.is_file():
                candidates.append(meta)
    # Also allow flat metadata next to best.pt
    for stage in ("patch_refine", "fine"):
        meta = root / stage / "metadata.json"
        if meta.is_file():
            candidates.append(meta)
    seen = set()
    for meta_path in candidates:
        key = str(meta_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Prefer matching dataset when present.
        ds = str(meta.get("dataset_name") or meta.get("dataset") or "")
        if ds and ds != dataset:
            continue
        tuned = meta.get("tuned_params") or {}
        if "max_scale" in tuned:
            ms = float(tuned["max_scale"])
            if ms <= 0.0:
                raise RuntimeError(f"{meta_path}: tuned_params.max_scale={ms} invalid")
            return ms
    raise RuntimeError(
        f"{ckpt_root}: cannot resolve max_scale for dataset={dataset!r} "
        f"(need tuned_params.max_scale in patch_refine/fine metadata.json)"
    )


def _window_norm_grid_config(
    state: Any,
    *,
    canvas_height: int,
    max_scale: float,
    skip_window_norm_variate_mask: Optional[List[bool]] = None,
) -> SimpleNamespace:
    if bool(getattr(state, "use_ordinal_window_norm", False)):
        raise RuntimeError("window-norm grid config requested for ordinal leaf")
    if not bool(getattr(state, "use_window_normalization", False)):
        raise RuntimeError(
            "canvas128 disc snap expects use_window_normalization=True; "
            "got False — refusing silent fallback"
        )
    mask = skip_window_norm_variate_mask
    if mask is None:
        mask = getattr(state, "skip_window_norm_variate_mask", None)
    if mask is None:
        extra = getattr(state, "extra", None) or {}
        hybrid = extra.get("hybrid_flat_norm_stats") or {}
        if hybrid.get("flat_variate_mask") is not None:
            mask = [bool(x) for x in hybrid["flat_variate_mask"]]
    return SimpleNamespace(
        use_ordinal_window_norm=False,
        use_window_normalization=True,
        window_norm_center=str(getattr(state, "window_norm_center", "mean")),
        window_norm_std_floor=float(getattr(state, "window_norm_std_floor", 0.1)),
        window_norm_low_var_threshold=float(
            getattr(state, "window_norm_low_var_threshold", 0.0)
        ),
        window_norm_low_var_unit_std=float(
            getattr(state, "window_norm_low_var_unit_std", 1.0)
        ),
        window_norm_low_var_unit_std_per_variate=getattr(
            state, "window_norm_low_var_unit_std_per_variate", None
        ),
        skip_window_norm_variate_mask=list(mask) if mask is not None else None,
        patch_refine_canvas_height=int(canvas_height),
        max_scale=float(max_scale),
    )


def _flat_mask_from_ckpt(ckpt_root: Path, dataset: str) -> Optional[List[bool]]:
    """Read hybrid flat mask from patch_refine/fine metadata (fail-closed if inconsistent)."""
    meta_paths: List[Path] = []
    for stage in ("patch_refine", "fine", "coarse"):
        meta_paths.extend(
            [
                ckpt_root / dataset / stage / "metadata.json",
                ckpt_root / stage / "metadata.json",
            ]
        )
        for base in (ckpt_root / stage, ckpt_root / dataset / stage):
            if base.is_dir():
                meta_paths.extend(base.glob("*/metadata.json"))
    for meta_path in meta_paths:
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if not meta.get("hybrid_flat_dataset_norm"):
            continue
        mask = meta.get("flat_variate_mask")
        if mask is None:
            raise RuntimeError(f"{meta_path}: hybrid_flat_dataset_norm set but flat_variate_mask missing")
        return [bool(x) for x in mask]
    return None


def _legal_levels_for_run(
    past: np.ndarray,
    *,
    state: Any,
    ckpt_root: Path,
    dataset: str,
    canvas_height: int,
    ladder: Any,
    device: torch.device,
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Pick the lattice that matches the binary training leaf (not a foreign ordinal one)."""
    h = int(canvas_height)
    if bool(getattr(state, "use_ordinal_window_norm", False)):
        if ladder is None:
            raise RuntimeError(f"{dataset}: ordinal ladder required for ordinal snap")
        levels = legal_patch_refine_levels_dataset_z(
            past, ladder=ladder, canvas_height=h, device=device,
        )
        return levels, "ordinal_absolute", {"canvas_height": float(h)}

    # Window-norm canvas128 (and non-ordinal window-norm) leaves.
    max_scale = _max_scale_from_ckpt_metadata(ckpt_root, dataset)
    flat_mask = _flat_mask_from_ckpt(ckpt_root, dataset)
    grid_cfg = _window_norm_grid_config(
        state,
        canvas_height=h,
        max_scale=max_scale,
        skip_window_norm_variate_mask=flat_mask,
    )
    levels = legal_window_norm_patch_refine_levels_dataset_z(past, grid_cfg)
    snap_mode = "window_norm_grid_hybrid_flat" if flat_mask and any(flat_mask) else "window_norm_grid"
    meta = {
        "canvas_height": float(h),
        "max_scale": float(max_scale),
        "window_norm_std_floor": float(grid_cfg.window_norm_std_floor),
    }
    if flat_mask is not None:
        meta["n_flat_variates"] = float(sum(1 for x in flat_mask if x))
    return levels, snap_mode, meta


def _load_models(
    *,
    dataset: str,
    ckpt_root: Path,
    config_path: str,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> Tuple[AnchorRun, Any, Any, Any, str, int]:
    run, stages, kind = load_ablation_run(dataset, ckpt_root)
    state = _build_state(ckpt_root, dataset, run_subset_id(run), config_path)
    resolve_pipeline_data_subset(state)
    flat_mask = _flat_mask_from_ckpt(ckpt_root, dataset)
    if flat_mask is not None:
        state.extra["hybrid_flat_norm_stats"] = {"flat_variate_mask": flat_mask}
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    canvas_height = _canvas_height_from_state(state, None)
    ladder = None
    # Model-side ladder only when the checkpoint itself was ordinal-trained.
    if bool(state.use_ordinal_window_norm):
        ladder = _ladder_only(
            dataset=dataset, run=run, lookback=lookback, horizon=horizon,
        )
        state.extra["global_ordinal_ladder"] = ladder
        pipeline_mod.GLOBAL_ORDINAL_LADDER = ladder
    else:
        state.extra.pop("global_ordinal_ladder", None)
        pipeline_mod.GLOBAL_ORDINAL_LADDER = None
        # Window-norm canvas128 leaves: disc snap uses window-norm grid, not ordinal.
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    guidance = None
    if bool(state.use_guidance_channel) or not bool(state.disable_cross_attention):
        path, guidance_type = _resolve_guidance_ckpt(ckpt_root, run_subset_id(run), "auto")
        guidance = load_wrapped_guidance(
            str(path),
            len(run_variate_indices(run)),
            device,
            guidance_type=guidance_type,
            dataset_lookback=lookback,
            dataset_horizon=horizon,
        )
        if hasattr(guidance, "ordinal_ladder") and bool(state.use_ordinal_window_norm):
            guidance.ordinal_ladder = ladder

    refine_stage = "patch_refine" if kind == "patch_refine" else "fine"
    coarse = _load_stage_model(
        state, "coarse", stages["coarse_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    refine = _load_stage_model(
        state, refine_stage, stages["refine_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    for model in (coarse, refine):
        if bool(state.use_ordinal_window_norm):
            model._ordinal_input_is_ranked = False
            model._ordinal_apply_ood_shift = True
        if flat_mask is not None:
            model.config.skip_window_norm_variate_mask = list(flat_mask)
            model.config.hybrid_flat_dataset_norm = True
    return run, coarse, refine, ladder, kind, canvas_height


def materialize_binary_pack(
    args: argparse.Namespace,
    *,
    dataset: str,
    run_name: str,
    ckpt_root: Path,
    config_path: str,
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], AnchorRun, Any, str, int]:
    pack_splits = parse_pack_splits(args.pack_splits)
    local_cache = args.output_dir / "raw" / (
        f"binary_{run_name}_{dataset}_{_pack_tag(pack_splits)}.npz"
    )
    shared_dir = _shared_cache_dir(args)
    shared_cache = (
        None
        if shared_dir is None
        else shared_dir / _binary_forecast_cache_name(
            run_name=run_name,
            dataset=dataset,
            ckpt_root=ckpt_root,
            pack_splits=pack_splits,
            args=args,
        )
    )

    def _hydrate_from_pack(pack: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], AnchorRun, Any, str, int]:
        run, _stages, kind = load_ablation_run(dataset, ckpt_root)
        kind = str(pack.get("kind", [kind])[0]) if "kind" in pack else kind
        state = _build_state(ckpt_root, dataset, run_subset_id(run), config_path)
        if "canvas_height" in pack:
            cached_h = int(np.asarray(pack["canvas_height"]).reshape(-1)[0])
        else:
            cached_h = 0
        canvas_height = _canvas_height_from_state(
            state, getattr(args, "canvas_height", None) or (cached_h or None),
        )
        ladder = None
        if bool(getattr(state, "use_ordinal_window_norm", False)):
            ladder = _ladder_only(
                dataset=dataset,
                run=run,
                lookback=args.lookback,
                horizon=args.horizon,
            )
        return pack, run, ladder, kind, canvas_height

    if not args.force_raw_eval:
        pack = _try_load_pack(local_cache, indices=indices, label=f"binary/{run_name}")
        if pack is not None:
            return _hydrate_from_pack(pack)
        if shared_cache is not None:
            pack = _try_load_pack(
                shared_cache, indices=indices, label=f"binary-shared/{run_name}",
            )
            if pack is not None:
                _write_pack(local_cache, pack, label=f"binary/{run_name}")
                return _hydrate_from_pack(pack)

    if bool(getattr(args, "require_forecast_cache", False)):
        raise FileNotFoundError(
            f"require-forecast-cache: missing binary pack "
            f"(local={local_cache}, shared={shared_cache})"
        )

    run, coarse, refine, ladder, kind, canvas_height = _load_models(
        dataset=dataset,
        ckpt_root=ckpt_root,
        config_path=config_path,
        lookback=args.lookback,
        horizon=args.horizon,
        device=device,
    )
    if getattr(args, "canvas_height", None) is not None:
        canvas_height = int(args.canvas_height)
        if canvas_height <= 0:
            raise RuntimeError(f"bad --canvas-height {canvas_height}")
    # Always dataset-z windows (never mixed train-rank / test-z). Absolute
    # series_starts come from paper borders + stride. Multi-split packs
    # (val,test) use pack_test_stride for both parts so density matches MMPD.
    train_stride, test_stride = _pack_strides(args, run, pack_splits)
    pool, starts, splits, _, _ = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=train_stride,
        test_stride=test_stride,
        pack_splits=pack_splits,
        use_ordinal_window_norm=False,
    )
    if not indices or min(indices) < 0 or max(indices) >= len(pool):
        raise ValueError(
            f"{dataset}/{run_name}: indices outside pack pool "
            f"(n={len(indices)}, pool_len={len(pool)}, stride={args.pack_test_stride})"
        )
    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=max(1, int(args.raw_binary_batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    past_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    samples_all: List[np.ndarray] = []
    n_batches = len(loader)
    print(
        f"[{run_name}/{dataset}] materializing: windows={len(indices)} "
        f"batches={n_batches} steps={args.num_sampling_steps} sampler={args.probabilistic_sampler} "
        f"canvas_height={canvas_height}",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            past = past.to(device)
            future = future.to(device)
            overlap = int(getattr(refine.config, "lookback_overlap", 0) or 0)
            target = future[..., overlap:] if overlap else future
            torch.manual_seed(int(args.seed) + batch_idx * 1009)
            result = generate_staged_forecast(
                coarse,
                refine,
                past,
                vertical_dual=False,
                sampler=args.probabilistic_sampler,
                num_inference_steps=int(args.num_sampling_steps),
            )
            # Dataset-z forecast (same coordinate family as pack y_true). Window-norm
            # ckpts still load their own training flags for sampling; disc snap uses
            # the matching window-norm H-row grid (not a foreign ordinal ladder).
            pred = result["prediction_global_norm"]
            if pred.shape != target.shape:
                raise RuntimeError(
                    f"pred/target mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
                )
            past_all.append(past.detach().cpu().numpy().astype(np.float32))
            y_true_all.append(target.detach().cpu().numpy().astype(np.float32))
            # Store sample0 draw as S=1 for reduce_pack_forecast(fake_agg=sample0).
            samples_all.append(pred.detach().cpu().numpy().astype(np.float32)[:, :, None, :])
            if (batch_idx + 1) == n_batches or (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(
                    f"[{run_name}/{dataset}] generate {batch_idx + 1}/{n_batches}",
                    flush=True,
                )
    pack = {
        # past (N,V,Lb), y_true (N,V,H), samples (N,V,1,H); indices + series_starts
        # keep absolute CSV alignment with the MMPD pack rows.
        "past": np.concatenate(past_all, axis=0).astype(np.float32),
        "y_true": np.concatenate(y_true_all, axis=0).astype(np.float32),
        "samples": np.concatenate(samples_all, axis=0).astype(np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": np.asarray(starts, dtype=np.int64)[np.asarray(indices, dtype=np.int64)],
        "pack_splits": np.asarray(list(splits) if not isinstance(splits, dict) else list(splits.keys()), dtype=object),
        "kind": np.asarray([kind]),
        "canvas_height": np.asarray([int(canvas_height)], dtype=np.int64),
    }
    _write_pack(local_cache, pack, label=f"binary/{run_name}")
    if shared_cache is not None:
        _write_pack(shared_cache, pack, label=f"binary-shared/{run_name}")
    print(f"[{run_name}/{dataset}] materialize done in {time.time() - t0:.1f}s", flush=True)
    return pack, run, ladder, kind, canvas_height


def _snap_bundle(
    *,
    binary_pack: Mapping[str, np.ndarray],
    mmpd_pack: Mapping[str, np.ndarray],
    run: AnchorRun,
    ladder: Any,
    args: argparse.Namespace,
    device: torch.device,
    canvas_height: int,
    ckpt_root: Path,
    config_path: str,
) -> Dict[str, np.ndarray]:
    """Align binary + MMPD onto one discrete alphabet (training lattice).

    Fair for “same lattice?” disc work — and it also destroys sub-rung signal.
    If models mostly differ from GT by sub-bin noise, AUROC collapses toward
    chance by design after this step.
    """
    binary_gt = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_pred = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    mmpd_gt = np.asarray(mmpd_pack["y_true"], dtype=np.float32)
    mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
    # Fail hard if thinning left mismatched index lists.
    if not np.array_equal(binary_pack["indices"], mmpd_pack["indices"]):
        raise RuntimeError("binary/MMPD indices differ after thinning")
    # Affine map MMPD z → binary train-scaler z (from saved scalers, not eval GT).
    scalers = binary_mmpd_train_scaler_map(args, run)
    mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_gt,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_pred,
        **scalers,
    )
    past = np.asarray(binary_pack["past"], dtype=np.float32)
    h = int(canvas_height)
    if h <= 0:
        raise RuntimeError(f"canvas_height must be positive, got {h}")

    state = _build_state(ckpt_root, str(args.dataset), run_subset_id(run), config_path)
    legal_levels, snap_mode, snap_meta = _legal_levels_for_run(
        past,
        state=state,
        ckpt_root=Path(ckpt_root),
        dataset=str(args.dataset),
        canvas_height=h,
        ladder=ladder,
        device=device,
    )
    print(
        f"  snap mode={snap_mode} canvas_height={h} meta={snap_meta} "
        f"(dataset-z values; no instance-norm on disc series)",
        flush=True,
    )
    # Nearest-rung snap for GT, aligned MMPD, and binary forecast.
    gt, gt_snap = snap_to_patch_refine_levels(binary_gt, legal_levels)
    mmpd, mmpd_snap = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
    atol = _binary_lattice_atol(legal_levels)
    binary_raw = np.asarray(binary_pred, dtype=np.float32)
    binary, binary_snap = snap_to_patch_refine_levels(binary_raw, legal_levels)
    raw_err = float(np.abs(binary_raw - binary).max(initial=0.0))
    if raw_err > atol:
        print(
            f"  binary off lattice max_error={raw_err:.6g} atol={atol:.6g}; "
            f"snapping (mean_abs_delta={binary_snap['mean_abs_snap_delta']:.6g})",
            flush=True,
        )
    elif snap_mode.startswith("window_norm_grid") and binary_snap["mean_abs_snap_delta"] > 1e-4:
        # Window-norm binary should already sit on its training grid after denorm.
        print(
            f"  warn: window_norm binary mean_abs_snap_delta="
            f"{binary_snap['mean_abs_snap_delta']:.6g} (expected ~0)",
            flush=True,
        )
    lattice = {
        "gt": assert_on_patch_refine_levels(gt, legal_levels),
        "binary": assert_on_patch_refine_levels(binary, legal_levels),
        "mmpd": assert_on_patch_refine_levels(mmpd, legal_levels),
        "gt_snap": gt_snap,
        "binary_snap": binary_snap,
        "mmpd_snap": mmpd_snap,
        "mmpd_align": align,
        "raw_binary_max_error": raw_err,
        "support_atol": atol,
        "canvas_height": h,
        "snap_mode": snap_mode,
        "snap_meta": snap_meta,
    }
    return {
        "gt": gt,
        "binary": binary,
        "mmpd": mmpd,
        "past": past,
        "legal_levels": np.asarray(legal_levels, dtype=np.float32),
        "indices": np.asarray(binary_pack["indices"], dtype=np.int64),
        "series_starts": np.asarray(binary_pack["series_starts"], dtype=np.int64),
        "lattice": lattice,
        "canvas_height": h,
        "snap_mode": snap_mode,
    }


def _snap_residual(values_1d: np.ndarray, levels_1d: np.ndarray) -> float:
    vals = np.asarray(values_1d, dtype=np.float32)
    lev = np.asarray(levels_1d, dtype=np.float32)
    return float(np.abs(vals[:, None] - lev[None, :]).min(axis=1).max(initial=0.0))


def _plot_snap_proof_panel(
    *,
    out_path: Path,
    title: str,
    levels_1d: np.ndarray,
    series: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    t0: int = 0,
) -> Dict[str, float]:
    """Marker + occupied-rung proof that values sit on the absolute ladder.

    Drawing *all* rungs on a dense canvas looks continuous, and steps-post
    verticals cross between rungs — both make a true snap look wrong. Here we
    only draw occupied legal levels and plot markers (no step verticals), plus
    an integer bin-index panel that cannot lie.
    """
    names = list(series.keys())
    y_stack = np.concatenate([np.asarray(series[n], dtype=np.float64) for n in names])
    x = np.arange(t0, t0 + int(y_stack.size // len(names)))
    # per-series length check
    length = int(next(iter(series.values())).shape[0])
    x = np.arange(t0, t0 + length)
    n_rows = int(np.asarray(levels_1d).shape[0])

    residuals = {n: _snap_residual(series[n], levels_1d) for n in names}
    max_err = float(max(residuals.values()))
    if max_err > 1e-5:
        raise RuntimeError(f"{title}: snap residual {max_err:.3e} — refusing to plot")

    occupied = np.unique(
        np.concatenate([np.asarray(series[n], dtype=np.float64) for n in names])
    )
    bins = {
        n: nearest_bin_indices(
            np.asarray(series[n], dtype=np.float32)[None, None, :],
            np.asarray(levels_1d, dtype=np.float32)[None, None, :],
        )[0, 0]
        for n in names
    }

    fig, (ax_y, ax_b) = plt.subplots(
        2, 1, figsize=(max(9.0, 0.55 * length + 3.5), 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.4]}, sharex=True,
    )
    ax_y.set_facecolor("white")
    # Occupied rungs only — exact membership of the plotted points.
    for y in occupied:
        ax_y.axhline(float(y), color="0.55", lw=0.9, alpha=0.85, zorder=0)
    for n in names:
        y = np.asarray(series[n], dtype=np.float64)
        # Faint polyline (no steps) so eye can track series; markers carry the snap proof.
        ax_y.plot(x, y, color=colors[n], lw=1.0, alpha=0.35, zorder=1)
        ax_y.plot(
            x, y, linestyle="none", marker="o", markersize=7.5,
            markerfacecolor=colors[n], markeredgecolor="white", markeredgewidth=0.6,
            label=f"{n} (max|Δ|={residuals[n]:.1e})", zorder=3,
        )
    ax_y.set_ylabel("dataset-z (snapped)")
    ax_y.set_title(
        f"{title}\noccupied rungs only ({occupied.size}/{n_rows}); "
        f"all markers on ladder (max residual {max_err:.1e})",
        fontsize=10,
    )
    ax_y.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_y.grid(alpha=0.15)

    for n in names:
        ax_b.plot(
            x, bins[n], color=colors[n], lw=1.0, alpha=0.35, zorder=1,
        )
        ax_b.plot(
            x, bins[n], linestyle="none", marker="s", markersize=6.5,
            markerfacecolor=colors[n], markeredgecolor="white", markeredgewidth=0.5,
            label=n, zorder=3,
        )
    ax_b.set_ylabel(f"{n_rows}-row bin index")
    ax_b.set_xlabel("horizon step t")
    ax_b.set_title("integer ladder row (discrete; same alphabet for GT / binary / MMPD)", fontsize=9)
    ax_b.legend(loc="best", fontsize=8, framealpha=0.9, ncol=3)
    ax_b.grid(alpha=0.15)
    ax_b.set_yticks(sorted({int(v) for b in bins.values() for v in b.tolist()}))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"max_snap_residual": max_err, "n_occupied_rungs": float(occupied.size), **{
        f"residual_{n}": residuals[n] for n in names
    }}


def _write_zoom_viz(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    n_windows: int,
    variate: int,
    slice_lengths: Sequence[int],
    zoom_steps: int,
    seed: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = snapped["gt"]
    binary = snapped["binary"]
    mmpd = snapped["mmpd"]
    levels = snapped["legal_levels"]
    indices = snapped["indices"]
    n = int(gt.shape[0])
    rng = np.random.default_rng(int(seed) + 17)
    picks = np.sort(rng.choice(n, size=min(int(n_windows), n), replace=False))
    colors = {"GT": "black", "binary": "#1f77b4", "MMPD": "#d62728"}
    paths: List[Path] = []
    for local in picks.tolist():
        pool_i = int(indices[local])
        levels_v = levels[local, variate]
        for L in slice_lengths:
            L = int(L)
            if L > int(gt.shape[-1]):
                continue
            offset = max(0, (int(gt.shape[-1]) - L) // 2)
            # Disc sees bin-center-shifted L-slice; show that exact input.
            series_raw = {
                "GT": gt[local, variate, offset : offset + L],
                "binary": binary[local, variate, offset : offset + L],
                "MMPD": mmpd[local, variate, offset : offset + L],
            }
            series_disc: Dict[str, np.ndarray] = {}
            for name, seg in series_raw.items():
                shifted, _ = bin_center_shift(
                    seg[None, None, :],
                    levels[local : local + 1, variate : variate + 1, :],
                    reduce="per_variate",
                )
                series_disc[name] = shifted[0, 0]
            # Zoom crop inside the L-slice for readability.
            z_steps = min(int(zoom_steps), L)
            z0 = max(0, (L - z_steps) // 2)
            z1 = z0 + z_steps
            path = out_dir / (
                f"{run_name}_{dataset}_v{variate}_local{local}_pool{pool_i}_"
                f"L{L}_off{offset}_snapproof.png"
            )
            _plot_snap_proof_panel(
                out_path=path,
                title=(
                    f"{run_name}/{dataset} pool={pool_i} local={local} v={variate} | "
                    f"disc L={L} off={offset} t=[{z0},{z1}) AFTER bin_center_shift "
                f"(dataset-z; NO instance norm; canvas{snapped.get('canvas_height', '?')} "
                f"snap={snapped.get('snap_mode', '?')})"
                ),
                levels_1d=levels_v,
                series={k: v[z0:z1] for k, v in series_disc.items()},
                colors=colors,
                t0=offset + z0,
            )
            paths.append(path)

        # Early-horizon snap proof (pre bin-center; post lattice snap).
        z1 = min(16, int(gt.shape[-1]))
        path = out_dir / (
            f"{run_name}_{dataset}_v{variate}_local{local}_pool{pool_i}_t0-{z1}_snapproof.png"
        )
        _plot_snap_proof_panel(
            out_path=path,
            title=(
                f"{run_name}/{dataset} pool={pool_i} local={local} v={variate} | "
                f"post-snap (pre bin_center) t=0..{z1 - 1} "
                f"(dataset-z; NO instance norm; canvas{snapped.get('canvas_height', '?')} "
                f"snap={snapped.get('snap_mode', '?')})"
            ),
            levels_1d=levels_v,
            series={
                "GT": gt[local, variate, :z1],
                "binary": binary[local, variate, :z1],
                "MMPD": mmpd[local, variate, :z1],
            },
            colors=colors,
            t0=0,
        )
        paths.append(path)
    return paths


def _score_index(scores: Mapping[str, np.ndarray]) -> Dict[Tuple[int, int, int, int], Dict[str, float]]:
    """Map (window, offset, variate, label) → {prob_fake, correct}."""
    out: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}
    n = int(scores["label"].shape[0])
    for i in range(n):
        label = int(scores["label"][i])
        prob = float(scores["prob_fake"][i])
        pred = 1 if prob >= 0.5 else 0
        key = (
            int(scores["window"][i]),
            int(scores["offset"][i]),
            int(scores["variate"][i]),
            label,
        )
        out[key] = {"prob_fake": prob, "pred": float(pred), "correct": float(pred == label)}
    return out


def _load_scores(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _plot_disagreement_panel(
    *,
    out_path: Path,
    title: str,
    past_1d: Optional[np.ndarray],
    gt_1d: np.ndarray,
    binary_1d: np.ndarray,
    mmpd_1d: np.ndarray,
    binary_prob: float,
    mmpd_prob: float,
    label: int,
    offset: int,
) -> None:
    """GT / binary / MMPD L-slice (±lookback) with disc P(fake) annotations."""
    L = int(gt_1d.shape[0])
    t_h = np.arange(offset, offset + L)
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    if past_1d is not None and past_1d.size:
        t_past = np.arange(offset - int(past_1d.shape[0]), offset)
        ax.plot(t_past, past_1d, color="#555555", lw=1.2, label="lookback", alpha=0.85)
        ax.axvline(offset, color="black", ls="--", lw=0.8, alpha=0.45)
    ax.plot(t_h, gt_1d, color="black", lw=2.0, label="GT")
    ax.plot(
        t_h, binary_1d, color="#1f77b4", lw=1.8, alpha=0.9,
        label=f"binary (Pfake={binary_prob:.2f})",
    )
    ax.plot(
        t_h, mmpd_1d, color="#d62728", lw=1.8, alpha=0.9,
        label=f"MMPD (Pfake={mmpd_prob:.2f})",
    )
    # Highlight the L-slice the disc scored.
    ax.axvspan(offset, offset + L - 1, color="#ffe08a", alpha=0.25, zorder=0)
    shown = "FAKE" if label == 1 else "REAL"
    ax.set_title(f"{title}\nshown_to_disc={shown} (label={label})", fontsize=10)
    ax.set_xlabel("horizon step t")
    ax.set_ylabel("dataset-z (snapped)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_disc_disagreement_viz(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    slice_len: int,
    snapped: Mapping[str, np.ndarray],
    binary_scores: Mapping[str, np.ndarray],
    mmpd_scores: Mapping[str, np.ndarray],
    include_past: bool,
    max_panels: int,
    seed: int,
) -> Dict[str, Any]:
    """Panels where one source's disc is correct and the other's is wrong.

    Keys align on (window, offset, variate, label). For label=0 both discs see
    GT; for label=1 each sees its own fake. Cap panels per direction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_ix = _score_index(binary_scores)
    mmpd_ix = _score_index(mmpd_scores)
    shared = sorted(set(bin_ix) & set(mmpd_ix))
    dirs = {
        "mmpd_wrong_binary_right": [],
        "binary_wrong_mmpd_right": [],
    }
    for key in shared:
        b = bin_ix[key]
        m = mmpd_ix[key]
        if b["correct"] >= 0.5 and m["correct"] < 0.5:
            dirs["mmpd_wrong_binary_right"].append(key)
        elif b["correct"] < 0.5 and m["correct"] >= 0.5:
            dirs["binary_wrong_mmpd_right"].append(key)

    rng = np.random.default_rng(int(seed) + int(slice_len) * 17)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    past = np.asarray(snapped["past"])
    L = int(slice_len)
    lookback_tail = min(32, int(past.shape[-1])) if include_past else 0
    paths: Dict[str, List[str]] = {}
    counts = {k: len(v) for k, v in dirs.items()}

    for direction, keys in dirs.items():
        # Prefer confident mistakes: large |Pfake - 0.5| on the wrong disc.
        def _wrong_margin(k: Tuple[int, int, int, int]) -> float:
            if direction.startswith("mmpd_wrong"):
                return abs(float(mmpd_ix[k]["prob_fake"]) - 0.5)
            return abs(float(bin_ix[k]["prob_fake"]) - 0.5)

        keys_sorted = sorted(keys, key=_wrong_margin, reverse=True)
        n = min(int(max_panels), len(keys_sorted))
        if n < len(keys_sorted):
            top = keys_sorted[: max(1, n // 2)]
            rest = keys_sorted[max(1, n // 2) :]
            extra = n - len(top)
            if extra > 0 and rest:
                pick = rng.choice(len(rest), size=min(extra, len(rest)), replace=False)
                top.extend([rest[int(i)] for i in np.atleast_1d(pick)])
            chosen = top[:n]
        else:
            chosen = keys_sorted

        dir_paths: List[str] = []
        for i, (window, offset, variate, label) in enumerate(chosen):
            past_1d = None
            if lookback_tail > 0:
                past_1d = past[window, variate, -lookback_tail:]
            path = out_dir / (
                f"{run_name}_{dataset}_L{L}_{direction}_"
                f"w{window}_off{offset}_v{variate}_lab{label}_{i:02d}.png"
            )
            _plot_disagreement_panel(
                out_path=path,
                title=(
                    f"{run_name}/{dataset} L={L} {direction} | "
                    f"w={window} off={offset} v={variate}"
                ),
                past_1d=past_1d,
                gt_1d=gt[window, variate, offset : offset + L],
                binary_1d=binary[window, variate, offset : offset + L],
                mmpd_1d=mmpd[window, variate, offset : offset + L],
                binary_prob=float(bin_ix[(window, offset, variate, label)]["prob_fake"]),
                mmpd_prob=float(mmpd_ix[(window, offset, variate, label)]["prob_fake"]),
                label=int(label),
                offset=int(offset),
            )
            dir_paths.append(str(path))
        paths[direction] = dir_paths
        print(
            f"[disc-disagree] {run_name}/{dataset}/L{L} {direction}: "
            f"pool={counts[direction]} wrote={len(dir_paths)} → {out_dir}",
            flush=True,
        )

    manifest = {
        "run": run_name,
        "dataset": dataset,
        "slice_len": L,
        "n_shared_keys": len(shared),
        "counts": counts,
        "paths": paths,
    }
    write_json(out_dir / f"manifest_L{L}.json", manifest)
    return manifest


def run_one(
    args: argparse.Namespace,
    *,
    run_name: str,
    ckpt_root: Path,
    config_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Pack pool → binary + MMPD fakes on same rows → snap → purged split → disc.

    ``pack_splits=test`` (default): drive indices from the precomputed MMPD pack
    (MMPD-aligned). ``pack_splits=val,test``: build the combined TSF pool at
    ``pack_test_stride`` density, rematerialize MMPD into output_dir/raw, and
    regenerate binary on those rows. Mixing precomputed test-only MMPD indices
    into a val+test ConcatDataset is refused (wrong offset — fail fast).
    """
    dataset = str(args.dataset)
    print(f"\n=== {run_name} ({ckpt_root.name}) ===", flush=True)
    pack_splits = parse_pack_splits(args.pack_splits)

    # Need run metadata (variates / strides) before building an expanded pool.
    run, _stages, _kind_peek = load_ablation_run(dataset, ckpt_root)
    train_stride, test_stride = _pack_strides(args, run, pack_splits)

    if list(pack_splits) == ["test"]:
        mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
        indices = [int(x) for x in np.asarray(mmpd_full["indices"], dtype=np.int64).tolist()]
        indices, mmpd_pack = _thin_windows(
            indices, mmpd_full, max_windows=args.max_windows, seed=args.seed,
        )
        print(
            f"[{run_name}] windows={len(indices)} (MMPD-aligned pack_splits=test)",
            flush=True,
        )
    else:
        # Expanded pack: indices are into ConcatDataset(pack_splits), not the
        # precomputed test-only MMPD pack. Rematerialize MMPD for this pool.
        pool, _starts, _splits, part_lengths, _ = load_tsf_pack_pool(
            dataset,
            run_variate_indices(run),
            lookback=args.lookback,
            horizon=args.horizon,
            train_stride=train_stride,
            test_stride=test_stride,
            pack_splits=pack_splits,
            use_ordinal_window_norm=False,
        )
        indices = list(range(len(pool)))
        if args.max_windows is not None and int(args.max_windows) < len(indices):
            rng = np.random.default_rng(int(args.seed))
            pick = np.sort(rng.choice(len(indices), size=int(args.max_windows), replace=False))
            indices = [indices[int(i)] for i in pick.tolist()]
        print(
            f"[{run_name}] windows={len(indices)}/{len(pool)} "
            f"pack_splits={pack_splits} parts={part_lengths} "
            f"stride_train={train_stride} stride_test={test_stride} "
            f"(rematerialize MMPD; refuse test-only index reuse)",
            flush=True,
        )
        mmpd_pack = _rematerialize_mmpd_pack(
            args, run=run, indices=indices, pack_splits=pack_splits,
        )
        # Keep only windows MMPD accepted (filter_valid may drop a few).
        mmpd_idx = [int(x) for x in np.asarray(mmpd_pack["indices"], dtype=np.int64).tolist()]
        if mmpd_idx != indices:
            print(
                f"[{run_name}] MMPD kept {len(mmpd_idx)}/{len(indices)} indices; aligning",
                flush=True,
            )
            indices = mmpd_idx

    # Generate (or cache-load) binary_staged pack on those exact pool indices.
    binary_pack, run, ladder, kind, canvas_height = materialize_binary_pack(
        args,
        dataset=dataset,
        run_name=run_name,
        ckpt_root=ckpt_root,
        config_path=config_path,
        indices=indices,
        device=device,
    )
    print(f"[{run_name}] stage_kind={kind} canvas_height={canvas_height}", flush=True)
    snapped = _snap_bundle(
        binary_pack=binary_pack,
        mmpd_pack=mmpd_pack,
        run=run,
        ladder=ladder,
        args=args,
        device=device,
        canvas_height=canvas_height,
        ckpt_root=ckpt_root,
        config_path=config_path,
    )
    write_json(
        args.output_dir / "partials" / f"lattice_{run_name}_{dataset}.json",
        {
            "kind": kind,
            "canvas_height": canvas_height,
            "snap_mode": snapped.get("snap_mode"),
            "snap_meta": snapped["lattice"].get("snap_meta"),
            "raw_binary_max_error": snapped["lattice"]["raw_binary_max_error"],
            "support_atol": snapped["lattice"]["support_atol"],
            "gt": snapped["lattice"]["gt"],
            "binary": snapped["lattice"]["binary"],
            "mmpd": snapped["lattice"]["mmpd"],
            "gt_snap": snapped["lattice"]["gt_snap"],
            "binary_snap": snapped["lattice"]["binary_snap"],
            "mmpd_snap": snapped["lattice"]["mmpd_snap"],
        },
    )

    viz_dir = args.output_dir / "viz" / run_name
    viz_paths = _write_zoom_viz(
        out_dir=viz_dir,
        run_name=run_name,
        dataset=dataset,
        snapped=snapped,
        n_windows=int(args.viz_n_windows),
        variate=int(args.viz_variate),
        slice_lengths=args.slice_lengths,
        zoom_steps=int(args.viz_zoom_steps),
        seed=int(args.seed),
    )
    print(f"[{run_name}] wrote {len(viz_paths)} viz panels under {viz_dir}", flush=True)

    redbox_paths: List[str] = []
    if bool(getattr(args, "redbox_viz", True)):
        redbox_paths = write_redbox_forecast_viz(
            args=args,
            run_name=run_name,
            ckpt_root=ckpt_root,
            config_path=config_path,
            device=device,
        )

    if args.viz_only:
        return {
            "kind": kind,
            "viz": [str(p) for p in viz_paths],
            "redbox_viz": redbox_paths,
            "metrics": {},
        }

    bundle = SimpleNamespace(
        fakes={"binary_staged": snapped["binary"], "mmpd": snapped["mmpd"]},
        # Same snapped GT for both fake sources (real-vs-fake, not binary-vs-mmpd).
        y_true_by_source={
            "binary_staged": snapped["gt"],
            "mmpd": snapped["gt"].copy(),
        },
        past=snapped["past"],
        legal_levels=snapped["legal_levels"],
        indices=snapped["indices"],
        series_starts=snapped["series_starts"],
        run=run,
        pack_splits=[str(x) for x in binary_pack["pack_splits"].tolist()],
    )
    # Chronological carve of the pack + hard purge vs disc test
    # (default 70/15/15; or 80/20 + val-from-train when val_fraction<=0).
    # With pack_splits=test this is *not* paper train — second carve of the pack.
    splits = split_windows(
        len(snapped["gt"]),
        args,
        dataset,
        indices=bundle.indices,
        lookback=args.lookback,
        horizon=args.horizon,
        test_stride=int(args.pack_test_stride),
        series_starts=bundle.series_starts,
    )
    metrics: Dict[str, Any] = {"kind": kind}
    # Univariate train_classifier (unique_abs / bin-center) — not disc_shared's
    # multivariate HorizonSliceDataset path.
    args.save_classification_scores = True
    args.return_test_scores = True
    scores_by_source: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for source in ("binary_staged", "mmpd"):
        per_len: Dict[str, Any] = {}
        scores_by_source[source] = {}
        for length in args.slice_lengths:
            if int(length) <= args.horizon:
                raw = train_classifier(
                    args, dataset, source, int(length), bundle, splits, device,
                )
                scores = raw.pop("_test_scores", None)
                if scores is None and raw.get("score_path"):
                    scores = _load_scores(Path(str(raw["score_path"])))
                if scores is not None:
                    scores_by_source[source][str(int(length))] = scores
                per_len[str(int(length))] = raw
        write_json(args.output_dir / "partials" / f"{run_name}__{dataset}__{source}.json", per_len)
        metrics[source] = per_len

    # Always write disagreement panels (MMPD wrong / binary right and vice versa).
    disagree_root = args.output_dir / "viz" / "disc_disagreement" / run_name
    disagree_manifests: Dict[str, Any] = {}
    for length in args.slice_lengths:
        key = str(int(length))
        if key not in scores_by_source.get("binary_staged", {}):
            continue
        if key not in scores_by_source.get("mmpd", {}):
            continue
        disagree_manifests[key] = write_disc_disagreement_viz(
            out_dir=disagree_root,
            run_name=run_name,
            dataset=dataset,
            slice_len=int(length),
            snapped=snapped,
            binary_scores=scores_by_source["binary_staged"][key],
            mmpd_scores=scores_by_source["mmpd"][key],
            include_past=not bool(args.candidate_only),
            max_panels=int(getattr(args, "disc_disagreement_max", 12)),
            seed=int(args.seed),
        )
    write_json(disagree_root / "summary.json", disagree_manifests)

    # Flat per-variate dump for the whole run.
    by_var_rows: List[Dict[str, Any]] = []
    for source, per_len in metrics.items():
        if source == "kind" or not isinstance(per_len, dict):
            continue
        for L, mets in per_len.items():
            if not isinstance(mets, dict):
                continue
            acc_bv = mets.get("acc_by_variate") or {}
            auc_bv = mets.get("auroc_by_variate") or {}
            n_bv = mets.get("n_by_variate") or {}
            for v in sorted(set(acc_bv) | set(auc_bv), key=lambda x: int(x)):
                by_var_rows.append(
                    {
                        "run": run_name,
                        "source": source,
                        "L": int(L),
                        "variate": int(v),
                        "disc_acc": float(acc_bv.get(v, float("nan"))),
                        "disc_auroc": float(auc_bv.get(v, float("nan"))),
                        "n_examples": float(n_bv.get(v, float("nan"))),
                    }
                )
    write_json(args.output_dir / "auroc_by_variate.json", by_var_rows)

    return {
        "kind": kind,
        "viz": [str(p) for p in viz_paths],
        "redbox_viz": redbox_paths,
        "disc_disagreement": disagree_manifests,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    apply_smoke(args)
    pack_splits = apply_disc_pack_protocol(args)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"device={device} viz_only={args.viz_only} smoke={args.smoke_test} "
        f"redbox_viz={args.redbox_viz} pack_splits={pack_splits} "
        f"train_frac={args.train_fraction} val_frac={args.val_fraction}",
        flush=True,
    )

    summary: Dict[str, Any] = {}
    for spec in _parse_run_specs(args.runs):
        ckpt = Path(spec["ckpt"])
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        summary[spec["name"]] = run_one(
            args,
            run_name=spec["name"],
            ckpt_root=ckpt,
            config_path=spec["config"],
            device=device,
        )
    write_json(args.output_dir / "summary.json", summary)
    # Flat table: run × source × L → disc_auroc / disc_acc (chance ≈ 0.5).
    rows = []
    by_var_all: List[Dict[str, Any]] = []
    for name, payload in summary.items():
        for source, per_len in (payload.get("metrics") or {}).items():
            if not isinstance(per_len, dict):
                continue
            for L, mets in per_len.items():
                if isinstance(mets, dict) and "disc_auroc" in mets:
                    rows.append(
                        {
                            "run": name,
                            "kind": payload.get("kind"),
                            "source": source,
                            "L": int(L),
                            "disc_auroc": float(mets["disc_auroc"]),
                            "disc_acc": float(mets.get("disc_acc", float("nan"))),
                            "acc_by_variate": mets.get("acc_by_variate") or {},
                            "auroc_by_variate": mets.get("auroc_by_variate") or {},
                        }
                    )
                    acc_bv = mets.get("acc_by_variate") or {}
                    auc_bv = mets.get("auroc_by_variate") or {}
                    n_bv = mets.get("n_by_variate") or {}
                    for v in sorted(set(acc_bv) | set(auc_bv), key=lambda x: int(x)):
                        by_var_all.append(
                            {
                                "run": name,
                                "source": source,
                                "L": int(L),
                                "variate": int(v),
                                "disc_acc": float(acc_bv.get(v, float("nan"))),
                                "disc_auroc": float(auc_bv.get(v, float("nan"))),
                                "n_examples": float(n_bv.get(v, float("nan"))),
                            }
                        )
    if rows:
        write_json(args.output_dir / "auroc_table.json", rows)
        print("\nAUROC table:", flush=True)
        for row in rows:
            print(
                f"  {row['run']:16s} {row['source']:14s} L{row['L']:<3d} "
                f"auroc={row['disc_auroc']:.4f} acc={row['disc_acc']:.4f}",
                flush=True,
            )
            auc_bv = row.get("auroc_by_variate") or {}
            acc_bv = row.get("acc_by_variate") or {}
            if auc_bv or acc_bv:
                parts = [
                    f"v{v}:acc={float(acc_bv.get(v, float('nan'))):.3f}/"
                    f"auc={float(auc_bv.get(v, float('nan'))):.3f}"
                    for v in sorted(set(auc_bv) | set(acc_bv), key=lambda x: int(x))
                ]
                print(f"    by_variate: {' | '.join(parts)}", flush=True)
    if by_var_all:
        write_json(args.output_dir / "auroc_by_variate.json", by_var_all)
    cache_dir = _shared_cache_dir(args)
    print(
        f"\nforecast_cache_dir={cache_dir} require={bool(args.require_forecast_cache)}",
        flush=True,
    )
    print(f"\ndone → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
