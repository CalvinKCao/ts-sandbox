"""Point-acc staged eval: top-gap binary↔MMPD panels + redbox on those windows.

Called from ``StagedEvalPhase`` after metrics. Ranking uses the **anchor** path
only (binary ``final_anchor`` vs MMPD ``deterministic``), ``abs_diff`` of
per-window MSE. Redbox reuses the same top-k window indices with
``sampler=anchor`` and guidance overlay.

MMPD campaign path is explicit (``visualization.mmpd_campaign_root``). The
legacy ``experiment.mmpd_root`` (``datasets/mmpd``) is source data — not used
here. Unset campaign → skip with a clear log. Set but missing pack → fail fast.

Stride: gap ranking uses ``viz_binary_mmpd_eval_test_stride`` (default 4) to
match MMPD matched-binary packs. If the in-memory pack's ``test_stride``
differs, we re-eval via ``run_or_load_dataset_eval`` at that lattice (cached).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MMPD_CONFIG = (
    "configs/mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary.yaml"
)
DEFAULT_MMPD_SUFFIX = "mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary"


def resolve_mmpd_campaign_root(viz_cfg: Dict[str, Any], dataset: str) -> Optional[Path]:
    """Return campaign dir or None when unset.

    Accepts str path, or mapping ``{dataset: path}`` under
    ``mmpd_campaign_root`` / ``mmpd_campaign_root_by_dataset``.
    """
    by_ds = viz_cfg.get("mmpd_campaign_root_by_dataset")
    if isinstance(by_ds, dict) and dataset in by_ds and by_ds[dataset]:
        return Path(str(by_ds[dataset])).expanduser()
    raw = viz_cfg.get("mmpd_campaign_root")
    if raw is None or raw is False or (isinstance(raw, str) and not str(raw).strip()):
        return None
    if isinstance(raw, dict):
        if dataset not in raw or not raw[dataset]:
            return None
        return Path(str(raw[dataset])).expanduser()
    return Path(str(raw)).expanduser()


def _mmpd_pack_path(campaign: Path, dataset: str) -> Path:
    return Path(campaign) / "raw" / f"mmpd_{dataset}.npz"


def _build_cache_from_packs(
    *,
    binary_pack: Dict[str, np.ndarray],
    mmpd_aligned: Dict[str, np.ndarray],
    test_stride: int,
) -> Dict[str, np.ndarray]:
    from models.diffusion_tsf.pipeline.visualize_utils import per_window_anchor_mse

    y_true = np.asarray(binary_pack["y_true"])
    bin_anchor = np.asarray(binary_pack["final_anchor"])
    mmpd_det = np.asarray(mmpd_aligned["deterministic"])
    if y_true.shape != bin_anchor.shape or y_true.shape != mmpd_det.shape:
        raise ValueError(
            f"pack shape mismatch y_true={y_true.shape} "
            f"binary={bin_anchor.shape} mmpd={mmpd_det.shape}"
        )
    y_mmpd = np.asarray(mmpd_aligned["y_true"])
    if not np.allclose(y_true, y_mmpd, rtol=0.0, atol=1e-4):
        raise ValueError(
            "binary vs MMPD y_true mismatch after align — check eval_test_stride / lattice"
        )
    window_indices = np.asarray(binary_pack["window_indices"], dtype=np.int64)
    series_starts = np.asarray(
        binary_pack.get("series_starts", window_indices * int(test_stride)),
        dtype=np.int64,
    )
    binary_mse = per_window_anchor_mse(y_true, bin_anchor)
    mmpd_mse = per_window_anchor_mse(y_true, mmpd_det)
    return {
        "window_indices": window_indices,
        "series_starts": series_starts,
        "binary_anchor_mse": np.asarray(binary_mse, dtype=np.float64),
        "mmpd_anchor_mse": np.asarray(mmpd_mse, dtype=np.float64),
        "error_diff": np.asarray(mmpd_mse - binary_mse, dtype=np.float64),
        "test_stride": np.asarray([int(test_stride)], dtype=np.int64),
    }


def _load_or_build_gap_cache(
    *,
    dataset: str,
    campaign: Path,
    binary_ckpt: Path,
    binary_config: str,
    pack: Optional[Dict[str, np.ndarray]],
    pack_test_stride: Optional[int],
    eval_test_stride: int,
    work_dir: Path,
    device: torch.device,
    smoke_test: bool,
    force_eval: bool,
    mmpd_config: str,
    mmpd_config_suffix: str,
) -> Dict[str, np.ndarray]:
    from utils.compare_binary_mmpd_staged_diag import (
        align_mmpd_pack,
        build_mmpd_args,
        run_or_load_dataset_eval,
    )

    pack_path = _mmpd_pack_path(campaign, dataset)
    if not pack_path.is_file():
        raise FileNotFoundError(
            f"mmpd_campaign_root set to {campaign} but missing pack {pack_path}"
        )

    # Fast path: in-memory staged pack already on the MMPD lattice.
    if (
        pack is not None
        and pack_test_stride is not None
        and int(pack_test_stride) == int(eval_test_stride)
        and "final_anchor" in pack
        and "y_true" in pack
        and "window_indices" in pack
        and not force_eval
    ):
        with np.load(pack_path) as z:
            mmpd_raw = {k: z[k] for k in z.files}
        aligned = align_mmpd_pack(mmpd_raw, pack["window_indices"])
        logger.info(
            "[%s] point-gap cache from in-memory pack (stride=%d, n=%d)",
            dataset,
            eval_test_stride,
            len(pack["window_indices"]),
        )
        return _build_cache_from_packs(
            binary_pack=pack,
            mmpd_aligned=aligned,
            test_stride=int(eval_test_stride),
        )

    # Slow path: re-eval / load compare cache at the pack lattice (default stride 4).
    logger.info(
        "[%s] point-gap: pack stride=%s != eval_test_stride=%d — "
        "using compare util cache under %s",
        dataset,
        pack_test_stride,
        eval_test_stride,
        work_dir,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    mmpd_args = build_mmpd_args(
        mmpd_dir=campaign.resolve(),
        mmpd_config=(REPO_ROOT / mmpd_config).resolve(),
        repo=REPO_ROOT,
        force_mmpd_eval=False,
        smoke_test=bool(smoke_test),
    )
    mmpd_args.mmpd_config_suffix = str(mmpd_config_suffix)
    mmpd_args.mmpd_output_root = campaign.resolve()
    mmpd_args.eval_test_stride = int(eval_test_stride)
    if smoke_test:
        mmpd_args.test_max_items = min(int(getattr(mmpd_args, "test_max_items", 8) or 8), 8)
    return run_or_load_dataset_eval(
        dataset=dataset,
        mmpd_args=mmpd_args,
        binary_ckpt=binary_ckpt,
        binary_config=binary_config,
        output_dir=work_dir,
        device=device,
        force_eval=bool(force_eval or smoke_test),
        test_fraction=1.0,
        datasets_root=(REPO_ROOT / "results" / "datasets").resolve(),
        auto_mmpd_ckpt=False,
    )


def run_binary_mmpd_gap_and_redbox_viz(
    *,
    state: Any,
    pack: Optional[Dict[str, np.ndarray]],
    coarse_model: Any,
    fine_model: Any,
    device: torch.device,
    viz_cfg: Dict[str, Any],
    patch_refine: bool,
    joint_dual: bool,
    pack_test_stride: Optional[int] = None,
    binary_config_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Write top-gap + redbox under ``state.results_dir/viz/point_gap*``.

    Returns dict with keys ``gap`` / ``redbox`` listing written paths.
    """
    do_gap = bool(viz_cfg.get("viz_binary_mmpd_gap", True))
    do_redbox = bool(viz_cfg.get("viz_binary_mmpd_redbox", True))
    if not do_gap and not do_redbox:
        return {"gap": [], "redbox": []}
    if joint_dual:
        logger.info(
            "[%s] point-gap/redbox skipped: joint_dual (vertical/channel)",
            state.dataset,
        )
        return {"gap": [], "redbox": []}

    dataset = str(state.dataset)
    campaign = resolve_mmpd_campaign_root(viz_cfg, dataset)
    if campaign is None:
        logger.info(
            "[%s] point-gap/redbox skipped: visualization.mmpd_campaign_root unset",
            dataset,
        )
        return {"gap": [], "redbox": []}
    if not campaign.is_absolute():
        campaign = (REPO_ROOT / campaign).resolve()
    pack_path = _mmpd_pack_path(campaign, dataset)
    if not pack_path.is_file():
        raise FileNotFoundError(
            f"[{dataset}] mmpd_campaign_root={campaign} missing {pack_path}"
        )

    top_k = int(viz_cfg.get("viz_binary_mmpd_top_k", 10) or 10)
    min_spacing = int(viz_cfg.get("viz_binary_mmpd_min_spacing", 48) or 48)
    diff_mode = str(viz_cfg.get("viz_binary_mmpd_diff_mode", "abs_diff") or "abs_diff")
    eval_stride = int(viz_cfg.get("viz_binary_mmpd_eval_test_stride", 4) or 4)
    vars_gap = int(viz_cfg.get("viz_binary_mmpd_variables_to_plot", 99) or 99)
    vars_rb = int(viz_cfg.get("viz_binary_mmpd_redbox_variables_to_plot", 0) or 0)
    jpeg_dpi = int(viz_cfg.get("jpeg_dpi", 100) or 100)
    if bool(getattr(state, "smoke_test", False)):
        top_k = min(top_k, int(viz_cfg.get("viz_binary_mmpd_smoke_top_k", 1) or 1))
        vars_gap = min(vars_gap, int(viz_cfg.get("viz_binary_mmpd_smoke_variables", 2) or 2))
        if vars_rb <= 0:
            vars_rb = int(viz_cfg.get("viz_binary_mmpd_smoke_variables", 2) or 2)

    ckpt = Path(str(state.checkpoint_dir)).resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"[{dataset}] checkpoint_dir missing: {ckpt}")
    cfg_path = binary_config_path or str(
        getattr(state, "config_path", None)
        or state.extra.get("config_path")
        or ""
    )
    if not cfg_path:
        # Fall back to experiment_name leaf under configs/ if present.
        stem = str(getattr(state, "experiment_name", "") or "")
        guess = REPO_ROOT / "configs" / f"{stem}.yaml"
        if guess.is_file():
            cfg_path = str(guess)
        else:
            raise RuntimeError(
                f"[{dataset}] cannot resolve binary config path for point-gap viz"
            )

    work_dir = Path(state.results_dir) / "viz" / "point_gap_work" / dataset
    cache = _load_or_build_gap_cache(
        dataset=dataset,
        campaign=campaign,
        binary_ckpt=ckpt,
        binary_config=cfg_path,
        pack=pack,
        pack_test_stride=pack_test_stride,
        eval_test_stride=eval_stride,
        work_dir=work_dir,
        device=device,
        smoke_test=bool(getattr(state, "smoke_test", False)),
        force_eval=bool(state.extra.get("force_point_gap_eval", False)),
        mmpd_config=str(viz_cfg.get("mmpd_config", DEFAULT_MMPD_CONFIG)),
        mmpd_config_suffix=str(
            viz_cfg.get("mmpd_config_suffix", DEFAULT_MMPD_SUFFIX)
        ),
    )

    from utils.compare_binary_mmpd_staged_diag import (
        plot_dataset_windows,
        select_top_windows,
    )

    top_manifest = select_top_windows(
        cache,
        top_k=top_k,
        random_k=0,
        min_spacing=min_spacing,
        diff_mode=diff_mode,
        seed=int(getattr(state, "seed", 2026)),
    )
    gap_dir = Path(state.results_dir) / "viz" / "point_gap" / dataset
    gap_dir.mkdir(parents=True, exist_ok=True)
    (gap_dir / "top_windows.json").write_text(
        json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8"
    )

    written: Dict[str, List[str]] = {"gap": [], "redbox": []}
    if do_gap:
        gap_paths = plot_dataset_windows(
            dataset=dataset,
            binary_ckpt=ckpt,
            binary_config=cfg_path,
            mmpd_pack_path=pack_path,
            cache=cache,
            top_manifest=top_manifest,
            output_dir=gap_dir,
            test_stride=eval_stride,
            device=device,
            variables_to_plot=vars_gap,
            jpeg_dpi=jpeg_dpi,
        )
        # plot_dataset_windows writes under output_dir/plots/<dataset>/
        for p in gap_paths:
            written["gap"].append(str(p))
        logger.info("[%s] point-gap wrote %d panels under %s", dataset, len(gap_paths), gap_dir)

    if do_redbox:
        from utils.staged_eval_sample_viz import write_staged_sample_panels

        picks = [
            int(e["window_index"])
            for e in top_manifest
            if str(e.get("pick_kind", "top_diff")) == "top_diff"
        ][:top_k]
        if not picks:
            raise RuntimeError(f"[{dataset}] empty top_diff picks for redbox")

        # Redbox pool must sit on the same lattice as ranking (eval_stride).
        pool = _load_test_pool_for_stride(
            state=state,
            binary_config=cfg_path,
            test_stride=eval_stride,
            device=device,
        )
        redbox_dir = Path(state.results_dir) / "viz" / "point_gap_redbox" / dataset
        kind = "patch_refine" if patch_refine else "fine"
        rb_paths = write_staged_sample_panels(
            out_dir=redbox_dir,
            run_name=str(getattr(state, "subset_id", dataset) or dataset),
            dataset=dataset,
            kind=kind,
            coarse_model=coarse_model,
            fine_model=fine_model,
            pool=pool,
            picks=picks,
            device=device,
            sampler="anchor",
            num_sampling_steps=1,
            seed=int(getattr(state, "seed", 42)),
            variables_to_plot=vars_rb,
            jpeg_dpi=jpeg_dpi,
        )
        written["redbox"] = [str(p) for p in rb_paths]
        (redbox_dir / "top_windows_used.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "window_indices": picks,
                    "sampler": "anchor",
                    "eval_test_stride": eval_stride,
                    "mmpd_campaign_root": str(campaign),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "[%s] point-gap redbox wrote %d panels under %s",
            dataset,
            len(rb_paths),
            redbox_dir,
        )
    return written


def _load_test_pool_for_stride(
    *,
    state: Any,
    binary_config: str,
    test_stride: int,
    device: torch.device,
) -> Sequence[Any]:
    """Rebuild test dataset at ``test_stride`` for redbox picks."""
    del device  # pool is CPU dataset; generate moves batches later
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    cfg = load_experiment_config(
        binary_config, cli_overrides={"dataset": str(state.dataset)}
    )
    # Honor state's subset / lookback / horizon when already resolved.
    lookback = int(getattr(state, "lookback", 336) or 336)
    horizon = int(getattr(state, "horizon", 96) or 96)
    variate_indices = list(state.variate_indices or [])
    if not variate_indices:
        raise RuntimeError(f"{state.dataset}: empty variate_indices for redbox pool")
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    _, _, test_ds, norm_stats = load_dataset(
        str(state.dataset),
        variate_indices,
        stride=int(getattr(state, "window_stride", 1) or 1),
        test_stride=int(test_stride),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(getattr(state, "ordinal_tie_atol", 0.0) or 0.0),
        use_ordinal_window_norm=bool(getattr(state, "use_ordinal_window_norm", False)),
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]
    # Silence unused cfg (kept for load_experiment_config side effects / validation).
    _ = cfg
    return test_ds
