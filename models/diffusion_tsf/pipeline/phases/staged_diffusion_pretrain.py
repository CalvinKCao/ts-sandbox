"""Fixed-HP synthetic pretrain for staged coarse/fine diffusion models."""

from __future__ import annotations

import json
import logging
import os
import hashlib
import time
from typing import Any, Dict, Optional

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.visualize_utils import (
    run_pretrain_diffusion_visualizations,
    run_staged_synthetic_pretrain_diagnostics,
)
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.haar_frequency_calibration import ensure_haar_frequency_calibration
from models.diffusion_tsf.pipeline.fourier_frequency_calibration import ensure_fourier_frequency_calibration

logger = logging.getLogger(__name__)

def _stage_pretrain_dir(state: PipelineState, stage: str) -> str:
    return os.path.join(state.checkpoint_dir, f"pretrained_{stage}")


def _stage_pretrain_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_pretrain_dir(state, stage), "pretrained_diffusion.pt")


def staged_diffusion_stages(state: PipelineState) -> tuple[str, ...]:
    return ("coarse", "fine", "finer") if state.use_triple_scale else ("coarse", "fine")


def _stage_pretrain_cache_enabled(phase: PipelinePhase, state: PipelineState) -> bool:
    if state.extra.get("force_retrain_synthetic", False):
        return False
    if phase.get("reuse_pretrain_from_config"):
        return False
    if state.smoke_test:
        return False
    value = phase.get("shared_cache", state.extra.get("staged_pretrain_shared_cache", True))
    return bool(value)


def _stage_pretrain_signature(state: PipelineState, config_name: str) -> str:
    max_scale = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    payload = {
        "config": config_name,
        "n_variates": int(state.n_variates),
        "model_type": state.model_type,
        "diffusion_type": state.diffusion_type,
        "image_height": int(state.image_height),
        "coarse_image_height": int(state.coarse_image_height),
        "fine_image_height": int(state.fine_image_height),
        "finer_image_height": int(state.finer_image_height),
        "use_triple_scale": bool(state.use_triple_scale),
        "staged_representation": str(state.staged_representation),
        "haar_high_freq_levels": int(state.haar_high_freq_levels),
        "haar_high_freq_percent": float(state.haar_high_freq_percent),
        "haar_fine_max_scale": float(state.haar_fine_max_scale),
        "fourier_high_freq_cutoff_bin": int(state.fourier_high_freq_cutoff_bin),
        "fourier_high_freq_percent": float(state.fourier_high_freq_percent),
        "fourier_fine_max_scale": float(state.fourier_fine_max_scale),
        "fourier_flatline_atol": float(state.fourier_flatline_atol),
        "coarse_flatline_blur_fine_target": bool(state.coarse_flatline_blur_fine_target),
        "coarse_flatline_blur_radius": int(state.coarse_flatline_blur_radius),
        "coarse_flatline_blur_kernel": str(state.coarse_flatline_blur_kernel),
        "max_scale": max_scale,
        "dit_patch_size": list(state.dit_patch_size),
        "dit_embed_dim": int(state.dit_embed_dim),
        "dit_depth": int(state.dit_depth),
        "dit_num_heads": int(state.dit_num_heads),
        "dit_mlp_ratio": float(state.dit_mlp_ratio),
        "dit_dropout": float(state.dit_dropout),
        "use_guidance_channel": bool(state.use_guidance_channel),
        "deterministic_anchor_loss": bool(state.deterministic_anchor_loss),
        "deterministic_anchor_lambda": float(state.deterministic_anchor_lambda),
        "deterministic_anchor_alpha": float(state.deterministic_anchor_alpha),
        "lookback_length": int(state.lookback_length),
        "forecast_length": int(state.forecast_length),
        "representation_time_stride": int(state.representation_time_stride),
        "past_cond_resize_to_horizon": bool(state.past_cond_resize_to_horizon),
        "diffusion_lookback_cap": int(state.diffusion_lookback_cap),
        "use_window_normalization": bool(state.use_window_normalization),
        "window_norm_center": str(state.window_norm_center),
        "window_norm_std_floor": float(state.window_norm_std_floor),
        "window_norm_low_var_threshold": float(state.window_norm_low_var_threshold),
        "window_norm_low_var_unit_std": float(state.window_norm_low_var_unit_std),
        "cross_variate_context_bias": float(state.cross_variate_context_bias),
        "use_raw_lookback_cond_channel": bool(state.use_raw_lookback_cond_channel),
        "use_ordinal_window_norm": bool(state.use_ordinal_window_norm),
        "ordinal_tie_atol": float(state.ordinal_tie_atol),
        "binary_anchor_input_mode": str(state.binary_anchor_input_mode),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return (
        f"{config_name}-v{payload['n_variates']}-h{payload['image_height']}"
        f"-ms{max_scale:g}-{digest}"
    )


def _stage_max_scale_for_dataset(state: PipelineState, dataset: str) -> float:
    return float(state.max_scale_by_dataset.get(dataset, state.max_scale))


def _current_run_config_suffix(state: PipelineState) -> str:
    """Infer the submit_grid config suffix from the isolated checkpoint dir name."""
    name = os.path.basename(os.path.abspath(state.checkpoint_dir))
    token = f"-{state.dataset}-"
    if token in name:
        return name.split(token, 1)[1]
    return state.experiment_name.replace("-", "_")


def _shared_stage_pretrain_dir(state: PipelineState, config_name: str, stage: str) -> str:
    key = _stage_pretrain_signature(state, config_name)
    return os.path.join(_phase1_ckpt_root(state), "_shared_staged_pretrain", key, stage)


def _shared_stage_pretrain_ckpt(state: PipelineState, config_name: str, stage: str) -> str:
    return os.path.join(
        _shared_stage_pretrain_dir(state, config_name, stage),
        "pretrained_diffusion.pt",
    )


def _wait_for_shared_stage_ckpt(
    state: PipelineState,
    config_name: str,
    stage: str,
    *,
    wait_seconds: float,
) -> str:
    shared_ckpt = _shared_stage_pretrain_ckpt(state, config_name, stage)
    lock_dir = f"{shared_ckpt}.lock"
    os.makedirs(os.path.dirname(shared_ckpt), exist_ok=True)
    start = time.time()

    while True:
        if os.path.exists(shared_ckpt):
            logger.info("  [staged_diffusion_pretrain] %s shared cached: %s", stage, shared_ckpt)
            return shared_ckpt
        try:
            os.mkdir(lock_dir)
            logger.info("  [staged_diffusion_pretrain] %s acquired shared lock: %s", stage, lock_dir)
            return ""
        except FileExistsError:
            elapsed = time.time() - start
            if elapsed > wait_seconds:
                raise TimeoutError(
                    f"Timed out waiting {wait_seconds:.0f}s for shared staged {stage} pretrain: "
                    f"{shared_ckpt} (lock: {lock_dir})"
                )
            logger.info(
                "  [staged_diffusion_pretrain] %s waiting for shared pretrain lock: %s",
                stage,
                lock_dir,
            )
            time.sleep(min(30.0, max(1.0, wait_seconds - elapsed)))


def _release_shared_lock(shared_ckpt: str) -> None:
    lock_dir = f"{shared_ckpt}.lock"
    try:
        os.rmdir(lock_dir)
    except FileNotFoundError:
        return
    except OSError as e:
        logger.warning("Failed to release shared pretrain lock %s: %s", lock_dir, e)


def _discover_existing_stage_pretrain(state: PipelineState, stage: str) -> Optional[str]:
    """Find a prior isolated run's staged pretrain that matches this run's geometry.

    Old runs did not write shared metadata, so keep the match conservative:
    same submit config suffix, same derived max_scale, same variate count implied by
    the current config, and the expected staged checkpoint path exists.
    """
    ckpt_root = _phase1_ckpt_root(state)
    config_suffix = _current_run_config_suffix(state)
    current_ms = _stage_max_scale_for_dataset(state, state.dataset)
    current_name = os.path.basename(os.path.abspath(state.checkpoint_dir))
    candidates = []
    try:
        for name in os.listdir(ckpt_root):
            if name == current_name or not name.endswith(f"-{config_suffix}"):
                continue
            prefix = name[: -len(f"-{config_suffix}")]
            source_dataset = prefix.rsplit("-", 1)[-1]
            if _stage_max_scale_for_dataset(state, source_dataset) != current_ms:
                continue
            ckpt = os.path.join(
                ckpt_root,
                name,
                f"pretrained_{stage}",
                "pretrained_diffusion.pt",
            )
            if os.path.exists(ckpt):
                candidates.append((os.path.getmtime(ckpt), ckpt))
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _phase1_ckpt_root(state: PipelineState) -> str:
    """Directory that holds per-run checkpoint folders (*-<dataset>-<config>)."""
    ckpt_dir = os.path.abspath(state.checkpoint_dir)
    if os.path.basename(ckpt_dir) == "ckpts":
        return ckpt_dir
    return os.path.dirname(ckpt_dir)


def _candidate_phase1_ckpt_roots(state: PipelineState) -> list[str]:
    """Likely checkpoint roots, including the old Killarney double-user path fixup."""
    roots = []

    def add(path: Optional[str]) -> None:
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            if path not in roots:
                roots.append(path)

    root = _phase1_ckpt_root(state)
    add(root)

    user = os.environ.get("USER")
    if user:
        doubled = f"{os.sep}{user}{os.sep}{user}{os.sep}"
        if doubled in root:
            add(root.replace(doubled, f"{os.sep}{user}{os.sep}", 1))

    scratch = os.environ.get("SCRATCH")
    if scratch:
        add(os.path.join(scratch, "ts-sandbox", "results", "ckpts"))
        if user and os.path.basename(os.path.abspath(scratch)) != user:
            add(os.path.join(scratch, user, "ts-sandbox", "results", "ckpts"))

    submit_dir = os.environ.get("SLURM_SUBMIT_DIR")
    if submit_dir:
        add(os.path.join(submit_dir, "results", "ckpts"))

    add(os.path.join(os.getcwd(), "results", "ckpts"))
    if state.datasets_dir:
        add(os.path.join(os.path.dirname(os.path.abspath(state.datasets_dir)), "results", "ckpts"))

    return roots


def source_run_stage_pretrain_ckpt(
    state: PipelineState,
    source_config: str,
    stage: str,
) -> Optional[str]:
    """``pretrained_{stage}/pretrained_diffusion.pt`` from a prior grid run.

    Prefer same-dataset donor. If missing (e.g. quota deleted electricity's
    pretrained_* while ETTh1's copy remains), fall back to any dataset's run
    of the same config stem — synthetic pretrain is geometry-keyed, not
    dataset-keyed.
    """
    ckpt_rel = f"pretrained_{stage}/pretrained_diffusion.pt"
    try:
        source_dir = discover_dataset_run_ckpt_dir(
            state, source_config, required_file=ckpt_rel,
        )
    except FileNotFoundError:
        source_dir = discover_any_dataset_run_ckpt_dir(
            state, source_config, required_file=ckpt_rel,
        )
        if source_dir is None:
            return None
        logger.warning(
            "  [staged_diffusion_pretrain] %s: no *-%s-%s pretrain; "
            "falling back to cross-dataset donor %s (synthetic pretrain is "
            "geometry-shared)",
            stage,
            state.dataset,
            source_config,
            source_dir,
        )
    ckpt = os.path.join(source_dir, ckpt_rel)
    if os.path.exists(ckpt):
        return ckpt
    return None


def _run_dir_matches_config(name: str, dataset: str, config_suffix: str) -> bool:
    """Exact stem match: ``*-{dataset}-{config_suffix}`` with nothing after the suffix.

    Do not treat ``{config}_bs_small`` / ``{config}_smoke`` as the same config — that
    silently reuses the newest sibling HP/smoke run instead of the named donor.
    """
    token = f"-{dataset}-{config_suffix}"
    idx = name.find(token)
    if idx < 0:
        return False
    return name[idx + len(token):] == ""


def _run_dir_matches_config_any_dataset(name: str, config_suffix: str) -> bool:
    """Exact stem match: ``*-<any_dataset>-{config_suffix}`` (nothing after suffix)."""
    token = f"-{config_suffix}"
    if not name.endswith(token):
        return False
    # Require a dataset segment: MM-DD-jobid-DATASET-config
    body = name[: -len(token)]
    return body.count("-") >= 3


def discover_dataset_run_ckpt_dir(
    state: PipelineState,
    config_suffix: str,
    *,
    required_file: Optional[str] = None,
) -> str:
    """Newest isolated run dir ``*-<dataset>-<config_suffix>`` under the ckpt root."""
    best_dir: Optional[str] = None
    best_mtime = 0.0
    roots = _candidate_phase1_ckpt_roots(state)
    for ckpt_root in roots:
        try:
            names = os.listdir(ckpt_root)
        except OSError:
            continue
        for name in names:
            if not _run_dir_matches_config(name, state.dataset, config_suffix):
                continue
            path = os.path.join(ckpt_root, name)
            if not os.path.isdir(path):
                continue
            if required_file and not os.path.exists(os.path.join(path, required_file)):
                continue
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = path
    if not best_dir:
        req = f" containing {required_file!r}" if required_file else ""
        raise FileNotFoundError(
            f"No prior run *-{state.dataset}-{config_suffix}{req} under any of {roots}. "
            "Complete the exhaustive staged grid first."
        )
    return best_dir


def discover_any_dataset_run_ckpt_dir(
    state: PipelineState,
    config_suffix: str,
    *,
    required_file: Optional[str] = None,
) -> Optional[str]:
    """Newest ``*-<any_dataset>-<config_suffix>`` dir with optional required file."""
    best_dir: Optional[str] = None
    best_mtime = 0.0
    roots = _candidate_phase1_ckpt_roots(state)
    for ckpt_root in roots:
        try:
            names = os.listdir(ckpt_root)
        except OSError:
            continue
        for name in names:
            if not _run_dir_matches_config_any_dataset(name, config_suffix):
                continue
            path = os.path.join(ckpt_root, name)
            if not os.path.isdir(path):
                continue
            if required_file and not os.path.exists(os.path.join(path, required_file)):
                continue
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = path
    return best_dir


def _phase1_config_suffix(state: PipelineState, config_name: str = "binary_dual_scale_staged") -> str:
    """Grid checkpoint stems use raw --dataset, not data_subset subset_id."""
    return f"-{state.dataset}-{config_name}"


def _discover_phase1_source_dir(
    state: PipelineState,
    *,
    config_name: str = "binary_dual_scale_staged",
) -> Optional[str]:
    """Newest *-<dataset>-<config_name> dir under ckpts/ with diff_hp.json."""
    best_dir: Optional[str] = None
    best_mtime = 0.0
    for ckpt_root in _candidate_phase1_ckpt_roots(state):
        try:
            names = os.listdir(ckpt_root)
        except OSError:
            continue
        for name in names:
            if not _run_dir_matches_config(name, state.dataset, config_name):
                continue
            path = os.path.join(ckpt_root, name)
            if not os.path.isdir(path):
                continue
            if not os.path.isfile(os.path.join(path, "diff_hp.json")):
                continue
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = path
    return best_dir


def _resolve_phase1_path(state: PipelineState, value: str) -> str:
    raw = value.format(
        dataset=state.dataset,
        subset_id=state.subset_id or state.dataset,
    )
    expanded = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(expanded):
        return expanded
    # Relative to ckpts/ (same layout as submit_grid: $STORE/ckpts/<stem>/)
    under_ckpts = os.path.join(_phase1_ckpt_root(state), raw)
    return os.path.abspath(under_ckpts)


def _phase1_source_dir(
    state: PipelineState,
    override: Optional[str] = None,
    *,
    config_name: str = "binary_dual_scale_staged",
) -> Optional[str]:
    value = override or state.extra.get("phase1_source_dir")
    if value:
        path = _resolve_phase1_path(state, str(value))
        if os.path.isdir(path):
            return path
        logger.warning("phase1_source_dir missing (%s); auto-discovering", path)
    discovered = _discover_phase1_source_dir(state, config_name=config_name)
    if discovered:
        logger.info("Using auto-discovered Phase 1 source: %s", discovered)
        return discovered
    return None


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _resolve_diff_hp(state: PipelineState, source_dir: Optional[str]) -> Dict[str, Any]:
    yaml_fixed = state.extra.get("fixed_synthetic_diff_hp")
    if yaml_fixed:
        params = dict(yaml_fixed)
        logger.info("Using YAML fixed_synthetic_diff_hp: %s", params)
        state.diffusion_best_params = params
        return params

    if state.extra.get("use_hardcoded_synthetic_hp", False):
        logger.info("Using hardcoded Phase 1 diffusion HP (use_hardcoded_synthetic_hp=True)")
        params = {"learning_rate": 0.0005, "batch_size": getattr(state, "diffusion_batch_size", 32)}
        state.diffusion_best_params = params
        return params

    candidates = []
    if source_dir:
        candidates.append(os.path.join(source_dir, "diff_hp.json"))
    candidates.append(os.path.join(state.checkpoint_dir, "diff_hp.json"))
    for path in candidates:
        if path and os.path.exists(path):
            logger.info("Using fixed Phase 1 diffusion HP from %s", path)
            params = _read_json(path)
            state.diffusion_best_params = params
            return params
    suffix = _phase1_config_suffix(state)
    raise FileNotFoundError(
        f"Staged pretrain requires Phase 1 diff_hp.json for {state.dataset!r}. "
        f"Expected *{suffix} under one of {_candidate_phase1_ckpt_roots(state)} "
        "or set phase1_source_dir or set use_hardcoded_synthetic_hp=True."
    )


def _resolve_itrans_pretrain(
    state: PipelineState,
    source_dir: Optional[str],
    *,
    retrain_synthetic_itrans: bool = False,
) -> tuple[str, Dict[str, Any]]:
    candidates = []
    if state.itrans_pretrain_ckpt:
        candidates.append(state.itrans_pretrain_ckpt)
    if source_dir and not retrain_synthetic_itrans:
        candidates.extend([
            os.path.join(source_dir, "pretrained_itransformer.pt"),
            os.path.join(source_dir, "itransformer.pt"),
            os.path.join(source_dir, "itrans_hp_best.pt"),
        ])
    candidates.append(os.path.join(state.checkpoint_dir, "itransformer.pt"))
    for path in candidates:
        if path and os.path.exists(path):
            state.itrans_pretrain_ckpt = path
            return path, {
                "loaded": True,
                "path": os.path.abspath(path),
                "source": "checkpoint",
            }
    logger.warning("No iTransformer pretrain checkpoint found.")
    from models.diffusion_tsf.train_multivariate_pipeline import run_itransformer_hp_tuning
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    import shutil
    
    # Patch globals before running dummy tuning
    patch_globals(pipeline_mod, state, honor_dataset_windows=False)
    
    # By default, do NOT skip tuning if n_itrans_hp_trials > 1.
    # The pipeline yaml can explicitly set extra: skip_synthetic_tuning: true to bypass.
    skip_tuning = state.extra.get("skip_synthetic_tuning", False)
    
    n_trials = getattr(state, "n_itrans_hp_trials", 1)
    if state.smoke_test:
        n_trials = 1
        logger.warning("Smoke test: Training a dummy 1-epoch iTransformer as fallback.")
    elif skip_tuning:
        n_trials = 1
        logger.info("skip_synthetic_tuning=True: Training a 1-trial iTransformer pretrain as fallback (bypassing full Optuna sweep).")
    else:
        logger.info(f"Training iTransformer from scratch with {n_trials} trials.")

    best_params, tune_ckpt_path = run_itransformer_hp_tuning(
        n_trials=n_trials,
        smoke_test=state.smoke_test,
        checkpoint_dir=state.checkpoint_dir,
        parallel_workers=state.parallel_optuna_workers if not skip_tuning else 1,
    )
    
    itrans_ckpt = os.path.join(state.checkpoint_dir, "itransformer.pt")
    if tune_ckpt_path and os.path.exists(tune_ckpt_path) and not os.path.exists(itrans_ckpt):
        shutil.copy2(tune_ckpt_path, itrans_ckpt)
        
    state.itrans_pretrain_ckpt = itrans_ckpt
    return itrans_ckpt, {
        "loaded": False,
        "path": os.path.abspath(itrans_ckpt),
        "source": "trained_fallback",
    }


def _log_staged_pretrain_diagnostics(
    state: PipelineState,
    *,
    itrans_ckpt: str,
    itrans_meta: Dict[str, Any],
    best_params: Dict[str, Any],
    n_samples: int,
    stages: Optional[list] = None,
) -> None:
    stage_list = stages or list(staged_diffusion_stages(state))
    try:
        for i, stage in enumerate(stage_list):
            ckpt = {
                "coarse": state.diffusion_coarse_pretrain_ckpt,
                "fine": state.diffusion_fine_pretrain_ckpt,
                "finer": state.diffusion_finer_pretrain_ckpt,
            }.get(stage)
            result = run_staged_synthetic_pretrain_diagnostics(
                state,
                itrans_ckpt_path=itrans_ckpt,
                itrans_meta=itrans_meta,
                tuned_params=best_params,
                n_samples=n_samples,
                stage=stage,
                diffusion_ckpt_path=ckpt,
                include_dataset_stats=(i == 0),
                include_phase_start=True,
            )
            wandb_utils.log_phase_diagnostics_result(result)
    except Exception as e:
        logger.warning("Staged synthetic-pretrain diagnostics failed: %s", e, exc_info=True)


def patch_stage_globals(
    mod: Any,
    state: PipelineState,
    stage: str,
    *,
    honor_dataset_windows: bool,
    for_synthetic_pretrain: bool = False,
) -> None:
    """Patch legacy train module globals for a single staged model."""
    if stage not in {"coarse", "fine", "finer"}:
        raise ValueError(f"Unknown staged diffusion stage: {stage!r}")
    if stage == "finer" and not state.use_triple_scale:
        raise ValueError("finer staged diffusion requires state.use_triple_scale=True")
    patch_globals(mod, state, honor_dataset_windows=honor_dataset_windows)
    mod.USE_TRIPLE_SCALE = bool(state.use_triple_scale)
    mod.DIFFUSION_STAGE = stage
    mod.IMAGE_HEIGHT = {
        "coarse": int(state.coarse_image_height),
        "fine": int(state.fine_image_height),
        "finer": int(state.finer_image_height),
    }[stage]
    mod.COARSE_IMAGE_HEIGHT = int(state.coarse_image_height)
    mod.FINE_IMAGE_HEIGHT = int(state.fine_image_height)
    mod.FINER_IMAGE_HEIGHT = int(state.finer_image_height)
    mod.USE_GUIDANCE_CHANNEL = state.use_guidance_channel
    mod.STAGED_REPRESENTATION = state.staged_representation
    mod.HAAR_HIGH_FREQ_PERCENT = float(state.haar_high_freq_percent)
    mod.HAAR_HIGH_FREQ_LEVELS = int(state.haar_high_freq_levels)
    mod.HAAR_FINE_MAX_SCALE = float(state.haar_fine_max_scale)
    mod.FOURIER_HIGH_FREQ_PERCENT = float(state.fourier_high_freq_percent)
    mod.FOURIER_HIGH_FREQ_CUTOFF_BIN = int(state.fourier_high_freq_cutoff_bin)
    mod.FOURIER_FINE_MAX_SCALE = float(state.fourier_fine_max_scale)
    mod.FOURIER_FLATLINE_ATOL = float(state.fourier_flatline_atol)
    mod.FOURIER_HIGH_FREQ_CUTOFF_BINS_PER_VARIATE = (
        list(state.fourier_high_freq_cutoff_bins_per_variate)
        if state.fourier_high_freq_cutoff_bins_per_variate
        else None
    )
    mod.FOURIER_FINE_MAX_SCALE_PER_VARIATE = (
        list(state.fourier_fine_max_scale_per_variate)
        if state.fourier_fine_max_scale_per_variate
        else None
    )
    mod.COARSE_FLATLINE_BLUR_FINE_TARGET = bool(state.coarse_flatline_blur_fine_target)
    mod.COARSE_FLATLINE_BLUR_RADIUS = int(state.coarse_flatline_blur_radius)
    mod.COARSE_FLATLINE_BLUR_KERNEL = str(state.coarse_flatline_blur_kernel)
    mod.COARSE_FLATLINE_BLUR_ATOL = state.coarse_flatline_blur_atol
    if for_synthetic_pretrain:
        mod.USE_ORDINAL_WINDOW_NORM = False
        mod.GLOBAL_ORDINAL_LADDER = None


class StagedDiffusionPretrainPhase(PipelinePhase):
    name = "staged_diffusion_pretrain"

    def _config_name(self, state: PipelineState) -> str:
        if "phase1_config_name" in self.overrides:
            return str(self.require("phase1_config_name"))
        if state.extra.get("phase1_config_name"):
            return str(state.extra["phase1_config_name"])
        raise KeyError(f"phase {self.name!r} missing required key 'phase1_config_name'")

    def _cached_stage_ckpt(self, state: PipelineState, config_name: str, stage: str) -> Optional[str]:
        reuse_from = self.get("reuse_pretrain_from_config")
        if reuse_from:
            reused = source_run_stage_pretrain_ckpt(state, str(reuse_from), stage)
            if reused:
                logger.info(
                    "  [%s] %s reused pretrain from *-%s-%s: %s",
                    self.name,
                    stage,
                    state.dataset,
                    reuse_from,
                    reused,
                )
                return reused
            return None

        local_ckpt = _stage_pretrain_ckpt(state, stage)
        if os.path.exists(local_ckpt):
            if _stage_pretrain_cache_enabled(self, state):
                logger.info("  [%s] %s local cached: %s", self.name, stage, local_ckpt)
                return local_ckpt
            sig_path = os.path.join(_stage_pretrain_dir(state, stage), ".signature")
            expected = _stage_pretrain_signature(state, config_name)
            if os.path.isfile(sig_path):
                with open(sig_path, encoding="utf-8") as f:
                    if f.read().strip() == expected:
                        logger.info(
                            "  [%s] %s local cached (signature match): %s",
                            self.name,
                            stage,
                            local_ckpt,
                        )
                        return local_ckpt
            logger.info(
                "  [%s] %s ignoring stale local pretrain (shared_cache=false): %s",
                self.name,
                stage,
                local_ckpt,
            )
        if _stage_pretrain_cache_enabled(self, state):
            shared_ckpt = _shared_stage_pretrain_ckpt(state, config_name, stage)
            if os.path.exists(shared_ckpt):
                logger.info("  [%s] %s shared cached: %s", self.name, stage, shared_ckpt)
                return shared_ckpt
            discovered = _discover_existing_stage_pretrain(state, stage)
            if discovered:
                logger.info("  [%s] %s discovered cached: %s", self.name, stage, discovered)
                return discovered
        return None

    def should_skip(self, state: PipelineState) -> bool:
        ensure_haar_frequency_calibration(state)
        ensure_fourier_frequency_calibration(state)
        config_name = self._config_name(state)
        ckpts = {
            stage: self._cached_stage_ckpt(state, config_name, stage)
            for stage in staged_diffusion_stages(state)
        }
        if all(ckpts.values()):
            state.diffusion_coarse_pretrain_ckpt = ckpts["coarse"]
            state.diffusion_fine_pretrain_ckpt = ckpts["fine"]
            if state.use_triple_scale:
                state.diffusion_finer_pretrain_ckpt = ckpts["finer"]
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import pretrain_diffusion
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        config_name = self._config_name(state)
        ensure_haar_frequency_calibration(state)
        ensure_fourier_frequency_calibration(state)
        reuse_from = self.get("reuse_pretrain_from_config")
        if reuse_from:
            missing = []
            for stage in staged_diffusion_stages(state):
                ckpt = self._cached_stage_ckpt(state, config_name, stage)
                if ckpt:
                    if stage == "coarse":
                        state.diffusion_coarse_pretrain_ckpt = ckpt
                    elif stage == "fine":
                        state.diffusion_fine_pretrain_ckpt = ckpt
                    else:
                        state.diffusion_finer_pretrain_ckpt = ckpt
                else:
                    missing.append(stage)
            if not missing:
                source_dir = _phase1_source_dir(
                    state,
                    self.get("phase1_source_dir"),
                    config_name=config_name,
                )
                best_params = _resolve_diff_hp(state, source_dir)
                itrans_ckpt, itrans_meta = _resolve_itrans_pretrain(
                    state,
                    source_dir,
                    retrain_synthetic_itrans=bool(self.get("retrain_synthetic_itrans", False)),
                )
                n_samples = int(self.require("n_samples"))
                if state.smoke_test:
                    n_samples = min(n_samples, 4)
                _log_staged_pretrain_diagnostics(
                    state,
                    itrans_ckpt=itrans_ckpt,
                    itrans_meta=itrans_meta,
                    best_params=best_params,
                    n_samples=n_samples,
                )
                return state
            # Soft-fail like patch_guidance: quota may have deleted the donor.
            logger.warning(
                "  [%s] reuse_pretrain_from_config=%r missing pretrained_%s under "
                "*-%s-%s (incl. cross-dataset fallback); training synthetic pretrain instead",
                self.name,
                reuse_from,
                "/pretrained_".join(missing),
                state.dataset,
                reuse_from,
            )
        source_dir = _phase1_source_dir(
            state,
            self.get("phase1_source_dir"),
            config_name=config_name,
        )
        best_params = _resolve_diff_hp(state, source_dir)
        itrans_ckpt, itrans_meta = _resolve_itrans_pretrain(
            state,
            source_dir,
            retrain_synthetic_itrans=bool(self.get("retrain_synthetic_itrans", False)),
        )

        n_samples = int(self.require("n_samples"))
        epochs = int(self.require("epochs"))
        patience = int(self.require("patience"))
        if state.smoke_test:
            n_samples = min(n_samples, 4)
            epochs = 1
            patience = 1

        shared_cache = _stage_pretrain_cache_enabled(self, state)
        shared_wait_seconds = float(self.get("shared_cache_wait_seconds", 6 * 60 * 60))

        for stage in staged_diffusion_stages(state):
            ckpt = self._cached_stage_ckpt(state, config_name, stage)
            if ckpt is None and shared_cache:
                shared_ckpt = _wait_for_shared_stage_ckpt(
                    state,
                    config_name,
                    stage,
                    wait_seconds=shared_wait_seconds,
                )
                if shared_ckpt:
                    ckpt = shared_ckpt
                else:
                    ckpt = _shared_stage_pretrain_ckpt(state, config_name, stage)
                    stage_dir = os.path.dirname(ckpt)
                    os.makedirs(stage_dir, exist_ok=True)
                    try:
                        patch_stage_globals(
                            pipeline_mod, state, stage,
                            honor_dataset_windows=False, for_synthetic_pretrain=True,
                        )
                        ckpt = pretrain_diffusion(
                            best_params=best_params,
                            itrans_checkpoint=itrans_ckpt,
                            n_samples=n_samples,
                            epochs=epochs,
                            patience=patience,
                            checkpoint_dir=stage_dir,
                            smoke_test=state.smoke_test,
                        )
                    finally:
                        _release_shared_lock(ckpt)

                    meta_path = os.path.join(stage_dir, "shared_pretrain_metadata.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dataset": state.dataset,
                                "n_variates": state.n_variates,
                                "config_name": config_name,
                                "stage": stage,
                                "signature": _stage_pretrain_signature(state, config_name),
                                "checkpoint": ckpt,
                            },
                            f,
                            indent=2,
                            sort_keys=True,
                        )
            elif ckpt is None:
                stage_dir = _stage_pretrain_dir(state, stage)
                os.makedirs(stage_dir, exist_ok=True)
                patch_stage_globals(
                    pipeline_mod, state, stage,
                    honor_dataset_windows=False, for_synthetic_pretrain=True,
                )
                ckpt = pretrain_diffusion(
                    best_params=best_params,
                    itrans_checkpoint=itrans_ckpt,
                    n_samples=n_samples,
                    epochs=epochs,
                    patience=patience,
                    checkpoint_dir=stage_dir,
                    smoke_test=state.smoke_test,
                )
                sig_path = os.path.join(stage_dir, ".signature")
                with open(sig_path, "w", encoding="utf-8") as f:
                    f.write(_stage_pretrain_signature(state, config_name))
            if stage == "coarse":
                state.diffusion_coarse_pretrain_ckpt = ckpt
            elif stage == "fine":
                state.diffusion_fine_pretrain_ckpt = ckpt
            else:
                state.diffusion_finer_pretrain_ckpt = ckpt

        _log_staged_pretrain_diagnostics(
            state,
            itrans_ckpt=itrans_ckpt,
            itrans_meta=itrans_meta,
            best_params=best_params,
            n_samples=n_samples,
        )

        viz_ckpt = state.diffusion_fine_pretrain_ckpt or state.diffusion_coarse_pretrain_ckpt
        if viz_ckpt and itrans_ckpt and not state.smoke_test:
            try:
                viz_paths = run_pretrain_diffusion_visualizations(
                    state,
                    coarse_ckpt_path=state.diffusion_coarse_pretrain_ckpt,
                    fine_ckpt_path=state.diffusion_fine_pretrain_ckpt,
                    itrans_ckpt_path=itrans_ckpt,
                    tuned_params=best_params,
                    tag="staged_diffusion_synthetic_pretrain",
                )
                wandb_utils.log_visualization_paths(
                    viz_paths, wandb_key="viz/staged_diffusion_synthetic_pretrain",
                )
            except Exception as e:
                logger.warning("Staged synthetic-pretrain viz failed: %s", e, exc_info=True)

        return state
