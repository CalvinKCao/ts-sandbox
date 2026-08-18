"""Fixed-HP synthetic pretrain for staged coarse/fine diffusion models."""

from __future__ import annotations

import json
import logging
import os
import hashlib
import shutil
import time
from dataclasses import replace
from typing import Any, Dict, Optional

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.reused_paths import find_reused_pretrain_ckpt
from models.diffusion_tsf.pipeline.visualize_utils import (
    run_staged_synthetic_pretrain_diagnostics,
)

logger = logging.getLogger(__name__)

def _stage_pretrain_dir(state: PipelineState, stage: str) -> str:
    return os.path.join(state.checkpoint_dir, f"pretrained_{stage}")


def _stage_pretrain_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_pretrain_dir(state, stage), "pretrained_diffusion.pt")


def staged_diffusion_stages(state: PipelineState) -> tuple[str, ...]:
    """Live binary path is coarse + absolute-HIR patch_refine only."""
    if not getattr(state, "use_patch_refine_stage", False):
        raise ValueError(
            "use_patch_refine_stage must be true; residual fine-as-primary path was removed"
        )
    return ("coarse", "patch_refine")


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
        "use_patch_refine_stage": bool(getattr(state, "use_patch_refine_stage", False)),
        "patch_refine_canvas_height": int(getattr(state, "patch_refine_canvas_height", 256)),
        "patch_refine_patch_height": int(getattr(state, "patch_refine_patch_height", 32)),
        "patch_refine_patch_width": int(getattr(state, "patch_refine_patch_width", 8)),
        "patch_refine_col_stride": int(getattr(state, "patch_refine_col_stride", 6)),
        "staged_representation": str(state.staged_representation),
        "max_scale": max_scale,
        "dit_patch_size": list(state.dit_patch_size),
        "dit_embed_dim": int(state.dit_embed_dim),
        "dit_depth": int(state.dit_depth),
        "dit_num_heads": int(state.dit_num_heads),
        "dit_mlp_ratio": float(state.dit_mlp_ratio),
        "dit_dropout": float(state.dit_dropout),
        "conditioning_architecture": "cross_attention_only_v1",
        "disable_cross_attention": bool(state.disable_cross_attention),
        "guidance_type": str(getattr(state, "guidance_type", "patch_decoder")),
        "prediction_target": str(getattr(state, "prediction_target", "x0")),
        "loss_weighting": str(getattr(state, "loss_weighting", "none")),
        "min_snr_gamma": float(getattr(state, "min_snr_gamma", 5.0)),
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
        "binary_cdf_distance_alpha": float(state.binary_cdf_distance_alpha),
        "binary_noise_schedule": str(state.binary_noise_schedule),
        "binary_length_mode": str(state.binary_length_mode),
        "binary_length_g": float(state.binary_length_g),
        "binary_length_scale": float(state.binary_length_scale),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return (
        f"{config_name}-v{payload['n_variates']}-h{payload['image_height']}"
        f"-ms{max_scale:g}-{digest}"
    )


def _stage_max_scale_for_dataset(state: PipelineState, dataset: str) -> float:
    return float(state.max_scale_by_dataset.get(dataset, state.max_scale))


def _current_run_config_suffix(state: PipelineState) -> str:
    """Infer the submit_binary config suffix from the isolated checkpoint dir name."""
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


def _pretrain_has_xattn_marker(path: str) -> bool:
    """True when the ckpt is the live cross-attention-only architecture."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    payload = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    return isinstance(payload, dict) and "conditioning_architecture_version" in payload


def source_run_stage_pretrain_ckpt(
    state: PipelineState,
    source_config: str,
    stage: str,
) -> Optional[str]:
    """``pretrained_{stage}/pretrained_diffusion.pt`` from reused store or a prior grid run."""
    reused = find_reused_pretrain_ckpt(source_config, stage)
    if reused:
        if _pretrain_has_xattn_marker(reused):
            logger.info(
                "  [staged_diffusion_pretrain] %s reused pretrain from %s",
                stage,
                reused,
            )
            return reused
        logger.warning(
            "  [staged_diffusion_pretrain] %s skipping reused pretrain without "
            "cross-attention-only marker: %s",
            stage,
            reused,
        )

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
    # Relative to ckpts/ (same layout as submit_binary: $STORE/ckpts/<stem>/)
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
    force = bool(state.extra.get("force_retrain_synthetic", False))
    if not force:
        for path in candidates:
            if path and os.path.exists(path):
                logger.info("Using fixed Phase 1 diffusion HP from %s", path)
                params = _read_json(path)
                state.diffusion_best_params = params
                return params

    n_trials = int(getattr(state, "n_diffusion_hp_trials", 0) or state.extra.get("n_diffusion_hp_trials", 0) or 0)
    if n_trials > 0:
        params = _run_synthetic_diffusion_hp_tuning(state, n_trials=n_trials)
        out_path = os.path.join(state.checkpoint_dir, "diff_hp.json")
        os.makedirs(state.checkpoint_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        logger.info("Wrote synthetic diffusion HP to %s: %s", out_path, params)
        state.diffusion_best_params = params
        return params

    suffix = _phase1_config_suffix(state)
    raise FileNotFoundError(
        f"Staged pretrain requires Phase 1 diff_hp.json for {state.dataset!r}. "
        f"Expected *{suffix} under one of {_candidate_phase1_ckpt_roots(state)} "
        "or set phase1_source_dir / use_hardcoded_synthetic_hp=True / n_diffusion_hp_trials>0."
    )


def _run_synthetic_diffusion_hp_tuning(state: PipelineState, *, n_trials: int) -> Dict[str, Any]:
    """Optuna LR search on tiny synthetic coarse pretrain (writes trial dirs under ckpt)."""
    import optuna
    from models.diffusion_tsf.pipeline.train.pretrain import pretrain_diffusion

    lr_min = float(state.extra.get("finetune_hp_lr_min", 1e-4))
    lr_max = float(state.extra.get("finetune_hp_lr_max", 5e-4))
    if lr_min >= lr_max:
        lr_max = lr_min * 5.0
    batch_size = int(getattr(state, "diffusion_batch_size", 2) or 2)
    n_samples = int(state.extra.get("synthetic_samples_diff_tune", 32) or 32)
    n_samples = max(20, min(n_samples, 64))  # need val_tail > 0 (n_samples//10)
    epochs = 1
    patience = 1
    study = optuna.create_study(direction="minimize", study_name="synthetic_diffusion_hp")
    logger.info(
        "Synthetic diffusion HP: %s trials, lr=[%.2e, %.2e], n_samples=%s, batch=%s",
        n_trials,
        lr_min,
        lr_max,
        n_samples,
        batch_size,
    )

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
        params = {"learning_rate": float(lr), "batch_size": batch_size}
        trial_dir = os.path.join(state.checkpoint_dir, f"_synth_diff_hp_trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        # Do not pass smoke_test=True — that zeros the val split and makes every trial look identical.
        _ckpt, best_val = pretrain_diffusion(
                stage_state(state, "coarse", for_synthetic_pretrain=True),
                best_params=params,
                guidance_checkpoint="",
                n_samples=n_samples,
                epochs=epochs,
                patience=patience,
                checkpoint_dir=trial_dir,
                smoke_test=False,
            )
        return float(best_val)

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=not state.smoke_test)
    best = study.best_params
    out = {
        "learning_rate": float(best["learning_rate"]),
        "batch_size": batch_size,
    }
    logger.info("Synthetic diffusion HP best: %s (val=%.4f)", out, study.best_value)
    return out


def _guidance_search_dirs(
    state: PipelineState,
    source_dir: Optional[str] = None,
) -> list[str]:
    """Dirs that may hold synthetic patch-guidance weights used with pretrain."""
    dirs: list[str] = []

    def add(path: Optional[str]) -> None:
        if not path:
            return
        path = os.path.abspath(path)
        if path not in dirs and os.path.isdir(path):
            dirs.append(path)

    add(source_dir)
    add(state.checkpoint_dir)
    for attr in (
        "diffusion_coarse_pretrain_ckpt",
        "diffusion_patch_refine_pretrain_ckpt",
    ):
        ckpt = getattr(state, attr, None)
        if ckpt and os.path.isfile(ckpt):
            # .../pretrained_<stage>/pretrained_diffusion.pt → run dir
            add(os.path.dirname(os.path.dirname(ckpt)))
            add(os.path.dirname(ckpt))
    return dirs


def _find_existing_synthetic_patch_guidance(
    state: PipelineState,
    source_dir: Optional[str] = None,
) -> Optional[str]:
    """Locate an on-disk synthetic guidance checkpoint."""
    ckpt_names = ("pretrained_patch_guidance.pt", "patch_guidance_synthetic.pt")
    candidates: list[str] = []
    pretrain_override = state.extra.get("patch_guidance_pretrain_ckpt")
    if pretrain_override:
        candidates.append(str(pretrain_override))
    for d in _guidance_search_dirs(state, source_dir):
        for name in ckpt_names:
            candidates.append(os.path.join(d, name))
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def _resolve_synthetic_patch_guidance(
    state: PipelineState,
    source_dir: Optional[str],
) -> tuple[str, Dict[str, Any]]:
    found = _find_existing_synthetic_patch_guidance(state, source_dir)
    if found:
        return found, {
            "loaded": True,
            "path": found,
            "source": "checkpoint",
        }
    raise FileNotFoundError(
        "Missing synthetic patch-guidance checkpoint. Expected "
        "pretrained_patch_guidance.pt or patch_guidance_synthetic.pt under "
        f"{source_dir or state.checkpoint_dir}."
    )


def _build_synthetic_patch_guidance(state: PipelineState) -> tuple[str, Dict[str, Any]]:
    """Train the frozen patch-decoder token source when no donor exists."""
    from models.diffusion_tsf.train_multivariate_pipeline import (
        run_patch_guidance_synthetic_tuning,
    )

    n_trials = 1 if state.smoke_test else int(state.n_itrans_hp_trials)
    _params, tuned_ckpt = run_patch_guidance_synthetic_tuning(
        state,
        n_trials=n_trials,
        smoke_test=state.smoke_test,
        checkpoint_dir=state.checkpoint_dir,
        parallel_workers=1,
    )
    if not tuned_ckpt or not os.path.isfile(tuned_ckpt):
        raise RuntimeError("synthetic patch-guidance tuning did not produce a checkpoint")
    target = os.path.join(state.checkpoint_dir, "patch_guidance_synthetic.pt")
    shutil.copy2(tuned_ckpt, target)
    return target, {"loaded": False, "path": target, "source": "trained"}


def _log_staged_pretrain_diagnostics(
    state: PipelineState,
    *,
    guidance_ckpt: str,
    guidance_meta: Dict[str, Any],
    best_params: Dict[str, Any],
    n_samples: int,
    stages: Optional[list] = None,
) -> None:
    stage_list = stages or list(staged_diffusion_stages(state))
    try:
        for i, stage in enumerate(stage_list):
            ckpt = {
                "coarse": state.diffusion_coarse_pretrain_ckpt,
                "patch_refine": state.diffusion_patch_refine_pretrain_ckpt,
            }.get(stage)
            result = run_staged_synthetic_pretrain_diagnostics(
                state,
                itrans_ckpt_path=guidance_ckpt,
                itrans_meta=guidance_meta,
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


def stage_state(
    state: PipelineState,
    stage: str,
    *,
    honor_dataset_windows: bool = True,
    for_synthetic_pretrain: bool = False,
) -> PipelineState:
    """Return a stage-local state without mutating the pipeline's base state."""
    if stage not in {"coarse", "patch_refine"}:
        raise ValueError(f"Unknown staged diffusion stage: {stage!r}")
    image_height = {
        "coarse": int(state.coarse_image_height),
        "patch_refine": int(getattr(state, "patch_refine_patch_height", 32)),
    }[stage]
    updates = {
        "diffusion_stage": stage,
        "image_height": image_height,
    }
    if for_synthetic_pretrain:
        updates.update({
            "use_ordinal_window_norm": False,
            "ordinal_ood_shift_causal_only": False,
            "ordinal_ladder": None,
        })
    return replace(state, **updates)


class StagedDiffusionPretrainPhase(PipelinePhase):
    name = "staged_diffusion_pretrain"

    def _requires_reused_pretrain(self) -> bool:
        return bool(self.get("require_reuse_pretrain", False))

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
        return None

    def should_skip(self, state: PipelineState) -> bool:
        config_name = self._config_name(state)
        reuse_from = self.get("reuse_pretrain_from_config")
        if self._requires_reused_pretrain() and not reuse_from:
            raise ValueError(
                "require_reuse_pretrain=true requires reuse_pretrain_from_config"
            )
        ckpts = {
            stage: self._cached_stage_ckpt(state, config_name, stage)
            for stage in staged_diffusion_stages(state)
        }
        if self._requires_reused_pretrain():
            missing = [stage for stage, ckpt in ckpts.items() if not ckpt]
            if missing:
                required = ", ".join(
                    f"pretrained_{stage}/pretrained_diffusion.pt" for stage in missing
                )
                raise FileNotFoundError(
                    "Required synthetic staged-pretrain donor missing for "
                    f"dataset={state.dataset!r}, config={reuse_from!r}: {required}. "
                    "Refusing to train a replacement synthetic pretrain."
                )
        if all(ckpts.values()):
            state.diffusion_coarse_pretrain_ckpt = ckpts["coarse"]
            state.diffusion_patch_refine_pretrain_ckpt = ckpts["patch_refine"]
            return True
        return False

    def on_skip(self, state: PipelineState) -> PipelineState:
        """Cached/reused pretrain still gets a couple RealTS pred panels + wandb."""
        try:
            config_name = self._config_name(state)
            source_dir = None
            try:
                source_dir = _phase1_source_dir(
                    state,
                    self.get("phase1_source_dir"),
                    config_name=config_name,
                )
            except FileNotFoundError:
                source_dir = None
            best_params: Dict[str, Any] = {}
            try:
                best_params = _resolve_diff_hp(state, source_dir)
            except FileNotFoundError:
                pass
            guidance_ckpt = _find_existing_synthetic_patch_guidance(state, source_dir)
        except Exception as e:
            logger.warning(
                "[%s] cached-pretrain viz failed: %s", self.name, e, exc_info=True,
            )
        return state

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.pipeline.train.pretrain import pretrain_diffusion

        config_name = self._config_name(state)
        reuse_from = self.get("reuse_pretrain_from_config")
        if reuse_from:
            missing = []
            for stage in staged_diffusion_stages(state):
                ckpt = self._cached_stage_ckpt(state, config_name, stage)
                if ckpt:
                    if stage == "coarse":
                        state.diffusion_coarse_pretrain_ckpt = ckpt
                    elif stage == "patch_refine":
                        state.diffusion_patch_refine_pretrain_ckpt = ckpt
                else:
                    missing.append(stage)
            if not missing:
                source_dir = _phase1_source_dir(
                    state,
                    self.get("phase1_source_dir"),
                    config_name=config_name,
                )
                best_params = _resolve_diff_hp(state, source_dir)
                guidance_ckpt, guidance_meta = _resolve_synthetic_patch_guidance(
                    state,
                    source_dir,
                )
                n_samples = int(self.require("n_samples"))
                if state.smoke_test:
                    n_samples = min(n_samples, 4)
                _log_staged_pretrain_diagnostics(
                    state,
                    guidance_ckpt=guidance_ckpt,
                    guidance_meta=guidance_meta,
                    best_params=best_params,
                    n_samples=n_samples,
                )
                return state
            required = ", ".join(
                f"pretrained_{stage}/pretrained_diffusion.pt" for stage in missing
            )
            raise FileNotFoundError(
                "Required synthetic staged-pretrain donor missing for "
                f"dataset={state.dataset!r}, config={reuse_from!r}: {required}. "
                "Refusing to train a replacement synthetic pretrain."
            )
        source_dir = _phase1_source_dir(
            state,
            self.get("phase1_source_dir"),
            config_name=config_name,
        )
        best_params = _resolve_diff_hp(state, source_dir)
        needs_guidance = state.needs_guidance
        if needs_guidance:
            try:
                guidance_ckpt, guidance_meta = _resolve_synthetic_patch_guidance(
                    state, source_dir,
                )
            except FileNotFoundError:
                guidance_ckpt, guidance_meta = _build_synthetic_patch_guidance(state)
        else:
            guidance_ckpt, guidance_meta = "", {}

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
            stage_epochs = 1 if state.smoke_test else int(self.get(f"{stage}_epochs", epochs))
            stage_patience = min(patience, stage_epochs)
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
                        # Stamp signature before training so a mid-run kill can resume
                        # from pretrained_diffusion.pt (best-so-far) without retraining.
                        sig_path = os.path.join(stage_dir, ".signature")
                        with open(sig_path, "w", encoding="utf-8") as f:
                            f.write(_stage_pretrain_signature(state, config_name))
                        ckpt, _best_val = pretrain_diffusion(
                            stage_state(state, stage, for_synthetic_pretrain=True),
                            best_params=best_params,
                            guidance_checkpoint=guidance_ckpt,
                            n_samples=n_samples,
                            epochs=stage_epochs,
                            patience=stage_patience,
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
                # Stamp before training so mid-run kills can reuse best-so-far weights.
                sig_path = os.path.join(stage_dir, ".signature")
                with open(sig_path, "w", encoding="utf-8") as f:
                    f.write(_stage_pretrain_signature(state, config_name))
                ckpt, _best_val = pretrain_diffusion(
                    stage_state(state, stage, for_synthetic_pretrain=True),
                    best_params=best_params,
                    guidance_checkpoint=guidance_ckpt,
                    n_samples=n_samples,
                    epochs=stage_epochs,
                    patience=stage_patience,
                    checkpoint_dir=stage_dir,
                    smoke_test=state.smoke_test,
                )
            if stage == "coarse":
                state.diffusion_coarse_pretrain_ckpt = ckpt
            elif stage == "patch_refine":
                state.diffusion_patch_refine_pretrain_ckpt = ckpt

        _log_staged_pretrain_diagnostics(
            state,
            guidance_ckpt=guidance_ckpt,
            guidance_meta=guidance_meta,
            best_params=best_params,
            n_samples=n_samples,
        )

        return state
