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
from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import _patch_globals

logger = logging.getLogger(__name__)


def _stage_pretrain_dir(state: PipelineState, stage: str) -> str:
    return os.path.join(state.checkpoint_dir, f"pretrained_{stage}")


def _stage_pretrain_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_pretrain_dir(state, stage), "pretrained_diffusion.pt")


def _stage_pretrain_cache_enabled(phase: PipelinePhase, state: PipelineState) -> bool:
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
        "max_scale": max_scale,
        "dit_patch_size": list(state.dit_patch_size),
        "use_guidance_channel": bool(state.use_guidance_channel),
        "deterministic_anchor_loss": bool(state.deterministic_anchor_loss),
        "deterministic_anchor_lambda": float(state.deterministic_anchor_lambda),
        "deterministic_anchor_alpha": float(state.deterministic_anchor_alpha),
        "lookback_length": int(state.lookback_length),
        "forecast_length": int(state.forecast_length),
        "use_window_normalization": bool(state.use_window_normalization),
        "window_norm_std_floor": float(state.window_norm_std_floor),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return (
        f"{config_name}-v{payload['n_variates']}-h{payload['image_height']}"
        f"-ms{max_scale:g}-{digest}"
    )


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


def _phase1_ckpt_root(state: PipelineState) -> str:
    """Directory that holds per-run checkpoint folders (*-<dataset>-<config>)."""
    ckpt_dir = os.path.abspath(state.checkpoint_dir)
    if os.path.basename(ckpt_dir) == "ckpts":
        return ckpt_dir
    return os.path.dirname(ckpt_dir)


def _phase1_config_suffix(state: PipelineState, config_name: str = "binary_dual_scale") -> str:
    """Grid checkpoint stems use raw --dataset, not data_subset subset_id."""
    return f"-{state.dataset}-{config_name}"


def _discover_phase1_source_dir(
    state: PipelineState,
    *,
    config_name: str = "binary_dual_scale",
) -> Optional[str]:
    """Newest *-<dataset>-<config_name> dir under ckpts/ with diff_hp.json."""
    ckpt_root = _phase1_ckpt_root(state)
    suffix = _phase1_config_suffix(state, config_name)
    best_dir: Optional[str] = None
    best_mtime = 0.0
    try:
        for name in os.listdir(ckpt_root):
            if not name.endswith(suffix):
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
    except OSError:
        return None
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
    config_name: str = "binary_dual_scale",
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
        f"Expected *{suffix} under {_phase1_ckpt_root(state)} or set phase1_source_dir."
    )


def _resolve_itrans_pretrain(state: PipelineState, source_dir: Optional[str]) -> str:
    candidates = []
    if state.itrans_pretrain_ckpt:
        candidates.append(state.itrans_pretrain_ckpt)
    if source_dir:
        candidates.extend([
            os.path.join(source_dir, "pretrained_itransformer.pt"),
            os.path.join(source_dir, "itransformer.pt"),
            os.path.join(source_dir, "itrans_hp_best.pt"),
        ])
    candidates.append(os.path.join(state.checkpoint_dir, "itransformer.pt"))
    for path in candidates:
        if path and os.path.exists(path):
            state.itrans_pretrain_ckpt = path
            return path
    raise FileNotFoundError(
        "Staged pretrain requires an iTransformer pretrain checkpoint for lookback tokens "
        f"(tried itransformer.pt / itrans_hp_best.pt / pretrained_itransformer.pt under {source_dir!r})."
    )


def patch_stage_globals(mod: Any, state: PipelineState, stage: str, *, honor_dataset_windows: bool) -> None:
    """Patch legacy train module globals for a single staged model."""
    if stage not in {"coarse", "fine"}:
        raise ValueError(f"Unknown staged diffusion stage: {stage!r}")
    _patch_globals(mod, state, honor_dataset_windows=honor_dataset_windows)
    mod.USE_DUAL_SCALE = False
    mod.DIFFUSION_STAGE = stage
    mod.IMAGE_HEIGHT = 16
    mod.USE_GUIDANCE_CHANNEL = state.use_guidance_channel


class StagedDiffusionPretrainPhase(PipelinePhase):
    name = "staged_diffusion_pretrain"

    def _config_name(self, state: PipelineState) -> str:
        return str(
            self.get("phase1_config_name")
            or state.extra.get("phase1_config_name")
            or "binary_dual_scale"
        )

    def _cached_stage_ckpt(self, state: PipelineState, config_name: str, stage: str) -> Optional[str]:
        local_ckpt = _stage_pretrain_ckpt(state, stage)
        if os.path.exists(local_ckpt):
            logger.info("  [%s] %s local cached: %s", self.name, stage, local_ckpt)
            return local_ckpt
        if _stage_pretrain_cache_enabled(self, state):
            shared_ckpt = _shared_stage_pretrain_ckpt(state, config_name, stage)
            if os.path.exists(shared_ckpt):
                logger.info("  [%s] %s shared cached: %s", self.name, stage, shared_ckpt)
                return shared_ckpt
        return None

    def should_skip(self, state: PipelineState) -> bool:
        config_name = self._config_name(state)
        coarse = self._cached_stage_ckpt(state, config_name, "coarse")
        fine = self._cached_stage_ckpt(state, config_name, "fine")
        if coarse and fine:
            state.diffusion_coarse_pretrain_ckpt = coarse
            state.diffusion_fine_pretrain_ckpt = fine
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            DIFFUSION_HP_PATIENCE,
            PRETRAIN_DIFFUSION_EPOCHS,
            SYNTHETIC_SAMPLES_DIFF_TUNE,
            pretrain_diffusion,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        config_name = self._config_name(state)
        source_dir = _phase1_source_dir(
            state,
            self.get("phase1_source_dir"),
            config_name=config_name,
        )
        best_params = _resolve_diff_hp(state, source_dir)
        itrans_ckpt = _resolve_itrans_pretrain(state, source_dir)

        n_samples = int(self.get("n_samples", SYNTHETIC_SAMPLES_DIFF_TUNE))
        epochs = int(self.get("epochs", PRETRAIN_DIFFUSION_EPOCHS))
        patience = int(self.get("patience", DIFFUSION_HP_PATIENCE))
        if state.smoke_test:
            n_samples = min(n_samples, 32)
            epochs = 1
            patience = 1
        shared_cache = _stage_pretrain_cache_enabled(self, state)
        shared_wait_seconds = float(self.get("shared_cache_wait_seconds", 6 * 60 * 60))

        for stage in ("coarse", "fine"):
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
                        patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=False)
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
                patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=False)
                ckpt = pretrain_diffusion(
                    best_params=best_params,
                    itrans_checkpoint=itrans_ckpt,
                    n_samples=n_samples,
                    epochs=epochs,
                    patience=patience,
                    checkpoint_dir=stage_dir,
                    smoke_test=state.smoke_test,
                )
            if stage == "coarse":
                state.diffusion_coarse_pretrain_ckpt = ckpt
            else:
                state.diffusion_fine_pretrain_ckpt = ckpt

        return state
