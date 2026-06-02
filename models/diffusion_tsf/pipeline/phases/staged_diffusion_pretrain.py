"""Fixed-HP synthetic pretrain for staged coarse/fine diffusion models."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import _patch_globals

logger = logging.getLogger(__name__)


def _stage_pretrain_dir(state: PipelineState, stage: str) -> str:
    return os.path.join(state.checkpoint_dir, f"pretrained_{stage}")


def _stage_pretrain_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_pretrain_dir(state, stage), "pretrained_diffusion.pt")


def _phase1_ckpt_root(state: PipelineState) -> str:
    """Directory that holds per-run checkpoint folders (*-<dataset>-<config>)."""
    ckpt_dir = os.path.abspath(state.checkpoint_dir)
    if os.path.basename(ckpt_dir) == "ckpts":
        return ckpt_dir
    return os.path.dirname(ckpt_dir)


def _discover_phase1_source_dir(state: PipelineState) -> Optional[str]:
    """Newest *-<dataset>-binary_dual_scale dir under ckpts/ with diff_hp.json."""
    ckpt_root = _phase1_ckpt_root(state)
    dataset = state.subset_id or state.dataset
    suffix = f"-{dataset}-binary_dual_scale"
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


def _phase1_source_dir(state: PipelineState, override: Optional[str] = None) -> Optional[str]:
    value = override or state.extra.get("phase1_source_dir")
    if value:
        dataset = state.subset_id or state.dataset
        value = value.format(dataset=dataset, subset_id=dataset)
        path = os.path.abspath(os.path.expanduser(value))
        if os.path.isdir(path):
            return path
        logger.warning("phase1_source_dir missing (%s); auto-discovering", path)
    discovered = _discover_phase1_source_dir(state)
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
    raise FileNotFoundError(
        f"Staged pretrain requires Phase 1 diff_hp.json for {state.dataset!r}. "
        f"Expected *-{state.subset_id or state.dataset}-binary_dual_scale under "
        f"{_phase1_ckpt_root(state)} or set phase1_source_dir."
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

    def should_skip(self, state: PipelineState) -> bool:
        coarse = _stage_pretrain_ckpt(state, "coarse")
        fine = _stage_pretrain_ckpt(state, "fine")
        if os.path.exists(coarse) and os.path.exists(fine):
            logger.info("  [%s] cached: %s / %s", self.name, coarse, fine)
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

        source_dir = _phase1_source_dir(state, self.get("phase1_source_dir"))
        best_params = _resolve_diff_hp(state, source_dir)
        itrans_ckpt = _resolve_itrans_pretrain(state, source_dir)

        n_samples = int(self.get("n_samples", SYNTHETIC_SAMPLES_DIFF_TUNE))
        epochs = int(self.get("epochs", PRETRAIN_DIFFUSION_EPOCHS))
        patience = int(self.get("patience", DIFFUSION_HP_PATIENCE))
        if state.smoke_test:
            n_samples = min(n_samples, 32)
            epochs = 1
            patience = 1

        for stage in ("coarse", "fine"):
            ckpt = _stage_pretrain_ckpt(state, stage)
            if os.path.exists(ckpt):
                logger.info("  [%s] %s cached: %s", self.name, stage, ckpt)
            else:
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
