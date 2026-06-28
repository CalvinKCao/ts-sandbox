"""Wandb helpers for grouped pipeline runs.

Each pipeline execution creates a wandb *group*. Each phase within that
pipeline creates its own wandb *run* inside the group. This gives a clean
dashboard where you can expand a group to see per-phase metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _ensure_dotenv_loaded() -> None:
    try:
        from utils.load_dotenv import load_repo_dotenv
    except ImportError:
        return
    load_repo_dotenv(_REPO_ROOT)


_ensure_dotenv_loaded()

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


def _api_key_usable() -> bool:
    try:
        from utils.repo_env import load_repo_env
    except ImportError:
        load_repo_env = None
    if load_repo_env is not None:
        load_repo_env()
    key = os.environ.get("WANDB_API_KEY", "").strip()
    return bool(key and re.fullmatch(r"[A-Za-z0-9_]+", key))


_RUN_STEM_RE = re.compile(r"^\d{2}-\d{2}-\d+-.+")

WANDB_RUN_NAME_MAX_LEN = 128


def run_stem_from_checkpoint_dir(checkpoint_dir: str) -> str:
    """Basename of the isolated checkpoint dir (same as Slurm log/ckpt stem)."""
    return os.path.basename(os.path.abspath(checkpoint_dir))


def _looks_like_run_stem(stem: str) -> bool:
    return bool(stem and _RUN_STEM_RE.match(stem))


def make_local_group_name(
    seed: int,
    *,
    config_slug: Optional[str] = None,
    experiment_name: str = "experiment",
) -> str:
    """Fallback wandb group for local runs without a Slurm-style checkpoint stem."""
    date_str = datetime.now().strftime("%m-%d")
    slug = config_slug or experiment_name
    return f"{date_str}-{slug}-s{seed}"


def infer_wandb_group(
    state: "PipelineState",
    merged_config: Dict[str, Any],
) -> str:
    """Resolve wandb group: YAML override, else checkpoint run stem, else local fallback."""
    if state.wandb_group:
        return state.wandb_group

    ckpt_stem = run_stem_from_checkpoint_dir(state.checkpoint_dir)
    env_stem = os.environ.get("GRID_RUN_STEM", "").strip()
    if env_stem and env_stem != ckpt_stem:
        logger.warning(
            "GRID_RUN_STEM=%r differs from checkpoint_dir stem %r; using checkpoint stem",
            env_stem,
            ckpt_stem,
        )
    if _looks_like_run_stem(ckpt_stem):
        return ckpt_stem

    yaml_path = merged_config.get("_yaml_path")
    return make_local_group_name(
        state.seed,
        config_slug=config_slug_from_yaml(yaml_path),
        experiment_name=state.experiment_name,
    )


def make_phase_run_name(group: str, phase_slug: str) -> str:
    """Build wandb run title: {group}-{phase}."""
    phase_slug = phase_slug.replace("_", "-")
    full = f"{group}-{phase_slug}"
    if len(full) <= WANDB_RUN_NAME_MAX_LEN:
        return full
    return full[:WANDB_RUN_NAME_MAX_LEN]


def config_slug_from_yaml(yaml_path: Optional[str]) -> Optional[str]:
    if not yaml_path:
        return None
    return os.path.splitext(os.path.basename(yaml_path))[0]


EVAL_PHASE_NAMES = frozenset({"eval", "staged_eval"})

WANDB_MANIFEST_VERSION = 1
WANDB_MANIFEST_FILENAME = "wandb_manifest.json"


def manifest_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, WANDB_MANIFEST_FILENAME)


def load_manifest(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    path = manifest_path(checkpoint_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def save_manifest(checkpoint_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = manifest_path(checkpoint_dir)
    tmp = f"{path}.tmp"
    payload = dict(manifest)
    payload["version"] = WANDB_MANIFEST_VERSION
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def delete_manifest(checkpoint_dir: str) -> None:
    path = manifest_path(checkpoint_dir)
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def resolve_wandb_settings(
    state: "PipelineState",
    merged_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Load or create the checkpoint-local wandb manifest (YAML is SSOT on first run)."""
    from models.diffusion_tsf.pipeline.config import wandb_settings

    if not state.wandb_enabled:
        return {}

    if state.fresh:
        delete_manifest(state.checkpoint_dir)

    wandb_cfg = wandb_settings(merged_config)
    yaml_path = merged_config.get("_yaml_path")
    manifest = load_manifest(state.checkpoint_dir)

    if state.resume and manifest is None:
        logger.warning(
            "resume=True but no %s under %s; wandb runs will be new",
            WANDB_MANIFEST_FILENAME,
            state.checkpoint_dir,
        )

    if manifest:
        yaml_project = str(wandb_cfg.get("project") or state.wandb_project)
        yaml_group = wandb_cfg.get("group")
        if yaml_project != manifest.get("project"):
            logger.warning(
                "wandb project YAML=%r differs from manifest=%r; using manifest on resume",
                yaml_project,
                manifest.get("project"),
            )
        if yaml_group and yaml_group != manifest.get("group"):
            logger.warning(
                "wandb group YAML=%r differs from manifest=%r; using manifest on resume",
                yaml_group,
                manifest.get("group"),
            )
        state.wandb_project = str(manifest.get("project") or state.wandb_project)
        state.wandb_group = manifest.get("group") or state.wandb_group
        manifest_tags = manifest.get("tags")
        if manifest_tags is not None:
            state.wandb_tags = list(manifest_tags)
        state.wandb_phase_run_ids = dict(manifest.get("phase_runs") or {})
        return manifest

    state.wandb_phase_run_ids = {}
    state.wandb_group = infer_wandb_group(state, merged_config)

    manifest = {
        "project": state.wandb_project,
        "group": state.wandb_group,
        "tags": state.wandb_tags,
        "config_yaml": yaml_path,
        "phase_runs": {},
    }
    save_manifest(state.checkpoint_dir, manifest)
    return manifest


def record_phase_run_id(
    checkpoint_dir: str,
    phase_name: str,
    run_id: str,
    manifest: Dict[str, Any],
) -> None:
    phase_runs = dict(manifest.get("phase_runs") or {})
    phase_runs[phase_name] = run_id
    manifest["phase_runs"] = phase_runs
    save_manifest(checkpoint_dir, manifest)


def build_run_tags(
    *,
    dataset: str,
    phase_name: str,
    extra_tags: Optional[list] = None,
) -> list:
    """Dataset tag on every run; ``eval`` only on eval phases unless overridden."""
    tags = [dataset]
    if extra_tags:
        tags.extend(extra_tags)
    elif phase_name in EVAL_PHASE_NAMES:
        tags.append("eval")
    return tags


def init_phase_run(
    phase_slug: str,
    group: str,
    project: str,
    job_type: str,
    config: Dict[str, Any],
    tags: Optional[list] = None,
    yaml_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[Any]:
    """Start or resume a wandb run for one pipeline phase.

    Returns the run object (or None if wandb is unavailable/disabled).
    """
    if not _WANDB_AVAILABLE or not _api_key_usable():
        return None

    run_name = make_phase_run_name(group, phase_slug)
    full_name = f"{group}-{phase_slug.replace('_', '-')}"
    try:
        init_kwargs: Dict[str, Any] = {
            "project": project,
            "group": group,
            "job_type": job_type,
            "name": run_name,
            "reinit": True,
            "tags": tags or [],
        }
        if run_name != full_name:
            init_kwargs["notes"] = full_name
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        else:
            init_kwargs["config"] = config

        run = wandb.init(**init_kwargs)
        if not run_id and yaml_path and os.path.isfile(yaml_path):
            artifact = wandb.Artifact("experiment-yaml", type="config")
            artifact.add_file(yaml_path)
            run.log_artifact(artifact)
        action = "resumed" if run_id else "started"
        logger.info("wandb run %s: %s", action, run.url)
        return run
    except Exception as e:
        logger.warning(f"Failed to init wandb run for {phase_slug}: {e}")
        return None


def finish_phase_run() -> None:
    """Finish the current wandb run (call at end of each phase)."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def log_summary(metrics: Dict[str, Any]) -> None:
    """Set summary metrics on the current run."""
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    for k, v in metrics.items():
        wandb.run.summary[k] = v


def merge_run_config(updates: Dict[str, Any]) -> None:
    """Merge keys into the active wandb run config (e.g. diagnostic metadata)."""
    if not _WANDB_AVAILABLE or wandb.run is None or not updates:
        return
    wandb.config.update(updates, allow_val_change=True)


def log_eval_metrics(metrics: Dict[str, Any], step: int = 0) -> None:
    """Log eval metrics to the run history and summary (eval phase)."""
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    clean = {k: v for k, v in metrics.items() if v is not None}
    if not clean:
        return
    wandb.log(clean, step=step)
    log_summary(clean)


def log_visualization_paths(
    paths: list,
    wandb_key: str = "visualizations",
    caption_prefix: str = "",
) -> None:
    """Log JPEG/PNG artifacts to wandb and print a line per file."""
    if not paths:
        return
    if not _WANDB_AVAILABLE or wandb.run is None:
        for p in paths:
            logger.info("visualization %s generated!", p)
        return
    images = []
    for p in sorted(paths):
        logger.info("visualization %s generated!", p)
        cap = os.path.basename(p)
        if caption_prefix:
            cap = f"{caption_prefix}/{cap}"
        images.append(wandb.Image(p, caption=cap))
    wandb.log({wandb_key: images})
