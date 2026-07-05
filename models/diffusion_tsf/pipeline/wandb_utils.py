"""Wandb helpers for pipeline runs.

Each pipeline execution creates one wandb run (named after the run stem).
Phases log into that run with prefixed metrics (hp/*, eval/*, viz/*).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PIPELINE_JOB_TYPE = "pipeline"


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


def make_pipeline_run_name(group: str) -> str:
    """Build wandb run title for a full pipeline (same as run stem / group)."""
    if len(group) <= WANDB_RUN_NAME_MAX_LEN:
        return group
    return group[:WANDB_RUN_NAME_MAX_LEN]


def make_phase_run_name(group: str, phase_slug: str) -> str:
    """Build legacy per-phase wandb run title: {group}-{phase}."""
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

WANDB_MANIFEST_VERSION = 2
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


def migrate_manifest_run_id(manifest: Dict[str, Any]) -> Optional[str]:
    """Resolve pipeline run_id from v2 manifest or legacy phase_runs map."""
    run_id = manifest.get("run_id")
    if run_id:
        return str(run_id)
    phase_runs = dict(manifest.get("phase_runs") or {})
    if not phase_runs:
        return None
    if "staged_eval" in phase_runs:
        return str(phase_runs["staged_eval"])
    return str(next(iter(phase_runs.values())))


def pipeline_has_eval_phase(phase_names: Iterable[str]) -> bool:
    return any(name in EVAL_PHASE_NAMES for name in phase_names)


def is_binary_eval_run(run) -> bool:
    """True for legacy staged_eval runs or unified pipeline runs with eval metrics."""
    job_type = getattr(run, "job_type", None)
    if job_type == "staged_eval":
        return True
    if job_type == PIPELINE_JOB_TYPE:
        summary = getattr(run, "summary", None) or {}
        try:
            return summary.get("eval/staged_crps") is not None
        except Exception:
            return False
    return False


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
        run_id = migrate_manifest_run_id(manifest)
        state.wandb_run_id = run_id
        if run_id and manifest.get("run_id") != run_id:
            manifest["run_id"] = run_id
            manifest.pop("phase_runs", None)
            save_manifest(state.checkpoint_dir, manifest)
        return manifest

    state.wandb_run_id = None
    state.wandb_group = infer_wandb_group(state, merged_config)

    manifest = {
        "project": state.wandb_project,
        "group": state.wandb_group,
        "tags": state.wandb_tags,
        "config_yaml": yaml_path,
        "run_id": None,
    }
    save_manifest(state.checkpoint_dir, manifest)
    return manifest


def record_pipeline_run_id(
    checkpoint_dir: str,
    run_id: str,
    manifest: Dict[str, Any],
) -> None:
    manifest["run_id"] = run_id
    manifest.pop("phase_runs", None)
    save_manifest(checkpoint_dir, manifest)


def build_pipeline_tags(
    *,
    dataset: str,
    phase_names: Iterable[str],
    extra_tags: Optional[list] = None,
) -> list:
    """Dataset tag on every run; ``eval`` when the pipeline includes an eval phase."""
    tags = [dataset]
    if extra_tags:
        tags.extend(extra_tags)
    if pipeline_has_eval_phase(phase_names) and "eval" not in tags:
        tags.append("eval")
    return tags


def build_run_tags(
    *,
    dataset: str,
    phase_name: str,
    extra_tags: Optional[list] = None,
) -> list:
    """Legacy per-phase tag builder (stub scripts)."""
    tags = [dataset]
    if extra_tags:
        tags.extend(extra_tags)
    if phase_name in EVAL_PHASE_NAMES and "eval" not in tags:
        tags.append("eval")
    return tags


def init_pipeline_run(
    group: str,
    project: str,
    config: Dict[str, Any],
    tags: Optional[list] = None,
    yaml_path: Optional[str] = None,
    run_id: Optional[str] = None,
    job_type: str = PIPELINE_JOB_TYPE,
) -> Optional[Any]:
    """Start or resume the single wandb run for a full pipeline."""
    if not _WANDB_AVAILABLE or not _api_key_usable():
        return None

    run_name = make_pipeline_run_name(group)
    try:
        init_kwargs: Dict[str, Any] = {
            "project": project,
            "group": group,
            "job_type": job_type,
            "name": run_name,
            "tags": tags or [],
        }
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
        logger.info("wandb pipeline run %s: %s", action, run.url)
        return run
    except Exception as e:
        logger.warning("Failed to init wandb pipeline run: %s", e)
        return None


def begin_phase(phase_config: Dict[str, Any]) -> None:
    """Merge phase-specific config into the active pipeline run."""
    if not phase_config:
        return
    merge_run_config(phase_config)


def finish_pipeline_run() -> None:
    """Finish the pipeline wandb run."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def finish_phase_run() -> None:
    """Alias for finish_pipeline_run (legacy call sites)."""
    finish_pipeline_run()


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


def _scalar_wandb_entries(data: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    """Keep only wandb-summary-safe scalar values."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, (int, float, bool)):
            out[full_key] = value
        elif isinstance(value, str) and len(value) <= 512:
            out[full_key] = value
    return out


def log_phase_diagnostics_result(
    result: Optional[Dict[str, Any]],
    *,
    summary_prefix: str = "",
) -> None:
    """Log summary scalars, config metadata, and viz paths from a diagnostics bundle."""
    if not result:
        return
    summary = result.get("summary") or {}
    if summary:
        scalars = _scalar_wandb_entries(summary, prefix=summary_prefix)
        config_text = {
            k: v for k, v in summary.items()
            if isinstance(v, str) and (k.startswith("architecture/") or k.startswith("loss/"))
        }
        if config_text:
            merge_run_config(config_text)
        if scalars:
            log_summary(scalars)
    config = result.get("config")
    if config:
        merge_run_config(config)
    for key, paths in (result.get("viz") or {}).items():
        log_visualization_paths(paths, wandb_key=key)
