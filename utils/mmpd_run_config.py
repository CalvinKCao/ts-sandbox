"""Load ``mmpd:`` block from MMPD run YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_mmpd_run_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    block = cfg.get("mmpd")
    if not isinstance(block, dict):
        raise ValueError(f"{path} missing top-level mmpd: mapping")
    return dict(block)


def resolve_subset_config_path(subset_config: str, *, repo_root: Path = REPO_ROOT) -> Path:
    p = Path(subset_config)
    if p.is_file():
        return p.resolve()
    candidate = repo_root / "configs" / subset_config
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"subset config not found: {subset_config}")


def apply_mmpd_run_config(args: Any, block: Dict[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    """Apply YAML ``mmpd`` fields onto eval_mmpd_gaussian_anchor argparse namespace."""
    if backbone := block.get("backbone"):
        args.mmpd_backbone = str(backbone)
    if subset_config := block.get("subset_config"):
        args.subset_config = resolve_subset_config_path(str(subset_config), repo_root=repo_root)
        args.mmpd_only = True
    if lookback := block.get("lookback"):
        args.lookback = int(lookback)
    if horizon := block.get("horizon"):
        args.horizon = int(horizon)
    if train_epochs := block.get("train_epochs"):
        args.mmpd_train_epochs = int(train_epochs)
    if patience := block.get("patience"):
        args.mmpd_patience = int(patience)
    if batch_size := block.get("batch_size"):
        args.mmpd_batch_size = int(batch_size)
    if sample_num := block.get("sample_num"):
        args.sample_num = int(sample_num)
    if num_sampling_steps := block.get("num_sampling_steps"):
        args.num_sampling_steps = int(num_sampling_steps)
    if gmm_components := block.get("gmm_components"):
        args.gmm_components = int(gmm_components)
    if gmm_iterations := block.get("gmm_iterations"):
        args.gmm_iterations = int(gmm_iterations)
    if tune_trials := block.get("tune_trials"):
        args.mmpd_tune_trials = int(tune_trials)
    if tune_epochs := block.get("tune_epochs"):
        args.mmpd_tune_epochs = int(tune_epochs)
    if tune_patience := block.get("tune_patience"):
        args.mmpd_tune_patience = int(tune_patience)
    tune_params = block.get("tune_params")
    if isinstance(tune_params, dict):
        args.mmpd_tune_params = dict(tune_params)
