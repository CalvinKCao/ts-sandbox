"""Fail-fast MMPD point-gap / redbox campaign checks (no torch).

Import-safe on a login node. Used by ``submit_binary.sh`` and pipeline start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_mmpd_campaign_root(viz_cfg: Mapping[str, Any], dataset: str) -> Optional[Path]:
    """Return campaign dir or None when unset for this dataset."""
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


def mmpd_pack_path(campaign: Path, dataset: str) -> Path:
    return Path(campaign) / "raw" / f"mmpd_{dataset}.npz"


def validate_mmpd_viz_requirements(
    viz_cfg: Mapping[str, Any],
    dataset: str,
    *,
    repo_root: Path,
) -> Optional[Path]:
    """If gap/redbox viz is on and a campaign is configured, the pack must exist.

    Unset campaign → skip (returns None). Set but missing pack → FileNotFoundError.
    """
    do_gap = bool(viz_cfg.get("viz_binary_mmpd_gap", True))
    do_redbox = bool(viz_cfg.get("viz_binary_mmpd_redbox", True))
    if not do_gap and not do_redbox:
        return None
    campaign = resolve_mmpd_campaign_root(viz_cfg, dataset)
    if campaign is None:
        return None
    if not campaign.is_absolute():
        campaign = (repo_root / campaign).resolve()
    pack = mmpd_pack_path(campaign, dataset)
    if not pack.is_file():
        raise FileNotFoundError(
            f"{dataset}: MMPD gap/redbox viz is enabled but pack is missing: {pack} "
            f"(campaign={campaign}). Set visualization.mmpd_campaign_root_by_dataset."
            f"{dataset} to a campaign that contains this dataset, or disable "
            "viz_binary_mmpd_gap / viz_binary_mmpd_redbox."
        )
    return pack


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Preflight MMPD gap/redbox packs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    from models.diffusion_tsf.pipeline.config import (
        load_experiment_config,
        visualization_settings,
    )

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": args.dataset})
    pack = validate_mmpd_viz_requirements(
        visualization_settings(cfg),
        args.dataset,
        repo_root=args.repo_root.resolve(),
    )
    if pack is None:
        print(f"{args.dataset}: MMPD gap/redbox viz skipped (no campaign for this dataset)")
    else:
        print(f"{args.dataset}: MMPD pack ok ({pack})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
