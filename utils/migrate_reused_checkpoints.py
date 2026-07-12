#!/usr/bin/env python3
"""Find compatible checkpoints on cluster storage and copy into $SCRATCH/ts-sandbox/reused/."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml

from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _candidate_phase1_ckpt_roots,
    _run_dir_matches_config,
    _run_dir_matches_config_any_dataset,
)
from models.diffusion_tsf.pipeline.reused_paths import (
    reused_binary_staged_root,
    reused_guidance_ckpt,
    reused_mmpd_campaign_root,
    reused_pretrain_ckpt,
    reused_root,
    reused_stage_best_ckpt,
    reused_tuned_params_meta,
)
from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

MMPD_CAMPAIGN_GLOBS = (
    "*mmpd-decoder-paper-lb336-hz96-subset*",
    "*mmpd-subset-lb336-hz96*",
    "*mmpd-decoder-paper-lb336-hz720*",
    "*mmpd-decoder-ordinal-norm-lb336-hz720*",
)

DEFAULT_MMPD_LB336_HZ96_DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "illness",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "PeMS",
    "solar_Alabama",
    "dynamic",
)

BINARY_GRID_LB336_HZ720_ORDINAL_NORM = (
    "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm"
)
MMPD_GRID_LB336_HZ720_ORDINAL_NORM = (
    "mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm"
)
GRID_LB336_HZ720_ORDINAL_FOUR_DATASETS = (
    "ETTh1",
    "traffic",
    "electricity",
    "exchange_rate",
)
# Jobs 4208596–4208599 (past-native stride-2 + per-dataset CRPS g).
GRID_LB336_HZ720_PAST_NATIVE_FOUR_BINARY_STEMS = {
    "ETTh1": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native",
    "traffic": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5",
    "electricity": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0",
    "exchange_rate": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0",
}
MMPD_GRID_LB336_HZ720_ORDINAL_FOUR_GLOBS = (
    "*mmpd-decoder-ordinal-norm-lb336-hz720*",
    "*mmpd-decoder-paper-lb336-hz720-subset*",
    "*mmpd-decoder-paper-lb336-hz720*",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _newest_matching_run(
    roots: Iterable[str],
    *,
    dataset: Optional[str],
    config_suffix: str,
    required_file: Optional[str] = None,
) -> Optional[str]:
    best_dir: Optional[str] = None
    best_mtime = 0.0
    for ckpt_root in roots:
        try:
            names = os.listdir(ckpt_root)
        except OSError:
            continue
        for name in names:
            if dataset is not None:
                if not _run_dir_matches_config(name, dataset, config_suffix):
                    continue
            elif not _run_dir_matches_config_any_dataset(name, config_suffix):
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


def _copy_file(src: str, dst: str, *, dry_run: bool) -> None:
    if not os.path.isfile(src):
        return
    if dry_run:
        logger.info("[dry-run] would copy %s -> %s", src, dst)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("copied %s -> %s", src, dst)


def _copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.exists():
        return
    if dry_run:
        logger.info("[dry-run] would copytree %s -> %s", src, dst)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    logger.info("copied tree %s -> %s", src, dst)


def _candidate_mmpd_campaign_dirs(
    campaign_globs: Sequence[str] = MMPD_CAMPAIGN_GLOBS,
) -> List[Path]:
    bases: List[Path] = []
    repo = _repo_root()
    bases.append(repo / "results" / "datasets")
    scratch = os.environ.get("SCRATCH")
    if scratch:
        bases.append(Path(scratch) / "ts-sandbox" / "results" / "datasets")
    submit = os.environ.get("SLURM_SUBMIT_DIR")
    if submit:
        bases.append(Path(submit) / "results" / "datasets")

    campaigns: List[Path] = []
    for base in bases:
        if not base.is_dir():
            continue
        for pattern in campaign_globs:
            campaigns.extend(p for p in base.glob(pattern) if p.is_dir())
    uniq = {str(p.resolve()): p for p in campaigns}
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _find_mmpd_ckpt_dir(
    campaign: Path,
    data_names: Sequence[str],
    *,
    backbone: str = "Decoder",
) -> Optional[Path]:
    base = campaign / "mmpd_out" / "checkpoints" / f"{backbone}-MMPD"
    if not base.is_dir():
        return None
    prefixes = [f"data{name}_" for name in data_names]
    best: Optional[tuple[float, Path]] = None
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if not any(d.name.startswith(pref) for pref in prefixes):
            continue
        ckpt = d / "model_checkpoint.pth"
        if not ckpt.is_file():
            continue
        mt = ckpt.stat().st_mtime
        if best is None or mt > best[0]:
            best = (mt, d)
    return best[1] if best else None


def _load_mmpd_datasets(config_suffix: str) -> List[str]:
    path = _repo_root() / "configs" / f"{config_suffix}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"MMPD config not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    datasets = (raw.get("mmpd") or {}).get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{path} missing mmpd.datasets list")
    return [str(d) for d in datasets]


def migrate_binary_config(
    *,
    config_suffix: str,
    dataset: str,
    subset_id: Optional[str],
    stages: List[str],
    dry_run: bool,
) -> None:
    state = PipelineState(dataset=dataset, checkpoint_dir="./results/ckpts")
    roots = _candidate_phase1_ckpt_roots(state)
    sid = subset_id or dataset

    for stage in stages:
        rel = f"pretrained_{stage}/pretrained_diffusion.pt"
        run_dir = _newest_matching_run(
            roots, dataset=None, config_suffix=config_suffix, required_file=rel,
        )
        if run_dir:
            src = os.path.join(run_dir, rel)
            dst = reused_pretrain_ckpt(config_suffix, stage)
            _copy_file(src, dst, dry_run=dry_run)

    guidance_name = f"{sid}_patch_guidance.pt"
    guidance_run = _newest_matching_run(
        roots, dataset=dataset, config_suffix=config_suffix, required_file=guidance_name,
    )
    if guidance_run:
        _copy_file(
            os.path.join(guidance_run, guidance_name),
            reused_guidance_ckpt(config_suffix, sid),
            dry_run=dry_run,
        )

    tuned_run = None
    for stage in stages:
        meta_rel = os.path.join(sid, stage, "metadata.json")
        tuned_run = _newest_matching_run(
            roots, dataset=dataset, config_suffix=config_suffix, required_file=meta_rel,
        )
        if not tuned_run:
            continue
        src_meta = os.path.join(tuned_run, meta_rel)
        dst_meta = reused_tuned_params_meta(config_suffix, sid, stage)
        _copy_file(src_meta, dst_meta, dry_run=dry_run)
        src_best = os.path.join(tuned_run, sid, stage, "best.pt")
        _copy_file(src_best, reused_stage_best_ckpt(config_suffix, sid, stage), dry_run=dry_run)

    if tuned_run:
        src_subset = os.path.join(tuned_run, sid)
        dst_subset = os.path.join(reused_binary_staged_root(config_suffix), sid)
        if os.path.isdir(src_subset):
            _copy_tree(Path(src_subset), Path(dst_subset), dry_run=dry_run)
        guidance_src = os.path.join(tuned_run, guidance_name)
        if os.path.isfile(guidance_src):
            dst_guidance = os.path.join(reused_binary_staged_root(config_suffix), guidance_name)
            _copy_file(guidance_src, dst_guidance, dry_run=dry_run)
    elif not dry_run:
        logger.warning(
            "  [binary] %s: no staged coarse/fine ckpt for config %s",
            dataset,
            config_suffix,
        )


def migrate_mmpd_campaign(
    *,
    config_suffix: str,
    datasets: Sequence[str],
    dry_run: bool,
    campaign_globs: Sequence[str] = MMPD_CAMPAIGN_GLOBS,
) -> None:
    from utils.eval_mmpd_gaussian_anchor import build_anchor_runs_from_subset_config
    from utils.mmpd_run_config import resolve_subset_config_path

    cfg_path = _repo_root() / "configs" / f"{config_suffix}.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    mmpd_block = raw.get("mmpd") or {}
    subset_cfg = resolve_subset_config_path(str(mmpd_block["subset_config"]))
    seed = int(raw.get("experiment", {}).get("seed", 2026))

    dst_root = Path(reused_mmpd_campaign_root(config_suffix))
    campaigns = _candidate_mmpd_campaign_dirs(campaign_globs)
    if not campaigns:
        raise FileNotFoundError(
            f"No MMPD campaign dirs matching {list(campaign_globs)} under results/datasets"
        )

    migrated = 0
    source_campaigns: dict[str, str] = {}
    for dataset in datasets:
        runs = build_anchor_runs_from_subset_config(subset_cfg, [dataset], seed)
        run = runs[dataset]
        from utils.eval_mmpd_gaussian_anchor import mmpd_checkpoint_data_names

        data_names = mmpd_checkpoint_data_names(run)
        src_dir = None
        src_campaign = None
        best_mt = 0.0
        for camp in campaigns:
            found = _find_mmpd_ckpt_dir(camp, data_names)
            if found is None:
                continue
            mt = (found / "model_checkpoint.pth").stat().st_mtime
            if src_dir is None or mt > best_mt:
                best_mt = mt
                src_dir = found
                src_campaign = camp
        if src_dir is None:
            logger.warning("  [mmpd] %s: no checkpoint found in campaigns", dataset)
            continue

        rel = src_dir.relative_to(src_campaign / "mmpd_out" / "checkpoints")
        dst_ckpt_dir = dst_root / "mmpd_out" / "checkpoints" / rel
        _copy_tree(src_dir, dst_ckpt_dir, dry_run=dry_run)

        for name in ("indices", "raw"):
            patterns = (
                f"indices_{dataset}.json",
                f"indices_{dataset}_mmpd_eval.json",
                f"indices_{data_names[0]}.json",
                f"indices_{data_names[0]}_mmpd_eval.json",
                f"mmpd_{dataset}.npz",
            )
            for pattern in patterns:
                src_idx = src_campaign / name / pattern
                if src_idx.is_file():
                    dst_idx = dst_root / name / pattern
                    _copy_file(str(src_idx), str(dst_idx), dry_run=dry_run)

        source_campaigns[dataset] = str(src_campaign.resolve())
        migrated += 1
        logger.info("  [mmpd] %s: from %s -> %s", dataset, src_dir, dst_ckpt_dir)

    manifest = {
        "config_suffix": config_suffix,
        "datasets": list(datasets),
        "migrated": migrated,
        "source_campaigns": source_campaigns,
        "reused_root": str(dst_root),
    }
    manifest_path = dst_root / "manifest.json"
    if dry_run:
        logger.info("[dry-run] would write %s", manifest_path)
    else:
        dst_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("wrote %s (%d datasets)", manifest_path, migrated)


def migrate_grid_lb336_hz720_ordinal_four(*, dry_run: bool) -> None:
    """Binary grid jobs 4208596–4208599 + matching MMPD ordinal-norm lb336/hz720 ckpts."""
    for dataset in GRID_LB336_HZ720_ORDINAL_FOUR_DATASETS:
        config_suffix = GRID_LB336_HZ720_PAST_NATIVE_FOUR_BINARY_STEMS[dataset]
        migrate_binary_config(
            config_suffix=config_suffix,
            dataset=dataset,
            subset_id=None,
            stages=["coarse", "fine"],
            dry_run=dry_run,
        )
    migrate_mmpd_campaign(
        config_suffix=MMPD_GRID_LB336_HZ720_ORDINAL_NORM,
        datasets=GRID_LB336_HZ720_ORDINAL_FOUR_DATASETS,
        dry_run=dry_run,
        campaign_globs=MMPD_GRID_LB336_HZ720_ORDINAL_FOUR_GLOBS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-suffix",
        help="Binary config stem for results/ckpts reuse, e.g. binary_anchor_ar_lb336_hz96_grad_accum_150",
    )
    parser.add_argument(
        "--mmpd-config-suffix",
        help="MMPD run config stem, e.g. mmpd_decoder_flat_subsets_paper_lb336_hz96",
    )
    parser.add_argument("--dataset", default=None, help="Single dataset (binary migrate)")
    parser.add_argument("--subset-id", default=None)
    parser.add_argument(
        "--stages",
        default="coarse,fine",
        help="Comma-separated diffusion stages for binary migrate",
    )
    parser.add_argument(
        "--migrate-mmpd",
        action="store_true",
        help="Migrate MMPD campaign checkpoints (uses all datasets from mmpd YAML unless --datasets)",
    )
    parser.add_argument(
        "--migrate-mmpd-lb336-hz96",
        action="store_true",
        help="Shorthand: --migrate-mmpd --mmpd-config-suffix mmpd_decoder_flat_subsets_paper_lb336_hz96",
    )
    parser.add_argument(
        "--migrate-grid-lb336-hz720-ordinal-four",
        action="store_true",
        help="Migrate binary+ MMPD ckpts for ETTh1,traffic,electricity,exchange_rate "
        f"(binary={BINARY_GRID_LB336_HZ720_ORDINAL_NORM}, "
        f"mmpd={MMPD_GRID_LB336_HZ720_ORDINAL_NORM})",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets (MMPD migrate default: mmpd YAML list; "
        "binary migrate: loops all when set with --config-suffix)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("reused root: %s", reused_root())

    if args.migrate_mmpd_lb336_hz96:
        args.migrate_mmpd = True
        args.mmpd_config_suffix = "mmpd_decoder_flat_subsets_paper_lb336_hz96"

    if args.migrate_grid_lb336_hz720_ordinal_four:
        migrate_grid_lb336_hz720_ordinal_four(dry_run=args.dry_run)
        return

    if args.migrate_mmpd:
        suffix = args.mmpd_config_suffix
        if not suffix:
            raise ValueError("--migrate-mmpd requires --mmpd-config-suffix")
        if args.datasets:
            datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        else:
            datasets = _load_mmpd_datasets(suffix)
        migrate_mmpd_campaign(
            config_suffix=suffix,
            datasets=datasets,
            dry_run=args.dry_run,
        )

    if args.config_suffix:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        if args.datasets:
            binary_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        elif args.dataset:
            binary_datasets = [args.dataset]
        else:
            binary_datasets = ["ETTh1"]
        for dataset in binary_datasets:
            migrate_binary_config(
                config_suffix=args.config_suffix,
                dataset=dataset,
                subset_id=args.subset_id,
                stages=stages,
                dry_run=args.dry_run,
            )

    if not args.migrate_mmpd and not args.config_suffix:
        parser.error("pass --config-suffix and/or --migrate-mmpd")


if __name__ == "__main__":
    main()
