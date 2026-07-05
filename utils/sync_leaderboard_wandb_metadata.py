"""Backfill config_nickname on leaderboard wandb runs and create MMPD subset stubs."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.leaderboard_config_nicknames import (
    MMPD_DECODER_GRAD_ACCUM_200_LR_LO_JOBS,
    MMPD_DECODER_GRAD_ACCUM_200_LR_LO_NICKNAME,
    MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW,
    MMPD_MASKAE_FAIR_13D_JOBS,
    MMPD_MASKAE_FAIR_13D_NICKNAME,
    MMPD_SUBSET_JOBS,
    MMPD_SUBSET_NICKNAME,
    all_dataset_tag_tokens,
    leaderboard_dataset_tags,
    leaderboard_nickname,
    leaderboard_staged_eval_tags,
    load_mmpd_decoder_grad_accum_200_lr_lo_metrics,
    load_mmpd_fair_13d_metrics,
    load_mmpd_subset_metrics,
    mmpd_decoder_grad_accum_200_lr_lo_run_stem,
    mmpd_fair_13d_run_stem,
    mmpd_stub_wandb_metrics,
    mmpd_subset_run_stem,
    nickname_for_wandb_run,
    parse_run_stem,
)
from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from models.diffusion_tsf.pipeline.wandb_utils import make_phase_run_name, is_binary_eval_run, PIPELINE_JOB_TYPE

PROJECT = "ts-sandbox-leaderboard"
ENTITY = "calvincao"
MMPD_JOB_TYPE = "mmpd_eval"
EVAL_JOB_TYPES = frozenset({"staged_eval", "mmpd_eval", "classical_baseline"})
META_TAGS = frozenset({"eval", "curated-relog", "mmpd", "stub", "binary"})


def _run_path(run) -> str:
    entity = getattr(run, "entity", None) or ENTITY
    project = getattr(run, "project", None) or PROJECT
    return f"{entity}/{project}/{run.id}"


def backfill_nicknames(api, *, entity: str, project: str, dry_run: bool) -> dict:
    updated = skipped = 0
    for run in api.runs(f"{entity}/{project}"):
        nick = nickname_for_wandb_run(run)
        if not nick:
            skipped += 1
            continue
        current = (dict(run.config).get("config_nickname") or "").strip()
        if current == nick:
            skipped += 1
            continue
        if dry_run:
            print(f"would set config_nickname on {_run_path(run)}: {nick!r}")
            updated += 1
            continue
        run.config["config_nickname"] = nick
        run.update()
        updated += 1
        if updated % 50 == 0:
            print(f"nicknames updated {updated}...", flush=True)
    return {"updated": updated, "skipped": skipped}


def _dataset_for_run(run) -> str:
    parsed = parse_run_stem(run.group or "")
    if parsed:
        return parsed[1]
    cfg = dict(run.config)
    if cfg.get("dataset"):
        return str(cfg["dataset"])
    exp = cfg.get("experiment") or {}
    if isinstance(exp, dict) and exp.get("dataset"):
        return str(exp["dataset"])
    return ""


def _manifest_tags_for_group(group: str) -> list[str]:
    path = os.path.join(REPO, "results", "ckpts", group, "wandb_manifest.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("tags") or [])


def backfill_dataset_and_tags(api, *, entity: str, project: str, dry_run: bool) -> dict:
    updated = skipped = 0
    for run in api.runs(f"{entity}/{project}"):
        dataset = _dataset_for_run(run)
        if not dataset:
            skipped += 1
            continue

        if run.job_type == "staged_eval" or (
            run.job_type == PIPELINE_JOB_TYPE and is_binary_eval_run(run)
        ):
            manifest_tags = _manifest_tags_for_group(run.group or "")
            tags = leaderboard_staged_eval_tags(manifest_tags, dataset)
            for t in (run.tags or []):
                if t in ("curated-relog", "archive") and t not in tags:
                    tags.append(t)
        else:
            meta = [t for t in (run.tags or []) if t in META_TAGS]
            other = [
                t for t in (run.tags or [])
                if t not in META_TAGS and t not in all_dataset_tag_tokens()
            ]
            tags = leaderboard_dataset_tags(dataset) + meta + other
            if run.job_type in EVAL_JOB_TYPES and "eval" not in tags:
                tags.append("eval")
            tags = list(dict.fromkeys(tags))

        needs_cfg = dict(run.config).get("dataset") != dataset
        needs_tags = set(tags) != set(run.tags or [])
        if not needs_cfg and not needs_tags:
            skipped += 1
            continue

        if dry_run:
            print(f"would fix {_run_path(run)} dataset={dataset!r} tags={tags}")
            updated += 1
            continue

        if needs_cfg:
            run.config["dataset"] = dataset
        if needs_tags:
            run.tags = tags
        run.update()
        updated += 1
        if updated % 50 == 0:
            print(f"dataset/tags updated {updated}...", flush=True)
    return {"updated": updated, "skipped": skipped}


def _existing_mmpd_groups(api, entity: str, project: str) -> set[str]:
    groups: set[str] = set()
    for run in api.runs(f"{entity}/{project}"):
        group = run.group or ""
        if run.job_type == MMPD_JOB_TYPE:
            groups.add(group)
            continue
        if group.endswith("-mmpd_subset"):
            groups.add(group)
            continue
        if group.endswith(f"-{MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW}"):
            groups.add(group)
    return groups


def create_mmpd_stubs(api, *, entity: str, project: str, dry_run: bool) -> dict:
    import wandb

    existing = _existing_mmpd_groups(api, entity, project)
    created = skipped = 0

    for dataset, job_id in sorted(MMPD_SUBSET_JOBS.items()):
        metrics = load_mmpd_subset_metrics(dataset)
        if metrics is None:
            print(f"[skip] no partial for {dataset}")
            skipped += 1
            continue

        group = mmpd_subset_run_stem(dataset, job_id)
        if group in existing:
            print(f"[skip] mmpd stub exists: {group}")
            skipped += 1
            continue

        name = make_phase_run_name(group, MMPD_JOB_TYPE)
        summary = mmpd_stub_wandb_metrics(metrics)

        if dry_run:
            print(f"would create mmpd stub: {name} | {group} | {dataset}")
            created += 1
            continue

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=MMPD_JOB_TYPE,
            tags=[dataset, "eval", "mmpd", "stub"],
            notes="offline MMPD subset eval stub from JSON partial",
            config={
                "config_nickname": MMPD_SUBSET_NICKNAME,
                "dataset": dataset,
                "baseline": "mmpd_subset",
                "job_id": job_id,
                "metrics_source": metrics["source"],
                "partial_path": metrics["partial_path"],
                "stub": True,
            },
            settings=wandb.Settings(console="off"),
        )
        try:
            clean = {k: v for k, v in summary.items() if v is not None}
            wandb.log(clean, step=0)
            for k, v in clean.items():
                run.summary[k] = v
            print(f"created {_run_path(run)} | {name}")
            created += 1
        finally:
            wandb.finish()

    return {"created": created, "skipped": skipped}


def create_mmpd_fair_13d_stubs(api, *, entity: str, project: str, dry_run: bool) -> dict:
    import wandb

    existing = _existing_mmpd_groups(api, entity, project)
    created = skipped = 0

    for dataset, job_id in sorted(MMPD_MASKAE_FAIR_13D_JOBS.items()):
        metrics = load_mmpd_fair_13d_metrics(dataset)
        if metrics is None:
            print(f"[skip] no partial for {dataset}")
            skipped += 1
            continue

        group = mmpd_fair_13d_run_stem(dataset, job_id)
        if group in existing:
            print(f"[skip] mmpd fair-13d stub exists: {group}")
            skipped += 1
            continue

        name = make_phase_run_name(group, MMPD_JOB_TYPE)
        summary = mmpd_stub_wandb_metrics(metrics)
        config = {
            "config_nickname": MMPD_MASKAE_FAIR_13D_NICKNAME,
            "dataset": dataset,
            "baseline": "mmpd_maskae_fair_13d",
            "job_id": job_id,
            "metrics_source": metrics["source"],
            "partial_path": metrics["partial_path"],
            "stub": True,
        }
        if metrics.get("tuning_path"):
            config["tuning_path"] = metrics["tuning_path"]
        if metrics.get("tuned_hparams"):
            config["mmpd_tuned_hparams"] = metrics["tuned_hparams"]

        if dry_run:
            print(f"would create mmpd fair-13d stub: {name} | {group} | {dataset}")
            created += 1
            continue

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=MMPD_JOB_TYPE,
            tags=[dataset, "eval", "mmpd", "stub", "maskae", "fair-13d"],
            notes="offline MMPD MaskAE fair-13d eval from JSON partial (no artifacts)",
            config=config,
            settings=wandb.Settings(console="off"),
        )
        try:
            clean = {k: v for k, v in summary.items() if v is not None}
            wandb.log(clean, step=0)
            for k, v in clean.items():
                run.summary[k] = v
            print(f"created {_run_path(run)} | {name}")
            created += 1
        finally:
            wandb.finish()

    return {"created": created, "skipped": skipped}


def create_mmpd_decoder_grad_accum_stubs(api, *, entity: str, project: str, dry_run: bool) -> dict:
    import wandb

    existing = _existing_mmpd_groups(api, entity, project)
    created = skipped = 0

    for dataset, job_id in sorted(MMPD_DECODER_GRAD_ACCUM_200_LR_LO_JOBS.items()):
        metrics = load_mmpd_decoder_grad_accum_200_lr_lo_metrics(dataset)
        if metrics is None:
            print(f"[skip] no partial for {dataset}")
            skipped += 1
            continue

        group = mmpd_decoder_grad_accum_200_lr_lo_run_stem(dataset, job_id)
        if group in existing:
            print(f"[skip] mmpd decoder stub exists: {group}")
            skipped += 1
            continue

        name = make_phase_run_name(group, MMPD_JOB_TYPE)
        summary = mmpd_stub_wandb_metrics(metrics)
        config = {
            "config_nickname": MMPD_DECODER_GRAD_ACCUM_200_LR_LO_NICKNAME,
            "dataset": dataset,
            "baseline": "mmpd_decoder_flat_subsets_grad_accum_200_lr_lo",
            "job_id": job_id,
            "metrics_source": metrics["source"],
            "partial_path": metrics["partial_path"],
            "stub": True,
        }
        if metrics.get("tuning_path"):
            config["tuning_path"] = metrics["tuning_path"]
        if metrics.get("tuned_hparams"):
            config["mmpd_tuned_hparams"] = metrics["tuned_hparams"]

        if dry_run:
            print(f"would create mmpd decoder stub: {name} | {group} | {dataset}")
            created += 1
            continue

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=MMPD_JOB_TYPE,
            tags=[dataset, "eval", "mmpd", "stub", "decoder", "subset-tuned"],
            notes="offline MMPD Decoder subset-tuned eval from JSON partial (no artifacts)",
            config=config,
            settings=wandb.Settings(console="off"),
        )
        try:
            clean = {k: v for k, v in summary.items() if v is not None}
            wandb.log(clean, step=0)
            for k, v in clean.items():
                run.summary[k] = v
            print(f"created {_run_path(run)} | {name}")
            created += 1
        finally:
            wandb.finish()

    return {"created": created, "skipped": skipped}


def patch_mmpd_stub_metrics(api, *, entity: str, project: str, dry_run: bool) -> dict:
    """Fix metric keys on existing MMPD stub runs (eval/anchor_* -> eval/staged_*)."""
    import wandb

    patched = skipped = 0
    for run in api.runs(f"{entity}/{project}"):
        if run.job_type != MMPD_JOB_TYPE:
            continue
        dataset = None
        for tag in run.tags or []:
            if tag in MMPD_SUBSET_JOBS:
                dataset = tag
                break
        if dataset is None:
            for ds in MMPD_SUBSET_JOBS:
                if ds in (run.group or ""):
                    dataset = ds
                    break
        if dataset is None:
            skipped += 1
            continue

        metrics = load_mmpd_subset_metrics(dataset)
        if metrics is None:
            skipped += 1
            continue

        summary = mmpd_stub_wandb_metrics(metrics)
        current = dict(run.summary)
        if all(current.get(k) == v for k, v in summary.items()):
            skipped += 1
            continue

        if dry_run:
            print(f"would patch {_run_path(run)}: {list(summary.keys())}")
            patched += 1
            continue

        resumed = wandb.init(
            id=run.id,
            resume="must",
            project=project,
            entity=entity,
            settings=wandb.Settings(console="off"),
        )
        try:
            wandb.log(summary, step=0)
            for k, v in summary.items():
                resumed.summary[k] = v
            print(f"patched {_run_path(run)} | {dataset}")
            patched += 1
        finally:
            wandb.finish()

    return {"patched": patched, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--nicknames-only", action="store_true")
    parser.add_argument("--mmpd-only", action="store_true")
    parser.add_argument(
        "--fair-13d-only",
        action="store_true",
        help="Create mmpd_eval stubs for 06-16-mmpd-maskae-fair-13d only",
    )
    parser.add_argument(
        "--decoder-grad-accum-only",
        action="store_true",
        help="Create mmpd_eval stubs for 07-02-mmpd-decoder-grad-accum-200-lr-lo-subset only",
    )
    parser.add_argument("--patch-mmpd-metrics", action="store_true", help="Fix metric keys on existing MMPD stubs")
    parser.add_argument("--dataset-tags-only", action="store_true", help="Only backfill dataset config + tags")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    results = {}

    if args.fair_13d_only:
        print("Creating MMPD MaskAE fair-13d stubs...")
        results["mmpd_fair_13d"] = create_mmpd_fair_13d_stubs(
            api, entity=args.entity, project=args.project, dry_run=args.dry_run
        )
        print(results["mmpd_fair_13d"])
        print("done:", results)
        return

    if args.decoder_grad_accum_only:
        print("Creating MMPD Decoder grad-accum stubs...")
        results["mmpd_decoder_grad_accum"] = create_mmpd_decoder_grad_accum_stubs(
            api, entity=args.entity, project=args.project, dry_run=args.dry_run
        )
        print(results["mmpd_decoder_grad_accum"])
        print("done:", results)
        return

    if not args.mmpd_only and not args.dataset_tags_only:
        print("Backfilling config_nickname...")
        results["nicknames"] = backfill_nicknames(
            api, entity=args.entity, project=args.project, dry_run=args.dry_run
        )
        print(results["nicknames"])

    if not args.nicknames_only and not args.mmpd_only:
        print("Backfilling dataset + tags...")
        results["dataset_tags"] = backfill_dataset_and_tags(
            api, entity=args.entity, project=args.project, dry_run=args.dry_run
        )
        print(results["dataset_tags"])

    if not args.nicknames_only and not args.dataset_tags_only:
        if args.patch_mmpd_metrics:
            print("Patching MMPD stub metric keys...")
            results["mmpd_patch"] = patch_mmpd_stub_metrics(
                api, entity=args.entity, project=args.project, dry_run=args.dry_run
            )
            print(results["mmpd_patch"])
        else:
            print("Creating MMPD subset stubs...")
            results["mmpd"] = create_mmpd_stubs(
                api, entity=args.entity, project=args.project, dry_run=args.dry_run
            )
            print(results["mmpd"])

    print("done:", results)


if __name__ == "__main__":
    main()
