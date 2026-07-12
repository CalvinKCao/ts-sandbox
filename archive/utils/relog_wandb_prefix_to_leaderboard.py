"""Relog wandb runs matching a group prefix into ts-sandbox-leaderboard."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.diffusion_tsf.pipeline.wandb_utils import PIPELINE_JOB_TYPE, is_binary_eval_run
from utils.leaderboard_config_nicknames import leaderboard_nickname, parse_run_stem
from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

DST_PROJECT = "ts-sandbox-leaderboard"
DST_ENTITY = "calvincao"
STEM_GROUP_RE = re.compile(r"^\d{2}-\d{2}-\d+-")
EVAL_JOB_TYPES = frozenset({"staged_eval", "mmpd_eval", PIPELINE_JOB_TYPE})
DEFAULT_COPY_MAP = os.path.join(
    REPO, "archive", "reports", "sweep_grid_report", "curated_wandb_copy_map.json"
)


def _load_curate():
    for rel in (
        "archive/reports/curate_wandb_leaderboard_runs.py",
        "reports/curate_wandb_leaderboard_runs.py",
    ):
        path = os.path.join(REPO, rel)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("curate_wandb", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("curate_wandb_leaderboard_runs.py not found")


def _leaderboard_keys(api, entity: str, project: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    src_paths: Set[str] = set()
    group_job: Set[Tuple[str, str]] = set()
    for run in api.runs(f"{entity}/{project}"):
        notes = run.notes or ""
        for line in notes.splitlines():
            if line.startswith("relogged from "):
                url = line.split("relogged from ", 1)[1].strip()
                # path from url is awkward; skip
                break
        if run.group and run.job_type:
            group_job.add((run.group, run.job_type))
    return src_paths, group_job


def discover_prefix_runs(
    api,
    *,
    prefix: str,
    entity: str,
    dst_project: str,
) -> List[Dict[str, Any]]:
    _, lb_group_job = _leaderboard_keys(api, entity, dst_project)
    hits: List[Dict[str, Any]] = []
    seen_src: Set[str] = set()

    for proj in api.projects(entity):
        if proj.name == dst_project:
            continue
        try:
            runs = api.runs(f"{entity}/{proj.name}")
        except Exception:
            continue
        for run in runs:
            group = run.group or ""
            if not group.startswith(prefix):
                continue
            job_type = run.job_type or ""
            if (group, job_type) in lb_group_job:
                continue
            src_path = f"{entity}/{proj.name}/{run.id}"
            if src_path in seen_src:
                continue
            seen_src.add(src_path)

            parsed = parse_run_stem(group)
            if not parsed:
                continue
            _job_id, dataset, raw_config = parsed
            nick = leaderboard_nickname(raw_config=raw_config)
            hits.append(
                {
                    "id": run.id,
                    "entity": entity,
                    "project": proj.name,
                    "path": src_path,
                    "url": run.url,
                    "run_stem": group,
                    "dataset": dataset,
                    "raw_config": raw_config,
                    "config": f"**{nick}**" if nick else None,
                    "source": "prefix_scan",
                }
            )
    return hits


def discover_missing_eval_runs(
    api,
    *,
    entity: str,
    dst_project: str,
    dataset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find stem-format staged_eval runs in legacy projects not yet in leaderboard."""
    _, lb_group_job = _leaderboard_keys(api, entity, dst_project)
    hits: List[Dict[str, Any]] = []
    seen_src: Set[str] = set()

    for proj in api.projects(entity):
        if proj.name == dst_project:
            continue
        try:
            runs = api.runs(f"{entity}/{proj.name}")
        except Exception:
            continue
        for run in runs:
            if run.job_type not in EVAL_JOB_TYPES:
                continue
            if run.job_type == PIPELINE_JOB_TYPE and not is_binary_eval_run(run):
                continue
            group = run.group or ""
            if not STEM_GROUP_RE.match(group):
                continue
            parsed = parse_run_stem(group)
            if not parsed:
                continue
            _job_id, ds, raw_config = parsed
            if dataset and ds != dataset:
                continue
            if (group, run.job_type) in lb_group_job:
                continue
            src_path = f"{entity}/{proj.name}/{run.id}"
            if src_path in seen_src:
                continue
            seen_src.add(src_path)
            nick = leaderboard_nickname(raw_config=raw_config)
            hits.append(
                {
                    "id": run.id,
                    "entity": entity,
                    "project": proj.name,
                    "path": src_path,
                    "url": run.url,
                    "run_stem": group,
                    "dataset": ds,
                    "raw_config": raw_config,
                    "config": f"**{nick}**" if nick else None,
                    "source": "missing_eval_scan",
                }
            )
    return hits


def relog_hits(
    api,
    hits: List[Dict[str, Any]],
    *,
    dst_project: str,
    dst_entity: str,
    copy_map_path: str,
    dry_run: bool,
    summary_only: bool,
    skip_files: bool,
) -> Dict[str, Any]:
    curate = _load_curate()
    copy_map = curate.load_copy_map(copy_map_path)
    ok = err = skipped = 0
    records = []
    orig_enrich = curate.enrich_config_with_subset

    for hit in hits:
        if curate.is_already_copied(copy_map, hit["path"]):
            skipped += 1
            continue

        def enrich_for_hit(config, *, dataset, raw_config, _hit=hit):
            out = orig_enrich(config, dataset=dataset, raw_config=raw_config)
            out = dict(out)
            ds = _hit.get("dataset") or dataset
            if ds:
                out["dataset"] = ds
            nick = leaderboard_nickname(
                raw_config=_hit.get("raw_config") or raw_config,
                config_label=_hit.get("config"),
            )
            if nick:
                out["config_nickname"] = nick
            return out

        curate.enrich_config_with_subset = enrich_for_hit
        try:
            rec = curate.relog_run(
                api,
                hit,
                dst_project=dst_project,
                dst_entity=dst_entity,
                copy_map=copy_map,
                dry_run=dry_run,
                skip_files=skip_files,
                summary_only=summary_only,
                copy_map_path=copy_map_path,
            )
        finally:
            curate.enrich_config_with_subset = orig_enrich

        records.append(rec)
        st = rec.get("status")
        if st in ("ok", "dry_run"):
            ok += 1
        elif st == "skipped":
            skipped += 1
        else:
            err += 1
            print(f"error {hit['path']}: {rec.get('error', rec)}", flush=True)

    return {"ok": ok, "skipped": skipped, "error": err, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="06-19-", help="Run group prefix to include")
    parser.add_argument(
        "--missing-eval",
        action="store_true",
        help="Relog stem-format staged_eval / mmpd_eval runs missing from leaderboard",
    )
    parser.add_argument("--dataset", help="Only include this dataset (with --missing-eval)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", default=DST_ENTITY)
    parser.add_argument("--project", default=DST_PROJECT)
    parser.add_argument("--copy-map", default=DEFAULT_COPY_MAP)
    parser.add_argument("--summary-only", action="store_true", default=True)
    parser.add_argument("--with-history", action="store_true", help="Copy full history (slow)")
    parser.add_argument("--skip-files", action="store_true", default=True)
    parser.add_argument("--with-files", action="store_true")
    args = parser.parse_args()

    if args.with_history:
        args.summary_only = False
    if args.with_files:
        args.skip_files = False

    import wandb

    api = wandb.Api()
    if args.missing_eval:
        hits = discover_missing_eval_runs(
            api, entity=args.entity, dst_project=args.project, dataset=args.dataset
        )
        label = f"missing eval runs{f' for {args.dataset}' if args.dataset else ''}"
    else:
        hits = discover_prefix_runs(api, prefix=args.prefix, entity=args.entity, dst_project=args.project)
        label = f"prefix {args.prefix!r}"
    groups = sorted({h["run_stem"] for h in hits})
    print(f"found {len(hits)} runs across {len(groups)} groups ({label})")
    for g in groups:
        n = sum(1 for h in hits if h["run_stem"] == g)
        print(f"  {g}: {n}")

    if not hits:
        return

    summary = relog_hits(
        api,
        hits,
        dst_project=args.project,
        dst_entity=args.entity,
        copy_map_path=args.copy_map,
        dry_run=args.dry_run,
        summary_only=args.summary_only,
        skip_files=args.skip_files,
    )
    print(summary)


if __name__ == "__main__":
    main()
