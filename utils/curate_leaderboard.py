#!/usr/bin/env python3
"""Curate ts-sandbox-leaderboard: 3 runs per tab (MMPD Subset Recent + top 2 binary).

Archives other mmpd/binary eval rows, simplifies report filters, strips legacy eval tags.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.backfill_leaderboard_horizon_fields import apply_horizon_fields, infer_lookback_horizon
from utils.leaderboard_config_nicknames import leaderboard_dataset_tags, parse_run_stem
from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from models.diffusion_tsf.pipeline.wandb_utils import is_binary_eval_run

REPORT_ID = "VmlldzoxNzMyOTkxMw=="
ENTITY = "calvincao"
PROJECT = "ts-sandbox-leaderboard"
MMPD_NICKNAME = "MMPD Subset Recent"

# 96/96 tab filters lb=96, hz=96. Keepers are picked from the hz=96 pool then normalized.
HORIZON_TABS = (
    ("96/96", 96, 96),
    ("336/720", 336, 720),
)

DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "PeMS",
    "dynamic",
    "electricity",
    "exchange_rate",
    "illness",
    "solar_Alabama",
    "traffic",
    "weather",
)

LEGACY_TAGS = frozenset({"eval"})


def _anchor_mse(run: Any) -> float:
    v = (getattr(run, "summary", None) or {}).get("eval/staged_anchor_mse")
    if v is None:
        return float("inf")
    return float(v)


def _config_nickname(run: Any) -> str:
    return str((dict(run.config or {}).get("config_nickname") or "")).strip()


def is_mmpd_run(run: Any) -> bool:
    tags = set(run.tags or [])
    return run.job_type == "mmpd_eval" or "mmpd" in tags


def is_leaderboard_candidate(run: Any) -> bool:
    """Binary or MMPD eval rows eligible for leaderboard tabs."""
    if is_mmpd_run(run):
        return _anchor_mse(run) != float("inf")
    tags = set(run.tags or [])
    if run.job_type == "classical_baseline" or "classical-baseline" in tags:
        return False
    if is_binary_eval_run(run):
        return _anchor_mse(run) != float("inf")
    if "binary" in tags and _anchor_mse(run) != float("inf"):
        return True
    return False


def canonical_dataset(run: Any) -> str:
    parsed = parse_run_stem(run.group or "")
    if parsed:
        return parsed[1]
    cfg = dict(run.config or {})
    if cfg.get("dataset"):
        return str(cfg["dataset"])
    exp = cfg.get("experiment") or {}
    if isinstance(exp, dict) and exp.get("dataset"):
        return str(exp["dataset"])
    return ""


def dataset_match(run: Any, dataset: str) -> bool:
    canon = canonical_dataset(run)
    if canon:
        return canon == dataset
    tags = set(run.tags or [])
    return dataset in tags or any(t in tags for t in leaderboard_dataset_tags(dataset))


def tab_key(dataset: str, lb: int, hz: int) -> str:
    return f"{dataset}|{lb}|{hz}"


def tab_label(dataset: str, horizon_label: str) -> str:
    return f"{dataset} — {horizon_label}"


def short_horizon_selection_pool(run: Any, dataset: str) -> bool:
    """Candidates for the 96/96 tab before normalizing lookback/horizon fields to 96."""
    if not dataset_match(run, dataset):
        return False
    if not is_leaderboard_candidate(run):
        return False
    _, hz = infer_lookback_horizon(run)
    return hz == 96


def horizon_match(run: Any, lb: int, hz: int) -> bool:
    rlb, rhz = infer_lookback_horizon(run)
    return rlb == lb and rhz == hz


def select_tab_keepers(
    candidates: List[Any],
    *,
    pin_binary_id: Optional[str] = None,
    prefer_mmpd_nickname: bool = False,
) -> List[Any]:
    mmpd_all = [r for r in candidates if is_mmpd_run(r)]
    if prefer_mmpd_nickname:
        named = [r for r in mmpd_all if _config_nickname(r) == MMPD_NICKNAME]
        mmpd = named if named else mmpd_all
    else:
        mmpd = mmpd_all
    mmpd.sort(key=_anchor_mse)
    binary = [r for r in candidates if not is_mmpd_run(r)]
    binary.sort(key=_anchor_mse)

    keep_binary: List[Any] = []
    if pin_binary_id:
        pinned = next((r for r in binary if r.id == pin_binary_id), None)
        if pinned is not None:
            keep_binary.append(pinned)
    for r in binary:
        if r in keep_binary:
            continue
        keep_binary.append(r)
        if len(keep_binary) >= 2:
            break

    keep: List[Any] = []
    if mmpd:
        keep.append(mmpd[0])
    keep.extend(keep_binary[:2])
    keep.sort(key=_anchor_mse)
    return keep


def normalize_short_tab_keepers(keepers: List[Any], *, dry_run: bool) -> int:
    """Pin 96/96 keepers to lb=96, hz=96 and label the MMPD row."""
    updated = 0
    for run in keepers:
        if dry_run:
            print(f"[96/96] would normalize {run.id} ({canonical_dataset(run)})")
            updated += 1
            continue
        if is_mmpd_run(run):
            run.config["config_nickname"] = MMPD_NICKNAME
            run.update()
        apply_horizon_fields(run, 96, 96)
        print(f"[96/96] normalized {run.id} ({canonical_dataset(run)})")
        updated += 1
    return updated


def record_baseline(runs: List[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASETS:
        for horizon_label, lb, hz in HORIZON_TABS:
            label = tab_label(dataset, horizon_label)
            candidates = [
                r
                for r in runs
                if "archive" not in (r.tags or [])
                and dataset_match(r, dataset)
                and horizon_match(r, lb, hz)
                and is_leaderboard_candidate(r)
                and not is_mmpd_run(r)
            ]
            candidates.sort(key=_anchor_mse)
            top = candidates[0] if candidates else None
            out[label] = {
                "id": top.id if top else None,
                "name": top.name if top else None,
                "group": top.group if top else None,
                "mse": _anchor_mse(top) if top else None,
                "config_nickname": _config_nickname(top) if top else None,
                "candidate_count": len(candidates),
            }
    return out


def backfill_horizons(api: Any, *, dry_run: bool) -> Tuple[int, int]:
    scanned = updated = 0
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        if not is_leaderboard_candidate(run):
            continue
        scanned += 1
        cfg = dict(run.config or {})
        if cfg.get("leaderboard_lookback") == 96 and cfg.get("leaderboard_horizon") == 96:
            continue
        lb, hz = infer_lookback_horizon(run)
        exp = cfg.get("experiment") or {}
        exp_lb = exp.get("lookback_length") if isinstance(exp, dict) else None
        exp_hz = exp.get("forecast_length") if isinstance(exp, dict) else None
        if (
            cfg.get("leaderboard_lookback") == lb
            and cfg.get("leaderboard_horizon") == hz
            and exp_lb == lb
            and exp_hz == hz
        ):
            continue
        if dry_run:
            print(f"[backfill] would set {run.id} -> lb={lb} hz={hz}")
        else:
            apply_horizon_fields(run, lb, hz)
            print(f"[backfill] {run.id} -> lb={lb} hz={hz}")
        updated += 1
    return scanned, updated


def has_leaderboard_metric(run: Any) -> bool:
    return (getattr(run, "summary", None) or {}).get("eval/staged_anchor_mse") is not None


def matches_any_leaderboard_tab(run: Any) -> bool:
    if "archive" in (run.tags or []):
        return False
    for dataset in DATASETS:
        for _, lb, hz in HORIZON_TABS:
            if dataset_match(run, dataset) and horizon_match(run, lb, hz):
                return True
    return False


def archive_tab_junk(runs: List[Any], *, dry_run: bool) -> int:
    """Archive rows that match a leaderboard tab but lack eval/staged_anchor_mse."""
    archived = 0
    for run in runs:
        if not matches_any_leaderboard_tab(run):
            continue
        if has_leaderboard_metric(run):
            continue
        tags = list(run.tags or [])
        if "archive" in tags:
            continue
        tags.append("archive")
        for legacy in LEGACY_TAGS:
            if legacy in tags:
                tags = [t for t in tags if t != legacy]
        if dry_run:
            print(f"[junk] would archive {run.id} ({run.job_type}) {run.group[:60]}")
            archived += 1
            continue
        run.tags = list(dict.fromkeys(tags))
        run.update()
        archived += 1
    return archived


def curate_tags(
    runs: List[Any],
    keep_ids: Set[str],
    *,
    dry_run: bool,
) -> Tuple[int, int]:
    archived = unarchived = 0
    for run in runs:
        if not is_leaderboard_candidate(run):
            continue
        tags = list(run.tags or [])
        changed = False
        if run.id in keep_ids:
            if "archive" in tags:
                tags = [t for t in tags if t != "archive"]
                unarchived += 1
                changed = True
        else:
            if "archive" not in tags:
                tags.append("archive")
                archived += 1
                changed = True
        for legacy in LEGACY_TAGS:
            if legacy in tags:
                tags = [t for t in tags if t != legacy]
                changed = True
        if not changed:
            continue
        if dry_run:
            print(f"[tags] would update {run.id} tags={tags}")
            continue
        run.tags = list(dict.fromkeys(tags))
        run.update()
    return archived, unarchived


def compute_keep_ids(
    runs: List[Any],
    baseline: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Set[str], Dict[str, List[str]]]:
    keep_ids: Set[str] = set()
    per_tab: Dict[str, List[str]] = {}
    for dataset in DATASETS:
        for horizon_label, lb, hz in HORIZON_TABS:
            label = tab_label(dataset, horizon_label)
            if horizon_label == "96/96":
                candidates = [r for r in runs if short_horizon_selection_pool(r, dataset)]
                pin_id = None
                if baseline and baseline.get(label, {}).get("id"):
                    pin_id = baseline[label]["id"]
                keepers = select_tab_keepers(
                    candidates,
                    pin_binary_id=pin_id,
                    prefer_mmpd_nickname=False,
                )
            else:
                candidates = [
                    r
                    for r in runs
                    if dataset_match(r, dataset)
                    and horizon_match(r, lb, hz)
                    and is_leaderboard_candidate(r)
                ]
                pin_id = None
                if baseline and baseline.get(label, {}).get("id"):
                    pin_id = baseline[label]["id"]
                keepers = select_tab_keepers(
                    candidates,
                    pin_binary_id=pin_id,
                    prefer_mmpd_nickname=True,
                )
            per_tab[label] = [r.id for r in keepers]
            keep_ids.update(r.id for r in keepers)
    return keep_ids, per_tab


def _new_runset_id() -> str:
    return secrets.token_urlsafe(9)[:12]


def tab_filters(dataset: str, lb: int, hz: int) -> List[Dict[str, Any]]:
    return [
        {
            "key": {"section": "tags", "name": dataset},
            "op": "=",
            "value": True,
            "disabled": False,
        },
        {
            "key": {"section": "tags", "name": "archive"},
            "op": "!=",
            "value": False,
            "disabled": False,
        },
        {
            "key": {"section": "config", "name": "experiment.lookback_length"},
            "op": "=",
            "value": lb,
        },
        {
            "key": {"section": "config", "name": "experiment.forecast_length"},
            "op": "=",
            "value": hz,
        },
        {
            "key": {"section": "summary", "name": "eval/staged_anchor_mse"},
            "op": "!=",
            "value": None,
        },
    ]


def rebuild_runsets_for_horizon(
    existing: List[Dict[str, Any]],
    horizon_label: str,
) -> List[Dict[str, Any]]:
    hz_map = {name: (lb, hz) for name, lb, hz in HORIZON_TABS}
    if horizon_label not in hz_map:
        raise ValueError(f"unknown horizon label: {horizon_label}")
    lb, hz = hz_map[horizon_label]

    by_dataset: Dict[str, Dict[str, Any]] = {}
    for rs in existing:
        name = rs.get("name") or ""
        dataset = name.split(" — ", 1)[0] if " — " in name else name
        if dataset not in by_dataset:
            by_dataset[dataset] = rs

    out: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        base = copy.deepcopy(by_dataset.get(dataset) or {})
        rs = copy.deepcopy(base) if base else {}
        rs["id"] = _new_runset_id()
        rs["name"] = f"{dataset} — {horizon_label}"
        rs["filters"] = {
            "filterFormat": "filterV2",
            "filters": tab_filters(dataset, lb, hz),
        }
        out.append(rs)
    return out


def rebuild_report_runsets(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, _, _ in HORIZON_TABS:
        out.extend(rebuild_runsets_for_horizon(existing, label))
    return out


def _heading_paragraph(text: str) -> Dict[str, Any]:
    return {
        "type": "paragraph",
        "children": [{"text": text}],
    }


def rebuild_report_spec(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Split 96/96 and 336/720 into separate panel-grid tables."""
    blocks = spec.get("blocks") or []
    grid = next((b for b in blocks if b.get("type") == "panel-grid"), None)
    if grid is None:
        return 0, 0

    before = len((grid.get("metadata") or {}).get("runSets") or [])
    existing_rs = (grid.get("metadata") or {}).get("runSets") or []
    short_rs = rebuild_runsets_for_horizon(existing_rs, "96/96")
    long_rs = rebuild_runsets_for_horizon(existing_rs, "336/720")

    short_grid = copy.deepcopy(grid)
    short_md = copy.deepcopy(grid.get("metadata") or {})
    short_md["runSets"] = short_rs
    short_md["openRunSet"] = 0
    short_md["name"] = "leaderboard-96-96"
    short_grid["metadata"] = short_md

    long_grid = copy.deepcopy(grid)
    long_md = copy.deepcopy(grid.get("metadata") or {})
    long_md["runSets"] = long_rs
    long_md["openRunSet"] = 0
    long_md["name"] = "leaderboard-336-720"
    long_grid["metadata"] = long_md

    other = [b for b in blocks if b.get("type") != "panel-grid"]
    spec["blocks"] = (
        other[:1]
        + [_heading_paragraph("Horizon 96 (lookback 96)")]
        + [short_grid]
        + [_heading_paragraph("Horizon 720 (lookback 336)")]
        + [long_grid]
        + other[1:]
    )
    return before, len(short_rs) + len(long_rs)


def promote_best_mmpd(
    runs: List[Any],
    *,
    lb: int,
    hz: int,
    dry_run: bool,
) -> int:
    """Set config_nickname=MMPD Subset Recent on best MMPD row per dataset for lb/hz."""
    updated = 0
    for dataset in DATASETS:
        candidates = [
            r
            for r in runs
            if is_mmpd_run(r)
            and dataset_match(r, dataset)
            and horizon_match(r, lb, hz)
            and _anchor_mse(r) != float("inf")
        ]
        if not candidates:
            print(f"[mmpd] {dataset} lb{lb}/hz{hz}: no candidates")
            continue
        best = min(candidates, key=_anchor_mse)
        tags = list(best.tags or [])
        changed = False
        if _config_nickname(best) != MMPD_NICKNAME:
            changed = True
        if "archive" in tags:
            tags = [t for t in tags if t != "archive"]
            changed = True
        for legacy in LEGACY_TAGS:
            if legacy in tags:
                tags = [t for t in tags if t != legacy]
                changed = True
        rlb, rhz = infer_lookback_horizon(best)
        cfg = dict(best.config or {})
        exp = cfg.get("experiment") or {}
        exp_lb = exp.get("lookback_length") if isinstance(exp, dict) else None
        exp_hz = exp.get("forecast_length") if isinstance(exp, dict) else None
        if rlb != lb or rhz != hz or exp_lb != lb or exp_hz != hz:
            changed = True
        if not changed:
            print(f"[mmpd] {dataset} lb{lb}/hz{hz}: keep {best.id}")
            continue
        if dry_run:
            print(
                f"[mmpd] would promote {best.id} ({dataset} lb{lb}/hz{hz}) "
                f"mse={_anchor_mse(best):.6f}"
            )
            updated += 1
            continue
        best.config["config_nickname"] = MMPD_NICKNAME
        best.update()
        apply_horizon_fields(best, lb, hz)
        best.tags = list(dict.fromkeys(tags))
        best.update()
        print(
            f"[mmpd] promoted {best.id} ({dataset} lb{lb}/hz{hz}) "
            f"mse={_anchor_mse(best):.6f}"
        )
        updated += 1
    return updated


def _strip_refs(obj: Any) -> None:
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            val = obj[key]
            if key == "ref" and isinstance(val, dict):
                obj.pop(key, None)
                continue
            _strip_refs(val)
    elif isinstance(obj, list):
        for item in obj:
            _strip_refs(item)


def fetch_report_view(api: Any, report_id: str) -> Dict[str, Any]:
    from wandb_workspaces.reports.v2 import gql
    from wandb_workspaces._graphql import execute_graphql

    return execute_graphql(api, gql.view_report, {"reportId": report_id})["view"]


def upsert_report_view(api: Any, view: Dict[str, Any], spec: Dict[str, Any], dry_run: bool) -> None:
    from wandb_workspaces.reports.v2 import gql
    from wandb_workspaces._graphql import execute_graphql

    _strip_refs(spec)
    variables = {
        "id": view["id"],
        "entityName": view["project"]["entityName"],
        "projectName": view["project"]["name"],
        "name": view["name"],
        "displayName": view["displayName"],
        "description": view.get("description") or "",
        "type": "runs",
        "spec": json.dumps(spec),
    }
    if dry_run:
        print("[report] dry-run: would upsert_view", variables["id"])
        return
    execute_graphql(api, gql.upsert_view, variables)
    print("[report] saved")


def report_url(view: Dict[str, Any]) -> str:
    title = (view.get("displayName") or "report").replace(" ", "-")
    return f"https://wandb.ai/{ENTITY}/{PROJECT}/reports/{title}--{view['id']}"


def verify_tabs(runs: List[Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for dataset in DATASETS:
        for horizon_label, lb, hz in HORIZON_TABS:
            tab = tab_label(dataset, horizon_label)
            visible = [
                r
                for r in runs
                if "archive" not in (r.tags or [])
                and dataset_match(r, dataset)
                and horizon_match(r, lb, hz)
                and is_leaderboard_candidate(r)
            ]
            mmpd = [r for r in visible if is_mmpd_run(r)]
            binary = [r for r in visible if not is_mmpd_run(r)]
            recent = [r for r in mmpd if _config_nickname(r) == MMPD_NICKNAME]
            if len(visible) == 0:
                warnings.append(f"{tab}: no runs")
                continue
            if len(visible) > 3:
                errors.append(f"{tab}: expected at most 3 visible runs, got {len(visible)}")
            if len(recent) != 1:
                errors.append(f"{tab}: expected 1 MMPD Subset Recent, got {len(recent)}")
            if len(binary) > 2:
                errors.append(f"{tab}: expected at most 2 binary runs, got {len(binary)}")
            if len(visible) < 3:
                warnings.append(
                    f"{tab}: only {len(visible)} visible "
                    f"(mmpd={len(mmpd)} recent={len(recent)} binary={len(binary)})"
                )
    return errors, warnings


def compare_baseline(
    before: Dict[str, Dict[str, Any]],
    runs: List[Any],
) -> List[str]:
    """Ensure each tab's pre-cleanup top binary is still visible on that tab."""
    mismatches: List[str] = []
    for label, b in before.items():
        top_id = b.get("id")
        if not top_id:
            continue
        parts = label.split(" — ", 1)
        if len(parts) != 2:
            continue
        dataset, horizon_label = parts
        hz_map = {name: (lb, hz) for name, lb, hz in HORIZON_TABS}
        if horizon_label not in hz_map:
            continue
        lb, hz = hz_map[horizon_label]
        visible_ids = {
            r.id
            for r in runs
            if "archive" not in (r.tags or [])
            and dataset_match(r, dataset)
            and horizon_match(r, lb, hz)
            and is_leaderboard_candidate(r)
            and not is_mmpd_run(r)
        }
        if top_id not in visible_ids:
            mismatches.append(f"{label}: top binary {top_id} not visible on tab")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-id", default=REPORT_ID)
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--skip-curate", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-mmpd-promote", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--baseline-out",
        default="",
        help="Write baseline top-binary snapshot JSON to this path.",
    )
    parser.add_argument(
        "--baseline-in",
        default="",
        help="Baseline snapshot JSON for pinning top binary (e.g. from a prior --baseline-out).",
    )
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"[load] {len(runs)} runs")

    baseline_before = record_baseline(runs)
    if args.baseline_in:
        with open(args.baseline_in, encoding="utf-8") as f:
            baseline_pins = json.load(f)
    else:
        baseline_pins = baseline_before
    if args.baseline_out:
        with open(args.baseline_out, "w", encoding="utf-8") as f:
            json.dump(baseline_before, f, indent=2)

    print("\n=== baseline top binary (before) ===")
    for label in sorted(baseline_before):
        b = baseline_before[label]
        mse = f"{b['mse']:.6f}" if b.get("mse") not in (None, float("inf")) else "N/A"
        print(f"{label}: {b.get('id')} mse={mse} nick={b.get('config_nickname')}")

    if not args.skip_backfill:
        scanned, updated = backfill_horizons(api, dry_run=args.dry_run)
        print(f"[backfill] scanned={scanned} updated={updated}")
        if updated and not args.dry_run:
            runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    if not args.skip_mmpd_promote:
        promoted = promote_best_mmpd(runs, lb=336, hz=720, dry_run=args.dry_run)
        print(f"[mmpd] promoted {promoted} lb336/hz720 rows")
        if promoted and not args.dry_run:
            runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    if not args.skip_curate:
        junk_archived = archive_tab_junk(runs, dry_run=args.dry_run)
        print(f"[junk] archived {junk_archived} metric-less tab matches")
        if junk_archived and not args.dry_run:
            runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

        keep_ids, per_tab = compute_keep_ids(runs, baseline_pins)
        short_ids = {rid for key, ids in per_tab.items() if key.endswith(" — 96/96") for rid in ids}
        short_keepers = [r for r in runs if r.id in short_ids]
        normalized = normalize_short_tab_keepers(short_keepers, dry_run=args.dry_run)
        print(f"[96/96] normalized {normalized} keeper rows")
        if normalized and not args.dry_run:
            runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

        print(f"[curate] keeping {len(keep_ids)} runs across {len(per_tab)} tabs")
        archived, unarchived = curate_tags(runs, keep_ids, dry_run=args.dry_run)
        print(f"[curate] archived={archived} unarchived={unarchived}")
        if (archived or unarchived) and not args.dry_run:
            runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    if not args.skip_report:
        view = fetch_report_view(api, args.report_id)
        spec = json.loads(view["spec"])
        before, after = rebuild_report_spec(spec)
        print(f"[report] runsets {before} -> {after} (2 panel-grid tables)")
        upsert_report_view(api, view, spec, dry_run=args.dry_run)
        print(f"[report] {report_url(view)}")

    mismatches = compare_baseline(baseline_pins, runs)
    if mismatches:
        print("\n=== WARNING: baseline top binary no longer visible ===")
        for line in mismatches:
            print(line)
    else:
        print("\n=== baseline top binary still visible on all tabs ===")

    if not args.skip_verify:
        errors, warnings = verify_tabs(runs)
        if warnings:
            print("\n=== verify warnings ===")
            for w in warnings:
                print(w)
        if errors:
            print("\n=== verify FAILED ===")
            for e in errors:
                print(e)
            if not args.dry_run:
                sys.exit(1)
        else:
            print("\n=== verify OK ===")


if __name__ == "__main__":
    main()
