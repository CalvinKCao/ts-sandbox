#!/usr/bin/env python3
"""Split leaderboard report tabs by horizon using original filters + config fields."""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.backfill_leaderboard_horizon_fields import backfill_runs
from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

REPORT_ID = "VmlldzoxNzMyOTkxMw=="
ENTITY = "calvincao"
PROJECT = "ts-sandbox-leaderboard"

HORIZON_TABS = (
    ("96/96", 96),
    ("336/720", 720),
)

META_TAGS = frozenset({"eval", "archive", "mmpd", "binary", "stub"})


def _new_runset_id() -> str:
    return secrets.token_urlsafe(9)[:12]


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


def _dataset_from_filters(filters: Dict[str, Any]) -> Optional[str]:
    for item in filters.get("filters") or []:
        if "filters" in item:
            continue
        key = item.get("key") or {}
        if key.get("section") != "tags":
            continue
        name = key.get("name")
        if not name or name in META_TAGS:
            continue
        if item.get("op") == "=" and item.get("value") is True:
            return name
    return None


def _is_original_base_filter(item: Dict[str, Any]) -> bool:
    if "filters" in item:
        return False
    key = item.get("key") or {}
    section = key.get("section")
    name = key.get("name")
    if section == "tags":
        if name == "eval":
            return item.get("op") == "=" and item.get("value") is True
        if name == "archive":
            return item.get("op") == "!=" and item.get("value") is False
        if name in META_TAGS:
            return False
        return item.get("op") == "=" and item.get("value") is True
    if section == "run" and name == "createdAt":
        return not item.get("disabled", False)
    if section == "config":
        return False
    return False


def _original_filters_from_runset(runset: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        copy.deepcopy(item)
        for item in (runset.get("filters") or {}).get("filters") or []
        if _is_original_base_filter(item)
    ]


def _horizon_config_filter(horizon: int) -> Dict[str, Any]:
    return {
        "key": {"section": "config", "name": "experiment.forecast_length"},
        "op": "=",
        "value": horizon,
    }


def _collapse_to_original_runsets(runsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for rs in runsets:
        name = rs.get("name") or ""
        dataset = name.split(" — ", 1)[0] if " — " in name else _dataset_from_filters(rs.get("filters") or {})
        if not dataset:
            continue
        if dataset in by_dataset:
            continue
        base = copy.deepcopy(rs)
        base["name"] = dataset
        base["filters"] = {
            "filterFormat": "filterV2",
            "filters": _original_filters_from_runset(rs),
        }
        by_dataset[dataset] = base
    return [by_dataset[k] for k in sorted(by_dataset)]


def _split_runset(runset: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset = runset.get("name") or _dataset_from_filters(runset.get("filters") or {}) or "dataset"
    base_filters = _original_filters_from_runset(runset)
    out: List[Dict[str, Any]] = []
    for label, horizon in HORIZON_TABS:
        rs = copy.deepcopy(runset)
        rs["id"] = _new_runset_id()
        rs["name"] = f"{dataset} — {label}"
        rs["filters"] = {
            "filterFormat": "filterV2",
            "filters": base_filters + [_horizon_config_filter(horizon)],
        }
        out.append(rs)
    return out


def rebuild_report(spec: Dict[str, Any]) -> Tuple[int, int]:
    before = 0
    after = 0
    for block in spec.get("blocks") or []:
        if block.get("type") != "panel-grid":
            continue
        md = block.get("metadata") or {}
        originals = _collapse_to_original_runsets(md.get("runSets") or [])
        before = len(md.get("runSets") or [])
        new_runsets: List[Dict[str, Any]] = []
        for rs in originals:
            new_runsets.extend(_split_runset(rs))
        md["runSets"] = new_runsets
        md["openRunSet"] = 0
        block["metadata"] = md
        after = len(new_runsets)
    return before, after


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-id", default=REPORT_ID)
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()

    if not args.skip_backfill:
        scanned, updated = backfill_runs(api, dry_run=args.dry_run)
        print(f"[backfill] scanned={scanned} updated={updated}")

    if not args.skip_report:
        view = fetch_report_view(api, args.report_id)
        spec = json.loads(view["spec"])
        before, after = rebuild_report(spec)
        print(f"[report] runsets {before} -> {after} (from {after // 2} datasets)")
        upsert_report_view(api, view, spec, dry_run=args.dry_run)
        print(f"[report] {report_url(view)}")


if __name__ == "__main__":
    main()
