#!/usr/bin/env python3
"""Split leaderboard report tabs by forecast horizon and prune bad ordinal_norm runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import secrets
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

REPORT_ID = "VmlldzoxNzMyOTkxMw=="
ENTITY = "calvincao"
PROJECT = "ts-sandbox-leaderboard"

META_TAGS = frozenset({"eval", "archive", "mmpd", "binary", "stub"})

HORIZON_SPECS = (
    ("96/96", 96, r"(lb336[_-]hz96|(?<!0)hz96)"),
    ("336/720", 720, r"(lb336[_-]hz720|lb96[_-]hz720|hz720)"),
)

BATCH_GROUP_RE = re.compile(
    r"^07-0[567]-.*(?:ordinal_norm|ordinal-norm)", re.IGNORECASE
)
PROTECT_GROUP_RE = re.compile(r"07-06-408756\d")


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


def _keep_base_filter(item: Dict[str, Any]) -> bool:
    if "filters" in item:
        return False
    key = item.get("key") or {}
    if key.get("section") == "run" and key.get("name") == "createdAt":
        return not item.get("disabled", False)
    if key.get("section") != "tags":
        return False
    name = key.get("name")
    if name == "eval":
        return item.get("op") == "=" and item.get("value") is True
    if name == "archive":
        return item.get("op") == "!=" and item.get("value") is False
    if name in META_TAGS:
        return False
    return item.get("op") == "=" and item.get("value") is True


def _horizon_filter_group(forecast_length: int, group_regex: str) -> Dict[str, Any]:
    # Group connector must be OR: MMPD stubs match run.group regex but have no
    # experiment.value.forecast_length in config (AND would drop them all).
    return {
        "filters": [
            {
                "key": {"section": "run", "name": "group"},
                "op": "=",
                "value": group_regex,
                "isRegex": True,
            },
            {
                "key": {
                    "section": "config",
                    "name": "experiment.value.forecast_length",
                },
                "op": "=",
                "value": forecast_length,
                "connector": "OR",
            },
        ],
        "connector": "OR",
    }


def _build_filters(original: Dict[str, Any], label: str) -> Dict[str, Any]:
    _ = label
    base_items: List[Dict[str, Any]] = []
    for item in original.get("filters") or []:
        if _keep_base_filter(item):
            base_items.append(copy.deepcopy(item))

    out_items = list(base_items)
    for horizon_label, forecast_len, group_re in HORIZON_SPECS:
        if horizon_label == label:
            out_items.append(_horizon_filter_group(forecast_len, group_re))
            break

    return {"filterFormat": "filterV2", "filters": out_items}


def _split_runset(runset: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset = _dataset_from_filters(runset.get("filters") or {})
    if not dataset:
        dataset = (runset.get("name") or "dataset").split()[0]

    out: List[Dict[str, Any]] = []
    for label, _, _ in HORIZON_SPECS:
        rs = copy.deepcopy(runset)
        rs["id"] = _new_runset_id()
        rs["name"] = f"{dataset} — {label}"
        rs["filters"] = _build_filters(runset.get("filters") or {}, label)
        out.append(rs)
    return out


def fix_horizon_filter_connectors(spec: Dict[str, Any]) -> int:
    """Patch already-split report runsets where horizon group used AND instead of OR."""
    fixed = 0
    for block in spec.get("blocks") or []:
        if block.get("type") != "panel-grid":
            continue
        for rs in (block.get("metadata") or {}).get("runSets") or []:
            filters = (rs.get("filters") or {}).get("filters") or []
            if not filters:
                continue
            last = filters[-1]
            if not isinstance(last, dict) or "filters" not in last:
                continue
            if last.get("connector") == "OR":
                continue
            last["connector"] = "OR"
            fixed += 1
    return fixed


def split_report_runsets(spec: Dict[str, Any]) -> Tuple[int, int]:
    before = 0
    after = 0
    for block in spec.get("blocks") or []:
        if block.get("type") != "panel-grid":
            continue
        md = block.get("metadata") or {}
        runsets = md.get("runSets") or []
        before += len(runsets)
        new_runsets: List[Dict[str, Any]] = []
        for rs in runsets:
            new_runsets.extend(_split_runset(rs))
        md["runSets"] = new_runsets
        md["openRunSet"] = 0
        block["metadata"] = md
        after += len(new_runsets)
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
        print("[report] dry-run: would upsert_view", variables["id"], variables["displayName"])
        return
    execute_graphql(api, gql.upsert_view, variables)
    print("[report] saved via upsert_view")


def report_url(view: Dict[str, Any]) -> str:
    title = (view.get("displayName") or "report").replace(" ", "-")
    rid = view["id"]
    return f"https://wandb.ai/{ENTITY}/{PROJECT}/reports/{title}--{rid}"


def _is_ordinal_norm_batch_run(run: Any) -> bool:
    group = run.group or ""
    if BATCH_GROUP_RE.search(group):
        return True
    cfg = dict(run.config)
    nick = str(cfg.get("config_nickname", ""))
    if re.match(r"^07-0[567]-", group) and "ordinal_norm" in nick:
        return True
    return False


def _should_keep_run(run: Any) -> bool:
    group = run.group or ""
    if run.job_type == "mmpd_eval":
        return True
    if PROTECT_GROUP_RE.search(group):
        return True
    summary = dict(run.summary)
    if summary.get("eval/staged_anchor_mse") is not None:
        return True
    if run.job_type == "staged_eval" and summary.get("eval/staged_crps") is not None:
        return True
    return False


def _should_delete_run(run: Any) -> bool:
    if not _is_ordinal_norm_batch_run(run):
        return False
    if _should_keep_run(run):
        return False
    state = run.state
    if state in ("crashed", "failed", "killed", "preempted"):
        return True
    if state != "finished":
        return False
    summary = dict(run.summary)
    if run.job_type == "pipeline" and summary.get("eval/staged_anchor_mse") is None:
        return True
    if (
        run.job_type == "staged_eval"
        and summary.get("eval/staged_anchor_mse") is None
        and summary.get("eval/staged_crps") is None
    ):
        return True
    return False


def _run_path(run: Any) -> str:
    path = getattr(run, "path", None)
    if isinstance(path, list):
        return "/".join(str(p) for p in path)
    return str(path).lstrip("/")


def delete_bad_runs(api: Any, dry_run: bool) -> List[str]:
    to_delete = [run for run in api.runs(f"{ENTITY}/{PROJECT}") if _should_delete_run(run)]
    deleted_paths: List[str] = []
    for run in to_delete:
        path = _run_path(run)
        deleted_paths.append(path)
        msg = f"[delete] {run.state} {run.job_type} {run.group} ({path})"
        if dry_run:
            print(f"would {msg}")
            continue
        print(msg, flush=True)
        run.delete()
    return deleted_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-id", default=REPORT_ID)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-delete", action="store_true")
    parser.add_argument(
        "--fix-horizon-only",
        action="store_true",
        help="Patch horizon OR connector on existing split tabs (no re-split).",
    )
    args = parser.parse_args()

    import wandb

    api = wandb.Api()

    before = after = 0
    view: Optional[Dict[str, Any]] = None
    if not args.skip_report:
        view = fetch_report_view(api, args.report_id)
        spec = json.loads(view["spec"])
        if args.fix_horizon_only:
            fixed = fix_horizon_filter_connectors(spec)
            print(f"[report] fixed horizon connector on {fixed} runsets")
        else:
            before, after = split_report_runsets(spec)
            print(f"[report] runsets before={before} after={after}")
        upsert_report_view(api, view, spec, dry_run=args.dry_run)
        print(f"[report] url={report_url(view)}")

    if not args.skip_delete:
        paths = delete_bad_runs(api, dry_run=args.dry_run)
        print(f"[delete] {'would delete' if args.dry_run else 'deleted'} {len(paths)} runs")

    if args.dry_run:
        print("[dry-run] no wandb mutations applied")


if __name__ == "__main__":
    main()
