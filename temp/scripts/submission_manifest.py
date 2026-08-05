#!/usr/bin/env python3
"""Small stdlib-only writer/reader for Slurm submission manifests.

Submitters write their exact job IDs and expected output roots here rather than
requiring orchestration scripts to scrape human-facing ``sbatch`` output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "jobs": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("jobs", []), list):
        raise ValueError(f"invalid submission manifest: {path}")
    return data


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _pairs(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError(f"expected non-empty KEY=VALUE, got {item!r}")
        values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--path", type=Path, required=True)
    init.add_argument("--component", required=True)
    init.add_argument("--repo", required=True)
    init.add_argument("--datasets", required=True)
    init.add_argument("--set", action="append", default=[])

    record = sub.add_parser("record")
    record.add_argument("--path", type=Path, required=True)
    record.add_argument("--role", required=True)
    record.add_argument("--job-id", required=True)
    record.add_argument("--dataset", default="")
    record.add_argument("--set", action="append", default=[])

    query = sub.add_parser("terminal-job-ids")
    query.add_argument("--path", type=Path, required=True)
    query.add_argument("--roles", required=True, help="comma-separated accepted terminal roles")

    roots = sub.add_parser("checkpoint-root-pairs")
    roots.add_argument("--path", type=Path, required=True)
    roots.add_argument("--role", default="binary_train")

    value = sub.add_parser("value")
    value.add_argument("--path", type=Path, required=True)
    value.add_argument("--key", required=True)

    args = parser.parse_args()
    if args.command == "init":
        payload: dict[str, Any] = {
            "schema_version": 1,
            "component": args.component,
            "repo": args.repo,
            "datasets": [item for item in args.datasets.split(",") if item],
            "jobs": [],
        }
        payload.update(_pairs(args.set))
        _write(args.path, payload)
        return

    payload = _read(args.path)
    if args.command == "record":
        entry: dict[str, Any] = {"role": args.role, "job_id": str(args.job_id)}
        if args.dataset:
            entry["dataset"] = args.dataset
        entry.update(_pairs(args.set))
        jobs = [job for job in payload["jobs"] if not (
            job.get("role") == entry["role"] and job.get("dataset", "") == entry.get("dataset", "")
        )]
        jobs.append(entry)
        payload["jobs"] = jobs
        _write(args.path, payload)
        return

    if args.command == "terminal-job-ids":
        roles = set(args.roles.split(","))
        job_ids = [str(job["job_id"]) for job in payload["jobs"] if job.get("role") in roles]
        if not job_ids:
            raise SystemExit(f"no terminal jobs with roles={args.roles} in {args.path}")
        print(":".join(job_ids))
        return

    if args.command == "checkpoint-root-pairs":
        pairs = []
        seen = set()
        for job in payload["jobs"]:
            if job.get("role") != args.role:
                continue
            dataset = job.get("dataset")
            root = job.get("checkpoint_root")
            if not isinstance(dataset, str) or not isinstance(root, str):
                raise SystemExit(f"missing dataset/checkpoint_root in {args.path}: {job}")
            if dataset in seen:
                raise SystemExit(f"duplicate checkpoint root for {dataset} in {args.path}")
            seen.add(dataset)
            pairs.append(f"{dataset}={root}")
        if not pairs:
            raise SystemExit(f"no checkpoint roots for role={args.role} in {args.path}")
        print(",".join(pairs))
        return

    if args.key not in payload or not isinstance(payload[args.key], str):
        raise SystemExit(f"missing string key={args.key} in {args.path}")
    print(payload[args.key])


if __name__ == "__main__":
    main()
