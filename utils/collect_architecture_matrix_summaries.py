#!/usr/bin/env python3
"""Print one consolidated table of metrics for a batch of architecture-matrix jobs.

Each job writes ``<stem>/datasets/summary.csv``. Stems look like
``MM-DD-<jobid>-<job-name-slug>`` (Slurm job name: variant, plus ``-smoke`` / ``-h100``
when applicable). Legacy folders may still contain ``unet-fullvar`` in the slug.

Feed the manifest TSV produced by ``utils/submit_architecture_matrix.sh`` (column
``job_id``), or pass explicit Slurm job ids.

Examples::

    python3 utils/collect_architecture_matrix_summaries.py \\
        results/architecture_matrix_manifest_20260508_210530.tsv

    python3 utils/collect_architecture_matrix_summaries.py \\
        --results-root results 3498123 3498124 3498125

Exit code 1 if no summary rows could be loaded.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _variant_slug_from_stem_name(name: str) -> str:
    """Suffix after MM-DD-JOBID- (supports legacy ``...-unet-fullvar-<rest>``)."""
    if "unet-fullvar-" in name:
        return name.split("unet-fullvar-", 1)[-1]
    m = re.match(r"^\d{2}-\d{2}-\d+-(.+)$", name)
    return m.group(1) if m else ""


def _find_stem_for_job(results_root: Path, job_id: str) -> Optional[Path]:
    if job_id in ("?", "-", ""):
        return None
    pat = f"*-{job_id}-*"
    matches = sorted(p for p in results_root.glob(pat) if p.is_dir())
    if not matches:
        return None
    # Prefer canonical ``MM-DD-JOBID-…`` top-level stems if globs overlap (e.g. backup copies).
    prefix_re = re.compile(rf"^\d{{2}}-\d{{2}}-{re.escape(job_id)}-")
    canonical = [m for m in matches if prefix_re.match(m.name)]
    if canonical:
        return canonical[0]
    return matches[0]


def _read_manifest(path: Path) -> List[Tuple[str, str, str]]:
    """Return list of (job_id, variant, description).

    Columns: ``job_id``, ``variant``, optional ``dataset``, ``description``
    (tab-separated). Older manifests omit ``dataset``.
    """
    rows: List[Tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "job_id":
                continue
            if len(parts) < 2:
                continue
            jid, variant = parts[0], parts[1]
            if len(parts) >= 4:
                desc = parts[3]
            else:
                desc = parts[2] if len(parts) > 2 else ""
            rows.append((jid, variant, desc))
    return rows


def _load_summary_csv(stem: Path) -> Optional[Dict[str, str]]:
    p = stem / "datasets" / "summary.csv"
    if not p.is_file():
        return None
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None
    return rows[-1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "manifest_or_jobs",
        nargs="*",
        help="Manifest .tsv path, or job ids when using --results-root",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory containing MM-DD-JOBID-… stems (default: ./results)",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Emit merged metrics as CSV on stdout (includes variant and stem columns)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    results_root = args.results_root
    if not results_root.is_dir():
        print(f"[ERROR] results root not found: {results_root}", file=sys.stderr)
        return 1

    entries: List[Tuple[str, str, str, Optional[Path]]] = []

    mo = args.manifest_or_jobs
    if len(mo) == 1 and str(mo[0]).endswith(".tsv") and Path(mo[0]).is_file():
        manifest = Path(mo[0])
        for jid, variant, desc in _read_manifest(manifest):
            stem = _find_stem_for_job(results_root, jid)
            entries.append((variant, desc, jid, stem))
    elif mo:
        for jid in mo:
            stem = _find_stem_for_job(results_root, jid)
            variant = _variant_slug_from_stem_name(stem.name) if stem else ""
            entries.append((variant, "", jid, stem))
    else:
        p.print_help()
        return 2

    merged: List[Dict[str, str]] = []
    pending: List[str] = []

    for variant, desc, jid, stem in entries:
        if stem is None or not stem.is_dir():
            pending.append(f"{variant or '?'}  job {jid}  [no results dir matching job id under {results_root}]")
            continue
        row_dict = _load_summary_csv(stem)
        if row_dict is None:
            pending.append(f"{variant or '?'}  job {jid}  [no summary.csv under {stem}/datasets — still running?]")
            continue
        out = dict(row_dict)
        out["arch_variant"] = variant
        out["slurm_job_id"] = jid
        out["results_stem"] = str(stem)
        if desc:
            out["arch_description"] = desc
        merged.append(out)

    if args.csv:
        if merged:
            all_keys: List[str] = []
            for row in merged:
                for k in row:
                    if k not in all_keys:
                        all_keys.append(k)
            w = csv.DictWriter(sys.stdout, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)
        return 0 if merged else 1

    print("")
    print("=" * 88)
    print("  Architecture matrix — consolidated summary (datasets/summary.csv per job)")
    print("=" * 88)
    print("")

    if pending:
        print("  Pending / missing:")
        for line in pending:
            print(f"    - {line}")
        print("")

    if not merged:
        print("  No summary.csv rows loaded — check job state or paths.")
        print("=" * 88)
        return 1

    # Human-readable table: key metric columns if present
    prefer = [
        "arch_variant",
        "slurm_job_id",
        "dataset",
        "subset_id",
        "avg_mae",
        "avg_mse",
        "avg_trend_acc",
        "itrans_mae",
        "itrans_mse",
        "best_val_loss",
    ]
    cols = [c for c in prefer if any(c in row for row in merged)]
    if not cols:
        cols = sorted({k for row in merged for k in row})

    widths = {c: max(len(c), max(len(str(row.get(c, ""))) for row in merged)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(f"  {header}")
    print(f"  {sep}")
    for row in merged:
        line = " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        print(f"  {line}")

    print("")
    print("=" * 88)
    print(f"  OK: {len(merged)} / {len(entries)} experiment(s) with summary.csv")
    print("  (Pass --csv for machine-readable merged output.)")
    print("=" * 88)
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
