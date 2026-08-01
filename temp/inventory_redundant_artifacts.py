#!/usr/bin/env python3
"""Static inventory of likely-redundant configs / scripts / helpers.

Candidates only — does not delete. Cross-checks coverage JSON when provided.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stems / paths known to be live entrypoints for the current campaign + probe.
LIVE_CONFIG_STEMS = {
    "coverage_deadcode_binary_patch_refine",
    "coverage_deadcode_mmpd",
    "binary_patch_refine_lb336_hz96",
    "binary_patch_refine_lb336_hz96_full",
    "binary_patch_refine_lb336_hz96_full_eval_viz",
    "binary_patch_refine_lb336_hz96_ordinal_tuned",
    "binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary",
    "mmpd_decoder_flat_subsets_paper_lb336_hz96",
    "mmpd_decoder_flat_subsets_paper_lb336_hz720",
    "mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm",
    "mmpd_decoder_flat_subsets_grad_accum_200_lr_lo",
    "mmpd_ordinal_upscale_lb96_hz16",
    "smoke_test",
    "base/binary_staged",
    "base/fixed_lr_pipeline_base",
}

LIVE_SCRIPTS = {
    "submit_binary.sh",
    "submit_mmpd.sh",
    "slurm_worker.sh",
    "temp/run_h96_ordinal_patch_refine_mmpd_dag.sh",
    "temp/run_pipeline_coverage_deadcode.py",
    "temp/submit_pipeline_coverage_deadcode.sh",
    "temp/inventory_redundant_artifacts.py",
    "temp/submission_manifest.py",
    "temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py",
}

MISSING_STEM_RE = re.compile(r"binary_anchor_stationary_flat[\w\-]*")


def _stem(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT / "configs").as_posix()
    if rel.endswith(".yaml"):
        rel = rel[: -len(".yaml")]
    return rel


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _iter_repo_text_files() -> Iterable[Path]:
    skip_dirs = {
        ".git",
        ".venv",
        "results",
        "node_modules",
        "MMPD",
        "mmpd_datasets",
        "synth_data",
        "htmlcov",
        "__pycache__",
        "coverage_wheels",
    }
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(REPO_ROOT).parts
        if any(p in skip_dirs for p in rel_parts):
            continue
        if path.suffix.lower() not in {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".txt"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        yield path


def _referenced_config_stems(texts: Dict[Path, str]) -> Set[str]:
    found: Set[str] = set(LIVE_CONFIG_STEMS)
    # extends: configs/foo.yaml or configs/base/bar.yaml
    pat = re.compile(
        r"(?:extends:\s*|configs/)([A-Za-z0-9_./\-]+?)(?:\.yaml)?(?:\s|$|'|\")"
    )
    for text in texts.values():
        for m in pat.finditer(text):
            token = m.group(1).strip().lstrip("./")
            if token.startswith("configs/"):
                token = token[len("configs/") :]
            if token.endswith(".yaml"):
                token = token[: -len(".yaml")]
            if token:
                found.add(token)
    return found


def _coverage_zero_files(coverage_json: Path | None) -> List[str]:
    if coverage_json is None or not coverage_json.is_file():
        return []
    payload = json.loads(coverage_json.read_text(encoding="utf-8"))
    files = payload.get("files") or {}
    out: List[str] = []
    for key, info in files.items():
        summary = info.get("summary") or {}
        if int(summary.get("covered_lines") or 0) == 0:
            try:
                rel = str(Path(key).resolve().relative_to(REPO_ROOT))
            except Exception:
                rel = key
            out.append(rel)
    return sorted(out)


def build_report(coverage_json: Path | None) -> str:
    configs = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
    texts = {p: _read_text(p) for p in _iter_repo_text_files()}
    referenced = _referenced_config_stems(texts)

    orphan_configs: List[str] = []
    for cfg in configs:
        stem = _stem(cfg)
        if stem.startswith("base/"):
            continue
        if stem in LIVE_CONFIG_STEMS:
            continue
        # Keep if any text file mentions the stem basename
        base = Path(stem).name
        mentioned = any(base in t for t in texts.values())
        if not mentioned and stem not in referenced:
            orphan_configs.append(stem)

    missing_stem_refs: List[str] = []
    for path, text in texts.items():
        for m in MISSING_STEM_RE.finditer(text):
            missing_stem_refs.append(f"{path.relative_to(REPO_ROOT)}:{m.group(0)}")

    temp_submits = sorted((REPO_ROOT / "temp").glob("submit_*_killarney.sh"))
    root_slurm = sorted(REPO_ROOT.glob("slurm_*.sh"))
    diag_scripts = [
        str(p.relative_to(REPO_ROOT))
        for p in [*temp_submits, *root_slurm]
        if str(p.relative_to(REPO_ROOT)) not in LIVE_SCRIPTS
    ]

    # JSON under temp/ that look like one-off manifests (not results/)
    temp_json = [
        str(p.relative_to(REPO_ROOT))
        for p in sorted((REPO_ROOT / "temp").rglob("*.json"))
        if "MMPD" not in p.parts and p.stat().st_size < 500_000
    ]

    zero_cov = _coverage_zero_files(coverage_json)

    lines: List[str] = []
    lines.append("# Redundant artifact candidates")
    lines.append("")
    lines.append("Generated by `temp/inventory_redundant_artifacts.py`.")
    lines.append("These are **candidates** for deletion/refactor — verify before removing.")
    lines.append("")
    lines.append("## Config YAML stems with no in-repo reference")
    lines.append("")
    if orphan_configs:
        for stem in orphan_configs:
            lines.append(f"- `configs/{stem}.yaml`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## References to missing `binary_anchor_stationary_flat*` stems")
    lines.append("")
    if missing_stem_refs:
        for row in missing_stem_refs[:80]:
            lines.append(f"- `{row}`")
        if len(missing_stem_refs) > 80:
            lines.append(f"- … and {len(missing_stem_refs) - 80} more")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Diagnostic / legacy submit scripts (not live DAG entrypoints)")
    lines.append("")
    for rel in diag_scripts:
        lines.append(f"- `{rel}`")
    if not diag_scripts:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Committed JSON under `temp/`")
    lines.append("")
    for rel in temp_json[:60]:
        lines.append(f"- `{rel}`")
    if len(temp_json) > 60:
        lines.append(f"- … and {len(temp_json) - 60} more")
    if not temp_json:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Coverage: files with 0 executed lines (when JSON provided)")
    lines.append("")
    if coverage_json is None:
        lines.append("- (no coverage JSON passed)")
    elif not zero_cov:
        lines.append("- (none with 0 covered lines in report)")
    else:
        for rel in zero_cov[:120]:
            lines.append(f"- `{rel}`")
        if len(zero_cov) > 120:
            lines.append(f"- … and {len(zero_cov) - 120} more")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Vertical-dual / channel-dual / past-native leaves may be unused by this")
    lines.append("  coverage run but still valid for other campaigns — do not mass-delete.")
    lines.append("- `legacy.md` is referenced by AGENTS.md / architecture.md but may be missing.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "redundant_candidates.md",
    )
    p.add_argument("--coverage-json", type=Path, default=None)
    args = p.parse_args()
    report = build_report(args.coverage_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
