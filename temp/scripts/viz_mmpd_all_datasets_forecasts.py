#!/usr/bin/env python3
"""MMPD forecast viz from canvas128 disc packs: L8 snapproof + H=96 panels.

Reuses snap / snapproof helpers from ``eval_ablation_disc_l8_l16`` (bin-center
shift, occupied-rung ladder, 128-row bin index). Reads binary+MMPD raw packs
under ``results/datasets/08-04-*-ablation-disc-l8-l16-*-c128*``; snaps onto the
absolute patch-refine ladder from ckpt metadata (weights not loaded).

Outputs under ``temp/viz_mmpd_all_datasets/{dataset}/L8_snapproof/`` and
``.../horizon96/``, plus a short README.

Fail-fast: missing pack raw or ckpt metadata raises.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _ladder_only,
    _plot_snap_proof_panel,
    _snap_bundle,
    load_ablation_run,
)
from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    DEFAULT_MMPD_DATA,
    run_variate_indices,
)
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402

RUN_NAME = "window_norm_c128"
DEFAULT_OUT = REPO_ROOT / "temp" / "viz_mmpd_all_datasets"
CANVAS_HEIGHT = 128
SLICE_L = 8

# Prefer by-var packs; exchange only has the non-byvar leaf locally/cluster.
DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "ETTh1": {
        "pack": "results/datasets/08-04-1843-ablation-disc-l8-l16-ETTh1-c128-valtest80-byvar",
        "ckpt": "results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
    },
    "ETTh2": {
        "pack": "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar",
        "ckpt": "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml",
    },
    "electricity": {
        "pack": "results/datasets/08-04-1845-ablation-disc-l8-l16-electricity-c128-valtest80-byvar",
        "ckpt": "results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml",
    },
    "traffic": {
        "pack": "results/datasets/08-04-1845-ablation-disc-l8-l16-traffic-c128-valtest80-byvar",
        "ckpt": "results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml",
    },
    "exchange_rate": {
        "pack": "results/datasets/08-04-1545-ablation-disc-l8-l16-exchange_rate-c128-valtest80",
        "ckpt": "results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml",
    },
}

DISAGREE_RE = re.compile(
    r"_L(?P<L>\d+)_.*_w(?P<w>\d+)_off(?P<off>\d+)_v(?P<v>\d+)_"
)


@dataclass
class PanelPick:
    local: int
    pool: int
    variate: int
    offset: int
    reason: str
    mae: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS.keys()),
        help="Datasets to viz (skip silently only when --allow-skip).",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--n-windows", type=int, default=6, help="Target distinct windows.")
    p.add_argument(
        "--variates",
        type=int,
        nargs="+",
        default=None,
        help="Restrict panels to these variate indices (e.g. 5 for ETTh2 LULL).",
    )
    p.add_argument(
        "--slice-lengths",
        type=int,
        nargs="+",
        default=[8],
        help="Snapproof segment lengths after bin_center_shift (default: L=8 only).",
    )
    p.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="Override pack dir for a single-dataset run.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Override ckpt dir for a single-dataset run.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config path for snap (needed for hybrid flat leaves).",
    )
    p.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=None,
        help="L8 offsets (default: mid + a few interesting / disagree offs).",
    )
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--allow-skip",
        action="store_true",
        help="Skip datasets with missing pack/raw instead of failing.",
    )
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    args.datasets = [d for raw in args.datasets for d in str(raw).split(",") if d]
    args.output_dir = args.output_dir.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    if args.pack is not None:
        args.pack = args.pack.expanduser().resolve()
    if args.ckpt is not None:
        args.ckpt = args.ckpt.expanduser().resolve()
    if args.smoke_test:
        args.datasets = args.datasets[:1]
        args.n_windows = min(args.n_windows, 2)
    if (args.pack is not None or args.ckpt is not None or args.config is not None):
        if len(args.datasets) != 1:
            raise ValueError("--pack/--ckpt/--config require exactly one --datasets entry")
    return args


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _find_binary_mmpd(raw_dir: Path, dataset: str) -> Tuple[Path, Path]:
    binary_hits = sorted(raw_dir.glob(f"binary_*{dataset}*.npz"))
    mmpd_hits = sorted(
        p for p in raw_dir.glob(f"mmpd_{dataset}*.npz") if "indices" not in p.name
    )
    # Prefer *-val-test.npz when both full and val-test exist.
    def prefer_valtest(paths: Sequence[Path]) -> Path:
        vt = [p for p in paths if "val-test" in p.name]
        if vt:
            return vt[0]
        if not paths:
            raise FileNotFoundError(f"no pack under {raw_dir} for {dataset}")
        return paths[0]

    return prefer_valtest(binary_hits), prefer_valtest(mmpd_hits)


def _mid_offset(horizon: int, slice_len: int = SLICE_L) -> int:
    return max(0, (horizon - slice_len) // 2)


def _parse_disagreement_picks(
    pack_dir: Path,
    *,
    indices: np.ndarray,
    slice_len: int = SLICE_L,
    max_per_kind: int = 8,
) -> List[Tuple[int, int, int, str]]:
    """Return (pool_i, offset, variate, reason) from disc-disagreement PNGs."""
    root = pack_dir / "viz" / "disc_disagreement" / RUN_NAME
    if not root.is_dir():
        return []
    pool_to_local = {int(p): i for i, p in enumerate(indices.tolist())}
    out: List[Tuple[int, int, int, str]] = []
    for kind in ("mmpd_wrong_binary_right", "binary_wrong_mmpd_right"):
        n = 0
        for path in sorted(root.glob(f"*L{slice_len}*{kind}*.png")):
            m = DISAGREE_RE.search(path.name)
            if m is None:
                continue
            if int(m.group("L")) != slice_len:
                continue
            pool_i = int(m.group("w"))
            if pool_i not in pool_to_local:
                continue
            out.append(
                (
                    pool_i,
                    int(m.group("off")),
                    int(m.group("v")),
                    f"disc_{kind}",
                )
            )
            n += 1
            if n >= max_per_kind:
                break
    return out


def _top_mae_pairs(
    gt: np.ndarray,
    mmpd: np.ndarray,
    indices: np.ndarray,
    *,
    n: int,
    seed: int,
) -> List[Tuple[int, int, float]]:
    """Top (local, variate, mae) by |MMPD-GT|, with per-variate coverage."""
    err = np.mean(np.abs(mmpd - gt), axis=-1)  # (N, V)
    n_win, n_var = err.shape
    flat = [
        (float(err[i, v]), int(i), int(v))
        for i in range(n_win)
        for v in range(n_var)
    ]
    flat.sort(reverse=True)
    picked: List[Tuple[int, int, float]] = []
    seen_windows: set[int] = set()
    seen_vars: set[int] = set()
    # First: one best window per variate.
    for mae, i, v in flat:
        if v in seen_vars:
            continue
        picked.append((i, v, mae))
        seen_windows.add(i)
        seen_vars.add(v)
        if len(seen_vars) >= n_var:
            break
    # Then fill remaining slots with highest-MAE unseen windows.
    for mae, i, v in flat:
        if len(picked) >= n:
            break
        if i in seen_windows:
            continue
        picked.append((i, v, mae))
        seen_windows.add(i)
    if len(picked) < n:
        rng = np.random.default_rng(seed)
        while len(picked) < n and len(seen_windows) < n_win:
            i = int(rng.integers(0, n_win))
            if i in seen_windows:
                continue
            v = int(np.argmax(err[i]))
            picked.append((i, v, float(err[i, v])))
            seen_windows.add(i)
    return picked[:n]


def _select_panels(
    *,
    pack_dir: Path,
    snapped: Mapping[str, np.ndarray],
    n_windows: int,
    seed: int,
    extra_offsets: Optional[Sequence[int]],
    max_l8: int = 16,
    variates: Optional[Sequence[int]] = None,
) -> List[PanelPick]:
    gt = np.asarray(snapped["gt"])
    mmpd = np.asarray(snapped["mmpd"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    horizon = int(gt.shape[-1])
    n_var = int(gt.shape[1])
    allowed = (
        {int(v) for v in variates}
        if variates is not None
        else set(range(n_var))
    )
    bad = sorted(v for v in allowed if v < 0 or v >= n_var)
    if bad:
        raise ValueError(f"variates out of range V={n_var}: {bad}")
    mid = _mid_offset(horizon)
    default_offs = [mid, 8, 24, 60, 80]
    if extra_offsets is not None:
        default_offs = [int(x) for x in extra_offsets]
    default_offs = [o for o in default_offs if 0 <= o <= horizon - SLICE_L]
    if not default_offs:
        raise RuntimeError("no valid L8 offsets")

    mae_pairs = [
        (i, v, mae)
        for i, v, mae in _top_mae_pairs(gt, mmpd, indices, n=max(n_windows * n_var, n_windows), seed=seed)
        if v in allowed
    ]
    # Cap disagreement so L8 count stays readable (~1–2 offs × few windows).
    disagree = [
        row
        for row in _parse_disagreement_picks(
            pack_dir, indices=indices, max_per_kind=max(2, min(4, n_windows)),
        )
        if row[2] in allowed
    ]

    picks: List[PanelPick] = []
    used: set[Tuple[int, int, int]] = set()

    def add(local: int, variate: int, offset: int, reason: str, mae: float) -> bool:
        if variate not in allowed:
            return False
        if len(picks) >= max_l8:
            return False
        key = (local, variate, offset)
        if key in used:
            return False
        if not (0 <= offset <= horizon - SLICE_L):
            return False
        used.add(key)
        picks.append(
            PanelPick(
                local=local,
                pool=int(indices[local]),
                variate=variate,
                offset=offset,
                reason=reason,
                mae=mae,
            )
        )
        return True

    pool_to_local = {int(p): i for i, p in enumerate(indices.tolist())}

    def best_alt_offset(local: int, v: int) -> int:
        best_off, best_d = mid, -1.0
        for o in default_offs:
            d = float(
                np.mean(
                    np.abs(
                        mmpd[local, v, o : o + SLICE_L]
                        - gt[local, v, o : o + SLICE_L]
                    )
                )
            )
            if d > best_d:
                best_d, best_off = d, o
        return best_off

    err = np.mean(np.abs(mmpd - gt), axis=-1)

    # Variate coverage first so max_l8 cannot drop a whole channel.
    for v in sorted(allowed):
        local = int(np.argmax(err[:, v]))
        mae = float(err[local, v])
        add(local, v, mid, "variate_cover_mid", mae)
        alt = best_alt_offset(local, v)
        if alt != mid:
            add(local, v, alt, "variate_cover_alt", mae)

    # Disagreement (interesting non-mid offsets), then pair with mid.
    for pool_i, off, v, reason in disagree:
        local = pool_to_local[pool_i]
        mae = float(np.mean(np.abs(mmpd[local, v] - gt[local, v])))
        add(local, v, off, reason, mae)
        if off != mid:
            add(local, v, mid, f"{reason}+mid", mae)

    # Remaining high-|Δ| windows.
    for local, v, mae in mae_pairs:
        add(local, v, mid, "high_mae_mid", mae)
        alt = best_alt_offset(local, v)
        if alt != mid:
            add(local, v, alt, "high_mae_alt_off", mae)

    if not picks:
        raise RuntimeError("no panels selected")
    return picks


def _bin_center_slice(
    snapped: Mapping[str, np.ndarray],
    *,
    local: int,
    variate: int,
    offset: int,
    slice_len: int,
) -> Dict[str, np.ndarray]:
    levels = np.asarray(snapped["legal_levels"])
    out: Dict[str, np.ndarray] = {}
    for name, key in (("GT", "gt"), ("binary", "binary"), ("MMPD", "mmpd")):
        seg = np.asarray(snapped[key])[local, variate, offset : offset + slice_len]
        shifted, _ = bin_center_shift(
            seg[None, None, :],
            levels[local : local + 1, variate : variate + 1, :],
            reduce="per_variate",
        )
        out[name] = shifted[0, 0]
    return out


def _write_snapproof(
    *,
    out_dir: Path,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    picks: Sequence[PanelPick],
    slice_len: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"GT": "black", "binary": "#1f77b4", "MMPD": "#d62728"}
    levels = np.asarray(snapped["legal_levels"])
    paths: List[Path] = []
    for pick in picks:
        # Re-center offset if slice length differs from the pick's L8 mid default.
        offset = int(pick.offset)
        h = int(np.asarray(snapped["gt"]).shape[-1])
        if offset + slice_len > h:
            offset = max(0, (h - slice_len) // 2)
        series = _bin_center_slice(
            snapped,
            local=pick.local,
            variate=pick.variate,
            offset=offset,
            slice_len=slice_len,
        )
        path = out_dir / (
            f"{RUN_NAME}_{dataset}_v{pick.variate}_local{pick.local}_pool{pick.pool}_"
            f"L{slice_len}_off{offset}_snapproof.png"
        )
        title = (
            f"{RUN_NAME}/{dataset} pool={pick.pool} local={pick.local} "
            f"v={pick.variate} | disc L={slice_len} off={offset} "
            f"t=[{offset},{offset + slice_len}) AFTER bin_center_shift"
        )
        _plot_snap_proof_panel(
            out_path=path,
            title=title,
            levels_1d=levels[pick.local, pick.variate],
            series=series,
            colors=colors,
            t0=offset,
        )
        paths.append(path)
    return paths


def _write_horizon96(
    *,
    out_dir: Path,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    picks: Sequence[PanelPick],
    dpi: int,
) -> List[Path]:
    """One PNG per (window, variate): full H=96 GT / binary / MMPD."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    # Dedup (local, variate) — horizon plot ignores offset.
    seen: set[Tuple[int, int]] = set()
    paths: List[Path] = []
    for pick in picks:
        key = (pick.local, pick.variate)
        if key in seen:
            continue
        seen.add(key)
        local, v = pick.local, pick.variate
        pool = int(indices[local])
        t = np.arange(gt.shape[-1])
        fig, ax = plt.subplots(figsize=(11.0, 3.6))
        ax.plot(t, gt[local, v], color="black", lw=1.8, label="GT")
        ax.plot(t, binary[local, v], color="#1f77b4", lw=1.4, alpha=0.9, label="binary")
        ax.plot(t, mmpd[local, v], color="#d62728", lw=1.4, alpha=0.9, label="MMPD")
        mae_b = float(np.mean(np.abs(binary[local, v] - gt[local, v])))
        mae_m = float(np.mean(np.abs(mmpd[local, v] - gt[local, v])))
        ax.set_title(
            f"{RUN_NAME}/{dataset} pool={pool} local={local} v={v} | "
            f"H={gt.shape[-1]} snapped  "
            f"MAE(binary)={mae_b:.3g}  MAE(MMPD)={mae_m:.3g}",
            fontsize=10,
        )
        ax.set_xlabel("horizon step t")
        ax.set_ylabel("dataset-z (snapped)")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = out_dir / (
            f"{RUN_NAME}_{dataset}_v{v}_local{local}_pool{pool}_H{gt.shape[-1]}.png"
        )
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def _snap_dataset(
    *,
    dataset: str,
    pack_dir: Path,
    ckpt_root: Path,
    config_path: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    raw_dir = pack_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{dataset}: missing raw/ under {pack_dir}")
    binary_path, mmpd_path = _find_binary_mmpd(raw_dir, dataset)
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)
    if not np.array_equal(
        np.asarray(binary_pack["indices"]), np.asarray(mmpd_pack["indices"])
    ):
        raise RuntimeError(f"{dataset}: binary/MMPD indices differ")

    run, _stages, kind = load_ablation_run(dataset, ckpt_root)
    if kind != "patch_refine":
        raise RuntimeError(f"{dataset}: expected patch_refine, got {kind}")
    ladder = _ladder_only(
        dataset=dataset,
        run=run,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
    )
    canvas_height = CANVAS_HEIGHT
    if "canvas_height" in binary_pack:
        canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])
    if canvas_height != CANVAS_HEIGHT:
        raise RuntimeError(
            f"{dataset}: pack canvas_height={canvas_height}, expected {CANVAS_HEIGHT}"
        )

    snap_args = SimpleNamespace(
        fake_agg=args.fake_agg,
        mmpd_data_dir=args.mmpd_data_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        dataset=dataset,
    )
    print(
        f"[{dataset}] snap N={binary_pack['y_true'].shape[0]} "
        f"V={binary_pack['y_true'].shape[1]} vars={run_variate_indices(run)} "
        f"binary={binary_path.name} mmpd={mmpd_path.name}",
        flush=True,
    )
    snapped = _snap_bundle(
        binary_pack=binary_pack,
        mmpd_pack=mmpd_pack,
        run=run,
        ladder=ladder,
        args=snap_args,
        device=device,
        canvas_height=canvas_height,
        ckpt_root=ckpt_root,
        config_path=config_path,
    )
    # Sanity: reduce matches pack sample0.
    _ = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    return snapped


def _write_readme(
    out_dir: Path,
    *,
    used: Mapping[str, Dict[str, Any]],
    skipped: Mapping[str, str],
) -> None:
    lines = [
        "# MMPD forecast visualizations (canvas128 packs)",
        "",
        "Generated by `temp/scripts/viz_mmpd_all_datasets_forecasts.py`.",
        "",
        "## Layout",
        "",
        "- `{dataset}/L8_snapproof/` — occupied-rung snapproof after `bin_center_shift`",
        "- `{dataset}/horizon96/` — full H=96 GT / binary / MMPD (snapped dataset-z)",
        "",
        "## Packs used",
        "",
    ]
    for ds, info in used.items():
        lines.append(f"### {ds}")
        lines.append(f"- pack: `{info['pack']}`")
        lines.append(f"- ckpt metadata: `{info['ckpt']}`")
        lines.append(f"- binary: `{info['binary']}`")
        lines.append(f"- mmpd: `{info['mmpd']}`")
        lines.append(
            f"- plots: L8_snapproof={info['n_l8']}  horizon96={info['n_h96']}"
        )
        lines.append("")
    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for ds, reason in skipped.items():
            lines.append(f"- **{ds}**: {reason}")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available()
        else f"cuda:{int(args.gpu)}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    used: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}

    for dataset in args.datasets:
        if dataset not in DATASET_SPECS:
            raise KeyError(f"unknown dataset {dataset!r}; known={list(DATASET_SPECS)}")
        spec = DATASET_SPECS[dataset]
        pack_dir = (
            args.pack
            if args.pack is not None
            else (REPO_ROOT / spec["pack"]).resolve()
        )
        ckpt_root = (
            args.ckpt
            if args.ckpt is not None
            else (REPO_ROOT / spec["ckpt"]).resolve()
        )
        config_path = args.config if args.config is not None else spec["config"]
        try:
            if not pack_dir.is_dir():
                raise FileNotFoundError(f"pack missing: {pack_dir}")
            if not ckpt_root.is_dir():
                raise FileNotFoundError(f"ckpt missing: {ckpt_root}")
            binary_path, mmpd_path = _find_binary_mmpd(pack_dir / "raw", dataset)
            snapped = _snap_dataset(
                dataset=dataset,
                pack_dir=pack_dir,
                ckpt_root=ckpt_root,
                config_path=config_path,
                args=args,
                device=device,
            )
            picks = _select_panels(
                pack_dir=pack_dir,
                snapped=snapped,
                n_windows=int(args.n_windows),
                seed=int(args.seed) + hash(dataset) % 10_000,
                extra_offsets=args.offsets,
                variates=args.variates,
            )
            ds_root = args.output_dir / dataset
            slice_counts: Dict[str, int] = {}
            for L in args.slice_lengths:
                L = int(L)
                paths = _write_snapproof(
                    out_dir=ds_root / f"L{L}_snapproof",
                    dataset=dataset,
                    snapped=snapped,
                    picks=picks,
                    slice_len=L,
                )
                slice_counts[f"n_l{L}"] = len(paths)
            h96_paths = _write_horizon96(
                out_dir=ds_root / "horizon96",
                dataset=dataset,
                snapped=snapped,
                picks=picks,
                dpi=int(args.dpi),
            )
            used[dataset] = {
                "pack": str(pack_dir.relative_to(REPO_ROOT))
                if pack_dir.is_relative_to(REPO_ROOT)
                else str(pack_dir),
                "ckpt": str(ckpt_root.relative_to(REPO_ROOT))
                if ckpt_root.is_relative_to(REPO_ROOT)
                else str(ckpt_root),
                "config": config_path,
                "binary": binary_path.name,
                "mmpd": mmpd_path.name,
                **slice_counts,
                "n_h96": len(h96_paths),
                "n_picks": len(picks),
                "variates": list(args.variates) if args.variates is not None else "all",
            }
            print(
                f"[{dataset}] wrote {slice_counts} + H96={len(h96_paths)} "
                f"from {len(picks)} picks → {ds_root}",
                flush=True,
            )
        except Exception as exc:
            if args.allow_skip:
                skipped[dataset] = f"{type(exc).__name__}: {exc}"
                print(f"[{dataset}] SKIP: {skipped[dataset]}", flush=True)
                continue
            raise

    _write_readme(args.output_dir, used=used, skipped=skipped)
    summary = {"used": used, "skipped": skipped, "output_dir": str(args.output_dir)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not used:
        raise RuntimeError("no datasets produced plots")


if __name__ == "__main__":
    main()
