# Pipeline integration: Prefer --viz-sanity snap + --viz-variates 5 on hybrid_flat snap_mode.
#!/usr/bin/env python3
"""LULL (ETTh2 v=5) forecast viz from hybrid-flat-dsnorm canvas128 packs.

Snaps GT / binary / MMPD with ``_snap_bundle`` (hybrid flat mask from ckpt
metadata — LULL skips window-norm). Writes:

- ``horizon96/`` — full H=96 GT vs binary vs MMPD
- ``L8_snapproof/`` / ``L16_snapproof/`` — occupied-rung panels AFTER bin_center_shift

Default sources (Killarney job 4609805 train + 4614062 disc):
  pack  results/datasets/08-05-1057-ablation-disc-l8-l16-ETTh2-c128-hybrid-flat-dsnorm-valtest80-byvar
  ckpt  results/ckpts/08-05-4609805-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

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
from utils.eval_mmpd_gaussian_anchor import DEFAULT_MMPD_DATA  # noqa: E402
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402

LULL_VARIATE = 5
DATASET = "ETTh2"
RUN_NAME = "hybrid_flat_dsnorm"
DEFAULT_PACK = (
    REPO_ROOT
    / "results/datasets/08-05-1057-ablation-disc-l8-l16-ETTh2-c128-hybrid-flat-dsnorm-valtest80-byvar"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-05-4609805-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm"
)
DEFAULT_CFG = (
    "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm.yaml"
)
DEFAULT_OUT = REPO_ROOT / "temp" / "viz_lull_etth2_hybrid"
# Known disc-disagreement window for LULL in the 1057 pack.
PREFERRED_POOLS = (1116,)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-root", type=Path, default=DEFAULT_PACK)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--variate", type=int, default=LULL_VARIATE)
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--pools", type=int, nargs="+", default=None,
                   help="Force pool indices (else prefer disagree + top MAE).")
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260805)
    args = p.parse_args()
    args.pack_root = args.pack_root.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    return args


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _find_pack(raw_dir: Path, prefix: str, dataset: str) -> Path:
    hits = sorted(p for p in raw_dir.glob(f"{prefix}*{dataset}*.npz") if "indices" not in p.name)
    vt = [p for p in hits if "val-test" in p.name]
    if vt:
        return vt[0]
    if not hits:
        raise FileNotFoundError(f"no {prefix} pack for {dataset} under {raw_dir}")
    return hits[0]


def _select_locals(
    *,
    indices: np.ndarray,
    gt: np.ndarray,
    mmpd: np.ndarray,
    variate: int,
    n_windows: int,
    forced_pools: Sequence[int] | None,
    seed: int,
) -> List[int]:
    """Return local indices into the pack for LULL panels."""
    n = int(gt.shape[0])
    pool_to_local = {int(indices[i]): i for i in range(n)}
    chosen: List[int] = []
    seen: set[int] = set()

    def add_pool(pool: int) -> None:
        local = pool_to_local.get(int(pool))
        if local is None or local in seen:
            return
        seen.add(local)
        chosen.append(local)

    if forced_pools:
        for pool in forced_pools:
            add_pool(int(pool))
    else:
        for pool in PREFERRED_POOLS:
            add_pool(int(pool))

    mae = np.mean(np.abs(mmpd[:, variate] - gt[:, variate]), axis=-1)
    order = np.argsort(-mae)
    for local in order.tolist():
        if len(chosen) >= n_windows:
            break
        if local in seen:
            continue
        seen.add(local)
        chosen.append(int(local))

    if len(chosen) < n_windows:
        rng = np.random.default_rng(seed)
        rest = [i for i in range(n) if i not in seen]
        need = min(n_windows - len(chosen), len(rest))
        if need:
            chosen.extend(rng.choice(rest, size=need, replace=False).tolist())
    return chosen[:n_windows]


def _bin_center_slice(
    snapped: Dict[str, np.ndarray],
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


def _write_horizon96(
    *,
    out_dir: Path,
    snapped: Dict[str, np.ndarray],
    locals_: Sequence[int],
    variate: int,
    dpi: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    paths: List[Path] = []
    for local in locals_:
        pool = int(indices[local])
        t = np.arange(gt.shape[-1])
        fig, ax = plt.subplots(figsize=(11.0, 3.6))
        ax.plot(t, gt[local, variate], color="black", lw=1.8, label="GT")
        ax.plot(t, binary[local, variate], color="#1f77b4", lw=1.4, alpha=0.9, label="binary")
        ax.plot(t, mmpd[local, variate], color="#d62728", lw=1.4, alpha=0.9, label="MMPD")
        mae_b = float(np.mean(np.abs(binary[local, variate] - gt[local, variate])))
        mae_m = float(np.mean(np.abs(mmpd[local, variate] - gt[local, variate])))
        ax.set_title(
            f"{RUN_NAME}/{DATASET} LULL v={variate} pool={pool} local={local} | "
            f"H={gt.shape[-1]} snapped ({snapped.get('snap_mode', '?')})  "
            f"MAE(binary)={mae_b:.3g}  MAE(MMPD)={mae_m:.3g}",
            fontsize=10,
        )
        ax.set_xlabel("horizon step t")
        ax.set_ylabel("dataset-z (snapped)")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = out_dir / (
            f"{RUN_NAME}_{DATASET}_LULL_v{variate}_local{local}_pool{pool}_H{gt.shape[-1]}.png"
        )
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_snapproof(
    *,
    out_dir: Path,
    snapped: Dict[str, np.ndarray],
    locals_: Sequence[int],
    variate: int,
    slice_len: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"GT": "black", "binary": "#1f77b4", "MMPD": "#d62728"}
    levels = np.asarray(snapped["legal_levels"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    h = int(np.asarray(snapped["gt"]).shape[-1])
    offset = max(0, (h - slice_len) // 2)
    paths: List[Path] = []
    for local in locals_:
        pool = int(indices[local])
        series = _bin_center_slice(
            snapped, local=local, variate=variate, offset=offset, slice_len=slice_len,
        )
        path = out_dir / (
            f"{RUN_NAME}_{DATASET}_LULL_v{variate}_local{local}_pool{pool}_"
            f"L{slice_len}_off{offset}_snapproof.png"
        )
        title = (
            f"{RUN_NAME}/{DATASET} LULL v={variate} pool={pool} local={local} | "
            f"L={slice_len} off={offset} AFTER bin_center_shift | "
            f"snap={snapped.get('snap_mode', '?')}"
        )
        _plot_snap_proof_panel(
            out_path=path,
            title=title,
            levels_1d=levels[local, variate],
            series=series,
            colors=colors,
            t0=offset,
        )
        paths.append(path)
    return paths


def _write_readme(
    out_dir: Path,
    *,
    meta: Dict[str, Any],
    h96: Sequence[Path],
    l8: Sequence[Path],
    l16: Sequence[Path],
) -> None:
    lines = [
        "# LULL (ETTh2 v=5) hybrid-flat-dsnorm forecast viz",
        "",
        "Generated by `temp/scripts/viz_lull_etth2_hybrid.py`.",
        "",
        "## Sources",
        "",
        f"- pack: `{meta['pack']}`",
        f"- ckpt: `{meta['ckpt']}` (train job **4609805**)",
        f"- disc pack leaf: `08-05-1057-…hybrid-flat-dsnorm-valtest80-byvar` (job **4614062**)",
        f"- config: `{meta['config']}`",
        f"- snap_mode: `{meta['snap_mode']}` (LULL is flat → dataset affine only)",
        "",
        "## Layout",
        "",
        "- `horizon96/` — full H=96 GT / binary / MMPD (snapped dataset-z)",
        "- `L8_snapproof/` — mid-horizon L=8 AFTER `bin_center_shift`",
        "- `L16_snapproof/` — mid-horizon L=16 AFTER `bin_center_shift`",
        "",
        f"## Panels (variate={meta['variate']} LULL)",
        "",
        f"- H96: {len(h96)}",
        f"- L8: {len(l8)}",
        f"- L16: {len(l16)}",
        f"- pools: {meta['pools']}",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{int(args.gpu)}"
    )
    raw_dir = args.pack_root / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw/ under {args.pack_root}")
    if not args.ckpt.is_dir():
        raise FileNotFoundError(f"missing ckpt: {args.ckpt}")

    binary_path = _find_pack(raw_dir, "binary_", DATASET)
    mmpd_path = _find_pack(raw_dir, "mmpd_", DATASET)
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)

    run, _stages, kind = load_ablation_run(DATASET, args.ckpt)
    if kind != "patch_refine":
        raise RuntimeError(f"expected patch_refine, got {kind}")
    ladder = _ladder_only(
        dataset=DATASET, run=run, lookback=int(args.lookback), horizon=int(args.horizon),
    )
    canvas_height = 128
    if "canvas_height" in binary_pack:
        canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])

    snap_args = SimpleNamespace(
        fake_agg=args.fake_agg,
        mmpd_data_dir=args.mmpd_data_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        dataset=DATASET,
    )
    print(
        f"[LULL] snap N={binary_pack['y_true'].shape[0]} V={binary_pack['y_true'].shape[1]} "
        f"binary={binary_path.name} mmpd={mmpd_path.name} device={device}",
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
        ckpt_root=args.ckpt,
        config_path=args.config,
    )
    # Touch reduce path so packs are exercised the same way as ladder viz.
    _ = reduce_pack_forecast(binary_pack, agg=args.fake_agg)

    v = int(args.variate)
    if v < 0 or v >= snapped["gt"].shape[1]:
        raise ValueError(f"variate={v} out of range V={snapped['gt'].shape[1]}")

    locals_ = _select_locals(
        indices=np.asarray(snapped["indices"], dtype=np.int64),
        gt=np.asarray(snapped["gt"]),
        mmpd=np.asarray(snapped["mmpd"]),
        variate=v,
        n_windows=int(args.n_windows),
        forced_pools=args.pools,
        seed=int(args.seed),
    )
    pools = [int(snapped["indices"][i]) for i in locals_]
    print(f"[LULL] locals={locals_} pools={pools} snap_mode={snapped.get('snap_mode')}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h96 = _write_horizon96(
        out_dir=args.output_dir / "horizon96",
        snapped=snapped,
        locals_=locals_,
        variate=v,
        dpi=int(args.dpi),
    )
    l8: List[Path] = []
    l16: List[Path] = []
    for L in args.slice_lengths:
        L = int(L)
        dest = args.output_dir / f"L{L}_snapproof"
        paths = _write_snapproof(
            out_dir=dest,
            snapped=snapped,
            locals_=locals_,
            variate=v,
            slice_len=L,
        )
        if L == 8:
            l8 = paths
        elif L == 16:
            l16 = paths

    meta = {
        "pack": str(args.pack_root.relative_to(REPO_ROOT))
        if args.pack_root.is_relative_to(REPO_ROOT)
        else str(args.pack_root),
        "ckpt": str(args.ckpt.relative_to(REPO_ROOT))
        if args.ckpt.is_relative_to(REPO_ROOT)
        else str(args.ckpt),
        "config": args.config,
        "binary": binary_path.name,
        "mmpd": mmpd_path.name,
        "variate": v,
        "variate_name": "LULL",
        "snap_mode": snapped.get("snap_mode"),
        "locals": locals_,
        "pools": pools,
        "n_h96": len(h96),
        "n_l8": len(l8),
        "n_l16": len(l16),
    }
    _write_readme(args.output_dir, meta=meta, h96=h96, l8=l8, l16=l16)
    (args.output_dir / "summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
