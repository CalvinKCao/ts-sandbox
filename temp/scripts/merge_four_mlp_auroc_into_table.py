#!/usr/bin/env python3
"""Merge PeMS/solar/ETTm1/ETTm2 mlp auroc_table.json into full_metrics_table.json."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LOCAL = Path("/home/cao/ts-sandbox/temp/lean_disc_c128_results")
WANT = ["PeMS", "solar_Alabama", "ETTm1", "ETTm2"]


def main() -> None:
    table_path = LOCAL / "full_metrics_table.json"
    table = json.loads(table_path.read_text())
    by = {d["dataset"]: d for d in table["datasets"]}
    for ds in WANT:
        at = LOCAL / ds / "auroc_table.json"
        if not at.exists():
            print("skip missing", ds)
            continue
        raw = json.loads(at.read_text())
        if not isinstance(raw, list):
            print("unexpected format", ds, type(raw))
            continue
        paired: dict[tuple[str, int], dict] = defaultdict(dict)
        for r in raw:
            if r.get("tag") and r["tag"] != "all_variates":
                continue
            arch = r.get("arch", "mlp")
            L = int(r.get("slice_len") or r.get("L"))
            src = r.get("source", "")
            key = (arch, L)
            if src in ("binary_staged", "binary"):
                paired[key]["binary_auroc"] = r.get("disc_auroc")
            elif src == "mmpd":
                paired[key]["mmpd_auroc"] = r.get("disc_auroc")
        rows = [
            {
                "arch": a,
                "L": L,
                "binary_auroc": v.get("binary_auroc"),
                "mmpd_auroc": v.get("mmpd_auroc"),
            }
            for (a, L), v in sorted(paired.items(), key=lambda x: (x[0][0], x[0][1]))
        ]
        stamp = None
        sm = LOCAL / ds / "summary.json"
        if sm.exists():
            try:
                s = json.loads(sm.read_text())
                od = s.get("out_dir") or s.get("output_dir") or s.get("pack")
                if od:
                    stamp = "-".join(Path(str(od)).name.split("-")[:3])
            except Exception as e:
                print("summary parse", e)
        if not stamp:
            stamp = datetime.fromtimestamp(at.stat().st_mtime).strftime("%m-%d-%H%M")
        entry = by[ds]
        entry["disc_auroc"] = {
            "pack_stamp": stamp,
            "status": "COMPLETED",
            "rows": rows,
            "source": str(at),
            "job_note": "mlp ablation only",
        }
        entry.pop("mmpd_hz96_pending_disc", None)
        print(ds, "rows", len(rows), "stamp", stamp)

    table["protocol"]["mmpd_new4"] = (
        "hz96 MMPD COMPLETED 08-06 matched-c128-four; "
        "mlp disc ablation COMPLETED (4628322/4628324/4628348/4628388[+resubs])"
    )
    table["protocol"]["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    print("wrote", table_path)
    print("\nMLP AUROC:")
    print(f"{'dataset':16} {'L':>3} {'binary':>10} {'mmpd':>10}")
    for ds in WANT:
        for r in by[ds].get("disc_auroc", {}).get("rows", []):
            if r.get("arch") != "mlp":
                continue
            b, m = r.get("binary_auroc"), r.get("mmpd_auroc")
            print(f"{ds:16} {r['L']:3d} {b:10.6f} {m:10.6f}")
    print("\nviz paths:")
    for ds in WANT:
        vp = LOCAL / "viz" / ds
        n = sum(1 for _ in vp.rglob("*")) if vp.is_dir() else 0
        print(f"  {vp} exists={vp.is_dir()} n={n}")


if __name__ == "__main__":
    main()
