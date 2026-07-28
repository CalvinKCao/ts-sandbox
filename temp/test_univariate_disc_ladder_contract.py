#!/usr/bin/env python3
"""Fast contract checks for vertical-dual discriminator sampling and ladder math."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.ordinal_window_norm import build_global_ladder_from_training
from utils.dual_scale_bin_filter import (
    apply_dual_scale_bin_filter,
    assert_on_binary_dual_ordinal_lattice,
    binary_dual_decode_levels_dataset_z,
    run_self_test,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import generate_staged_forecast


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    def generate(self, past: torch.Tensor, **kwargs):
        self.calls.append(dict(kwargs))
        if self.name == "coarse":
            return {"future_2d_coarse": torch.ones(1, 1, 16, 4)}
        if self.name == "fine":
            assert "future_coarse_2d" in kwargs
            return {"prediction_global_norm": torch.full((1, 1, 4), 2.0)}
        assert "future_coarse_2d" not in kwargs
        return {"prediction_global_norm": torch.full((1, 1, 4), 3.0)}


def _write_synthetic_256_bin_visualization() -> Path:
    """Run synthetic GT/binary/MMPD through binary's exact 256-bin decode path."""
    rng = np.random.default_rng(17)
    rank_axis = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    # A deliberately nonuniform continuous dataset-z ladder: this catches the
    # invalid extra global-nearest-ladder snap that used to move legal values.
    train = (rank_axis**3 * 2.7 + 0.12 * np.sin(rank_axis * 19.0)).reshape(-1, 1)
    ladder = build_global_ladder_from_training(train, tie_atol=1e-7)
    past = (0.42 * np.sin(np.linspace(-6.0, 0.0, 96, dtype=np.float32))).reshape(1, 1, -1)
    horizon = 0.7 * np.sin(np.linspace(0.0, 5.0, 96, dtype=np.float32))
    gt_raw = horizon.reshape(1, 1, -1)
    binary_raw = gt_raw + 0.10 * np.sin(np.linspace(0.0, 13.0, 96, dtype=np.float32)).reshape(1, 1, -1)
    mmpd_raw = gt_raw + rng.normal(0.0, 0.12, size=gt_raw.shape).astype(np.float32)
    device = torch.device("cpu")
    common = {
        "past": past,
        "ladder": ladder,
        "coarse_height": 16,
        "fine_height": 16,
        "decoder": "mean",
        "device": device,
        "repr_time_stride": 1,
    }
    gt = apply_dual_scale_bin_filter(gt_raw, **common)
    binary = apply_dual_scale_bin_filter(binary_raw, **common)
    mmpd = apply_dual_scale_bin_filter(mmpd_raw, **common)
    for name, values in (("GT", gt), ("binary", binary), ("MMPD", mmpd)):
        stats = assert_on_binary_dual_ordinal_lattice(
            values,
            past,
            ladder=ladder,
            coarse_height=16,
            fine_height=16,
            device=device,
        )
        if stats["max_decode_delta"] != 0.0:
            raise AssertionError(f"{name} did not land exactly on a decoded binary bin")
    levels = binary_dual_decode_levels_dataset_z(
        past,
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        device=device,
    )[0, 0]
    if np.unique(levels).size != 256:
        raise AssertionError("synthetic ladder must yield 256 distinct dataset-z decode values")

    fig, (forecast_ax, ladder_ax) = plt.subplots(1, 2, figsize=(15, 5.2))
    forecast_x = np.arange(gt.shape[-1])
    past_x = np.arange(-past.shape[-1], 0)
    forecast_ax.hlines(levels, past_x[0], forecast_x[-1], color="0.55", lw=0.25, alpha=0.16, zorder=0)
    forecast_ax.plot(past_x, past[0, 0], color="0.55", lw=1.25, label="lookback")
    forecast_ax.plot(forecast_x, gt[0, 0], color="black", lw=1.3, label="GT snapped")
    forecast_ax.plot(forecast_x, binary[0, 0], color="#1f77b4", lw=1.05, label="binary snapped")
    forecast_ax.plot(forecast_x, mmpd[0, 0], color="#ff7f0e", lw=1.05, label="MMPD snapped")
    forecast_ax.axvline(0, color="0.2", lw=0.8)
    forecast_ax.set_title("All 256 binary dataset-z decode values are the faint grid")
    forecast_ax.set_xlabel("time step (forecast starts at 0)")
    forecast_ax.set_ylabel("binary dataset-z")
    forecast_ax.legend(loc="upper left", ncol=2)
    forecast_ax.grid(alpha=0.15)

    ladder_ax.scatter(np.arange(256), levels, s=10, color="#1f77b4")
    ladder_ax.set_title("Ordinal 16×16 bin ID → continuous binary dataset-z")
    ladder_ax.set_xlabel("bin path ID")
    ladder_ax.set_ylabel("decoded dataset-z value")
    ladder_ax.grid(alpha=0.2)
    fig.tight_layout()
    out_dir = REPO_ROOT / "reports" / "univariate_disc_ladder_contract"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_256_bin_snap.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote synthetic 256-bin visualization: {out_path}")
    return out_path


def main() -> None:
    run_self_test()
    _write_synthetic_256_bin_visualization()
    past = torch.zeros(1, 1, 8)

    vertical = _FakeModel("vertical")
    out = generate_staged_forecast(
        vertical, vertical, past, vertical_dual=True, sampler="anchor"
    )
    assert len(vertical.calls) == 1
    assert float(out["prediction_global_norm"].mean()) == 3.0

    coarse = _FakeModel("coarse")
    fine = _FakeModel("fine")
    out = generate_staged_forecast(
        coarse, fine, past, vertical_dual=False, sampler="anchor", fine_seed=17
    )
    assert len(coarse.calls) == len(fine.calls) == 1
    assert float(out["prediction_global_norm"].mean()) == 2.0
    print("univariate discriminator ladder contract test ok")


if __name__ == "__main__":
    main()
