"""De-bias staged binary fakes for discriminator texture eval.

Staged binary decode lands on a dual-scale occupancy lattice. Flat plateaus are
left alone; other timesteps get sub-fine-bin jitter so the discriminator cannot
trivially separate real vs fake from quantization alone.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
DEFAULT_COARSE_HEIGHT = 16
DEFAULT_FINE_HEIGHT = 16


def fine_bin_width(max_scale: float, coarse_height: int, fine_height: int) -> float:
    """Width of one fine-stage bin in global-norm space (see TimeSeriesTo2D.decode_dual)."""
    return 2.0 * float(max_scale) / (int(coarse_height) * int(fine_height))


def fine_bin_half_width(max_scale: float, coarse_height: int, fine_height: int) -> float:
    return fine_bin_width(max_scale, coarse_height, fine_height) / 2.0


def resolve_dual_scale_bin_params(
    dataset: str,
    sub: Mapping[str, Any],
    *,
    fallback_max_scale: float,
    coarse_height: int = DEFAULT_COARSE_HEIGHT,
    fine_height: int = DEFAULT_FINE_HEIGHT,
) -> Tuple[float, int, int]:
    fine_meta = dict(sub.get("fine_metadata") or {})
    tuned = dict(fine_meta.get("tuned_params") or {})
    max_scale = tuned.get("max_scale")
    if max_scale is None:
        from models.diffusion_tsf.pipeline.config import load_experiment_config

        cfg = load_experiment_config(str(REPO_ROOT / "configs/base/binary_staged.yaml"))
        ms_map = dict(cfg.get("experiment", {}).get("max_scale_by_dataset") or {})
        max_scale = ms_map.get(dataset, fallback_max_scale)
    return (
        float(max_scale),
        int(fine_meta.get("coarse_image_height") or coarse_height),
        int(fine_meta.get("fine_image_height") or fine_height),
    )


def flatline_mask(y: np.ndarray, *, atol: float = 0.0) -> np.ndarray:
    """True on timesteps that share a value with an immediate neighbor (plateaus)."""
    arr = np.asarray(y)
    out = np.zeros(arr.shape, dtype=bool)
    if arr.shape[-1] < 2:
        return out
    same = np.abs(arr[..., 1:] - arr[..., :-1]) <= atol
    out[..., :-1] |= same
    out[..., 1:] |= same
    return out


def debias_binary_staged_fakes(
    fakes: np.ndarray,
    *,
    max_scale: float,
    coarse_height: int,
    fine_height: int,
    seed: int,
    dataset: str = "",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Add clipped Gaussian jitter on non-flatline timesteps (±½ fine bin)."""
    src = np.asarray(fakes, dtype=np.float32)
    half = fine_bin_half_width(max_scale, coarse_height, fine_height)
    plateau = flatline_mask(src, atol=0.0)
    debias = ~plateau

    # Stable per-window noise; dataset string spreads seeds across runs.
    seed_u = int(np.uint32(seed) ^ (hash(dataset) & 0xFFFFFFFF))
    rng = np.random.default_rng(seed_u)
    noise = rng.normal(0.0, half / 2.0, size=src.shape).astype(np.float32)
    np.clip(noise, -half, half, out=noise)

    out = src.copy()
    out[debias] += noise[debias]

    return out, {
        "fine_bin_width": float(2.0 * half),
        "half_fine_bin": float(half),
        "flatline_frac": float(plateau.mean()),
        "debias_frac": float(debias.mean()),
    }
