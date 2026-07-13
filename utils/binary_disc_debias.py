"""De-bias / quantize helpers for discriminator texture eval.

Staged binary decode lands on a dual-scale occupancy lattice. Optional jitter
(disabled by default for ordinal campaigns) was used to blunt trivial lattice
cues. MMPD continuous preds can be snapped to the same global ordinal ladder
rungs binary uses after decode.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

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
    import zlib

    src = np.asarray(fakes, dtype=np.float32)
    half = fine_bin_half_width(max_scale, coarse_height, fine_height)
    plateau = flatline_mask(src, atol=0.0)
    debias = ~plateau

    ds_tag = zlib.crc32(str(dataset).encode("utf-8")) & 0xFFFFFFFF
    seed_u = int(np.uint32(seed) ^ np.uint32(ds_tag))
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


def quantize_to_ordinal_ladder(
    values: np.ndarray,
    ladder: Any,
    *,
    batch_chunk: int = 64,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Snap continuous values to nearest global-ordinal ladder rung (binary decode bins).

    Accepts ``(N, V, T)`` or ``(N, V, S, T)``. Returns float32 array in the same
    shape plus stats (fraction of timesteps that moved, mean |delta|).
    """
    from models.diffusion_tsf.ordinal_window_norm import decode_with_ladder, encode_with_ladder

    src = np.asarray(values, dtype=np.float32)
    if src.ndim == 3:
        n, v, t = src.shape
        flat = src
        sample_axis = False
    elif src.ndim == 4:
        n, v, s, t = src.shape
        flat = src.transpose(0, 2, 1, 3).reshape(n * s, v, t)
        sample_axis = True
    else:
        raise ValueError(f"expected (N,V,T) or (N,V,S,T), got {src.shape}")

    if int(ladder.values.shape[1]) != int(flat.shape[1]):
        raise ValueError(
            f"ladder variates {int(ladder.values.shape[1])} != array variates {flat.shape[1]}"
        )

    chunks = []
    for start in range(0, flat.shape[0], max(1, int(batch_chunk))):
        end = min(flat.shape[0], start + max(1, int(batch_chunk)))
        x = torch.from_numpy(flat[start:end])
        ranks = encode_with_ladder(x, ladder)
        snapped = decode_with_ladder(ranks, ladder)
        chunks.append(snapped.detach().cpu().numpy().astype(np.float32))
    out_flat = np.concatenate(chunks, axis=0)
    delta = np.abs(out_flat - flat)
    changed = delta > 0
    stats = {
        "changed_frac": float(changed.mean()),
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()) if delta.size else 0.0,
        "n_unique_max": float(int(ladder.n_unique[0].max().item())),
    }
    if sample_axis:
        out = out_flat.reshape(n, s, v, t).transpose(0, 2, 1, 3)
    else:
        out = out_flat
    return out.astype(np.float32, copy=False), stats
