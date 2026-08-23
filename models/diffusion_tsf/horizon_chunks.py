"""Fixed-canvas overlap-average horizon chunks.

Canvas is always lookback_overlap + inner (104 = 8 + 96 for the live
canvas128 geometry). Global position is a chunk start t0 in future
coords (0 for the first inner-steps), not grow-the-past AR.
"""

from __future__ import annotations

from typing import Sequence

import torch


def chunk_starts(horizon: int, inner: int = 96, overlap: int = 8) -> list[int]:
    """t0 values in future coords so each slice is exactly overlap+inner wide.

    Last chunk is end-aligned (t0 = H - inner). Neighbor overlaps are
    ``overlap`` except the last pair, which may overlap more.
    """
    h = int(horizon)
    inner_w = int(inner)
    k = int(overlap)
    if h < inner_w:
        raise ValueError(f"horizon {h} is shorter than inner chunk {inner_w}")
    if inner_w <= 0:
        raise ValueError(f"inner must be positive, got {inner_w}")
    if k < 0:
        raise ValueError(f"overlap must be >= 0, got {k}")
    if k >= inner_w:
        raise ValueError(f"overlap {k} must be < inner {inner_w}")
    if h == inner_w:
        return [0]
    last = h - inner_w
    starts = [0]
    t = inner_w
    while t < last:
        starts.append(t)
        t += inner_w
    if starts[-1] != last:
        starts.append(last)
    return starts


def slice_future_canvas(
    future: torch.Tensor,
    t0: torch.Tensor,
    *,
    inner: int,
    overlap: int,
) -> torch.Tensor:
    """Slice a (B, V, overlap+inner) canvas from a long future (B, V, overlap+H)."""
    if future.ndim != 3:
        raise ValueError(f"future must be (B, V, T), got {tuple(future.shape)}")
    b, v, _ = future.shape
    if t0.shape != (b,):
        raise ValueError(f"t0 must be ({b},), got {tuple(t0.shape)}")
    canvas_w = int(overlap) + int(inner)
    idx = t0.to(device=future.device, dtype=torch.long).view(b, 1, 1) + torch.arange(
        canvas_w, device=future.device, dtype=torch.long
    ).view(1, 1, canvas_w)
    return future.gather(-1, idx.expand(b, v, canvas_w))


def overlap_average_stitch(
    chunk_preds: torch.Tensor,
    starts: Sequence[int],
    *,
    horizon: int,
    inner: int,
    overlap: int,
) -> torch.Tensor:
    """Average decoded 1D canvases onto length ``horizon``.

    ``chunk_preds`` is (B, n_chunks, V, overlap+inner) in the same 1D space as
    metrics (window-denormalized). Bins that no chunk covers are illegal.
    """
    if chunk_preds.ndim != 4:
        raise ValueError(
            f"chunk_preds must be (B, n_chunks, V, canvas), got {tuple(chunk_preds.shape)}"
        )
    b, n_chunks, v, canvas_w = chunk_preds.shape
    starts_l = [int(s) for s in starts]
    if n_chunks != len(starts_l):
        raise ValueError(f"n_chunks {n_chunks} != len(starts) {len(starts_l)}")
    expected_w = int(overlap) + int(inner)
    if canvas_w != expected_w:
        raise ValueError(f"canvas width {canvas_w} != overlap+inner {expected_w}")
    h = int(horizon)
    k = int(overlap)
    inner_w = int(inner)
    accum = chunk_preds.new_zeros(b, v, h)
    counts = chunk_preds.new_zeros(b, v, h)
    for i, t0 in enumerate(starts_l):
        start = t0 - k
        end = t0 + inner_w
        src_lo = max(0, -start)
        dst_lo = max(0, start)
        dst_hi = min(h, end)
        width = dst_hi - dst_lo
        if width <= 0:
            raise ValueError(f"chunk t0={t0} does not overlap horizon [0, {h})")
        src_hi = src_lo + width
        accum[:, :, dst_lo:dst_hi] = accum[:, :, dst_lo:dst_hi] + chunk_preds[:, i, :, src_lo:src_hi]
        counts[:, :, dst_lo:dst_hi] = counts[:, :, dst_lo:dst_hi] + 1
    if bool((counts <= 0).any()):
        missing = (counts[0, 0] <= 0).nonzero(as_tuple=False).view(-1).tolist()
        raise ValueError(f"stitch left uncovered forecast steps: {missing[:12]}")
    return accum / counts
