"""Fourier low/high frequency split with flatline compression."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

CutoffSpec = Union[int, Sequence[int], torch.Tensor]


def rle_compress_1d(x: np.ndarray, atol: float) -> Tuple[np.ndarray, List[int]]:
    """Collapse contiguous near-equal runs to one timestep per run."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32), []
    vals: List[float] = []
    lens: List[int] = []
    i = 0
    n = x.size
    while i < n:
        j = i + 1
        while j < n and abs(x[j] - x[i]) <= atol:
            j += 1
        vals.append(float(x[i]))
        lens.append(j - i)
        i = j
    return np.asarray(vals, dtype=np.float32), lens


def rle_expand(compressed: np.ndarray, segment_lengths: Sequence[int]) -> np.ndarray:
    if not segment_lengths:
        return np.asarray(compressed, dtype=np.float32).reshape(-1)
    return np.repeat(np.asarray(compressed, dtype=np.float32).ravel(), np.asarray(segment_lengths, dtype=np.int64))


def fft_frequency_bins(seq_len: int) -> int:
    if seq_len < 1:
        return 0
    return seq_len // 2 + 1


def mirror_reflect_pad_1d(x: np.ndarray, pad: int) -> Tuple[np.ndarray, int]:
    """Symmetric reflection pad; returns (padded, pad_len)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2 or pad <= 0:
        return x, 0
    pad = min(pad, n - 1)
    left = x[1 : pad + 1][::-1]
    right = x[-2 : -pad - 2 : -1]
    return np.concatenate([left, x, right]), pad


def _hann_taper_weights(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    return np.hanning(n).astype(np.float64)


def fft_split_compressed(
    compressed: np.ndarray,
    cutoff_bin: int,
    *,
    edge_mode: str = "mirror_pad",
    mirror_pad_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split compressed series at rFFT bin index ``cutoff_bin`` (low=[0,k), high=[k,end)).

    ``edge_mode``:
      - ``none``: raw periodic rFFT (can ring at window ends).
      - ``mirror_pad``: reflect-pad before rFFT, crop center (reduces end mismatch).
      - ``tukey``: Hann taper before rFFT, divide out taper after irfft (exact recon on interior).
    """
    comp = np.asarray(compressed, dtype=np.float64).ravel()
    n = comp.size
    if n < 2:
        return comp.astype(np.float32), np.zeros_like(comp, dtype=np.float32)

    work = comp
    pad = 0
    taper = None
    if edge_mode == "mirror_pad":
        pad = max(4, int(round(n * mirror_pad_frac)))
        work, pad = mirror_reflect_pad_1d(comp, pad)
    elif edge_mode == "tukey":
        taper = _hann_taper_weights(n)
        work = comp * taper

    spec = np.fft.rfft(work)
    n_bins = spec.size
    k = int(max(1, min(cutoff_bin, n_bins - 1)))
    mask_low = np.zeros(n_bins, dtype=np.float64)
    mask_high = np.zeros(n_bins, dtype=np.float64)
    mask_low[:k] = 1.0
    mask_high[k:] = 1.0
    low = np.fft.irfft(spec * mask_low, n=len(work))
    high = np.fft.irfft(spec * mask_high, n=len(work))

    if edge_mode == "mirror_pad" and pad > 0:
        low = low[pad : pad + n]
        high = high[pad : pad + n]
    elif edge_mode == "tukey" and taper is not None:
        safe = np.maximum(taper, 1e-8)
        low = low / safe
        high = high / safe

    return low.astype(np.float32), high.astype(np.float32)


def cutoff_bin_for_high_percent(n_bins: int, high_percent: float) -> int:
    """Map desired % of rFFT bins in high band to cutoff index k."""
    if n_bins <= 1:
        return 1
    pct = float(high_percent) / 100.0
    high_bins = max(1, int(round(n_bins * pct)))
    return max(1, min(n_bins - 1, n_bins - high_bins))


def fourier_frequency_split_np(
    x: np.ndarray,
    *,
    cutoff_bin: int,
    flatline_atol: float,
    edge_mode: str = "mirror_pad",
    mirror_pad_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split 1D series into low/high Fourier components with flatline preservation."""
    comp, lens = rle_compress_1d(x, flatline_atol)
    low_c, high_c = fft_split_compressed(
        comp, cutoff_bin, edge_mode=edge_mode, mirror_pad_frac=mirror_pad_frac,
    )
    low = rle_expand(low_c, lens)
    high = rle_expand(high_c, lens)
    if low.shape[0] != x.shape[0]:
        raise ValueError(f"expand length mismatch: {low.shape[0]} vs {x.shape[0]}")
    return low, high


def _torch_rle_compress_1d(x: torch.Tensor, atol: float) -> Tuple[torch.Tensor, List[int]]:
    arr = x.detach().cpu().numpy()
    comp, lens = rle_compress_1d(arr, atol)
    return torch.from_numpy(comp).to(device=x.device, dtype=x.dtype), lens


def _torch_rle_expand(compressed: torch.Tensor, segment_lengths: Sequence[int], out_len: int) -> torch.Tensor:
    expanded = rle_expand(compressed.detach().cpu().numpy(), segment_lengths)
    if expanded.size != out_len:
        raise ValueError(f"expand length mismatch: {expanded.size} vs {out_len}")
    return torch.from_numpy(expanded).to(device=compressed.device, dtype=compressed.dtype)


def _torch_fft_split_compressed(
    compressed: torch.Tensor,
    cutoff_bin: int,
    *,
    edge_mode: str = "mirror_pad",
    mirror_pad_frac: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    low, high = fft_split_compressed(
        compressed.detach().cpu().numpy(),
        cutoff_bin,
        edge_mode=edge_mode,
        mirror_pad_frac=mirror_pad_frac,
    )
    return (
        torch.from_numpy(low).to(device=compressed.device, dtype=compressed.dtype),
        torch.from_numpy(high).to(device=compressed.device, dtype=compressed.dtype),
    )


def _cutoff_per_variate(cutoff_bin: CutoffSpec, n_vars: int) -> List[int]:
    if isinstance(cutoff_bin, int):
        return [int(cutoff_bin)] * n_vars
    if isinstance(cutoff_bin, torch.Tensor):
        vals = [int(v) for v in cutoff_bin.detach().cpu().tolist()]
    else:
        vals = [int(v) for v in cutoff_bin]
    if len(vals) != n_vars:
        raise ValueError(f"expected {n_vars} cutoffs, got {len(vals)}")
    return vals


def fourier_frequency_split_torch(
    x: torch.Tensor,
    *,
    cutoff_bin: CutoffSpec,
    flatline_atol: float,
    edge_mode: str = "mirror_pad",
    mirror_pad_frac: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split (B, V, T) along time; each variate may use its own cutoff bin."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"expected (B,V,T) or (B,T), got {tuple(x.shape)}")
    b, v, t = x.shape
    cutoffs = _cutoff_per_variate(cutoff_bin, v)
    lows = []
    highs = []
    for bi in range(b):
        row_low = []
        row_high = []
        for vi in range(v):
            series = x[bi, vi]
            comp, lens = _torch_rle_compress_1d(series, flatline_atol)
            low_c, high_c = _torch_fft_split_compressed(
                comp, cutoffs[vi], edge_mode=edge_mode, mirror_pad_frac=mirror_pad_frac,
            )
            row_low.append(_torch_rle_expand(low_c, lens, t))
            row_high.append(_torch_rle_expand(high_c, lens, t))
        lows.append(torch.stack(row_low, dim=0))
        highs.append(torch.stack(row_high, dim=0))
    low = torch.stack(lows, dim=0)
    high = torch.stack(highs, dim=0)
    return low, high


def high_band_std_by_cutoff(
    series_list: Sequence[np.ndarray],
    *,
    n_bins: int,
    flatline_atol: float,
    edge_mode: str = "mirror_pad",
    mirror_pad_frac: float = 0.25,
) -> Dict[int, float]:
    """Pool high-band timesteps per cutoff across windows for one variate."""
    pooled: Dict[int, List[np.ndarray]] = {k: [] for k in range(1, n_bins)}
    for series in series_list:
        for k in pooled:
            _low, high = fourier_frequency_split_np(
                series,
                cutoff_bin=k,
                flatline_atol=flatline_atol,
                edge_mode=edge_mode,
                mirror_pad_frac=mirror_pad_frac,
            )
            pooled[k].append(high)
    return {k: float(np.std(np.concatenate(vals))) if vals else 0.0 for k, vals in pooled.items()}


def select_cutoff_min_low_band(
    std_by_k: Dict[int, float],
    *,
    target_std: float = 1.0,
) -> Tuple[int, str]:
    """Smallest cutoff k (fewest low bins) with pooled high-band std <= target."""
    if not std_by_k:
        return 1, "empty"
    for k in sorted(std_by_k):
        if std_by_k[k] <= target_std:
            return k, "min_low_band_std"
    return max(std_by_k), "max_cutoff_fallback"


def prior_cutoff_bin(n_bins: int, high_freq_percent: float) -> int:
    if n_bins <= 1:
        return 1
    high_bins = max(1, int(np.ceil(n_bins * float(high_freq_percent))))
    return max(1, n_bins - high_bins)
