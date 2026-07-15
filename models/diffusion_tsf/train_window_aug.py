"""Train-time window augmentations applied on z-scored values before ordinal encoding.

Augs are per-variate and per-window. ~50% of samples are left untouched; the rest
stack 1–3 randomly chosen transforms. Heavy augs (flip / time-stretch / trend)
are skipped for variates marked periodic via a precomputed ACF flag.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NON_HEAVY_AUGS = (
    "amplitude_shift",
    "window_shift",
    "reverse",
    "gaussian_noise",
    "masking",
    "sudden_shock",
    "irregularity",
)
HEAVY_AUGS = (
    "flip",
    "time_stretch",
    "linear_trend",
)
ALL_AUGS = NON_HEAVY_AUGS + HEAVY_AUGS


def estimate_variate_periodicity(
    train_tv: np.ndarray,
    *,
    min_period: int = 12,
    max_period: int = 168,
    acf_threshold: float = 0.35,
) -> np.ndarray:
    """Return bool flags (V,) — True if variate looks highly periodic.

    Uses FFT autocorrelation peak in ``[min_period, max_period]`` on the full
    training split (z-scored). Cheap one-shot precompute.
    """
    if train_tv.ndim != 2:
        raise ValueError(f"expected (T, V), got {train_tv.shape}")
    t_len, n_v = train_tv.shape
    flags = np.zeros(n_v, dtype=bool)
    hi = min(int(max_period), max(int(min_period) + 1, t_len // 3))
    lo = int(min_period)
    if hi <= lo or t_len < lo * 3:
        return flags

    for vi in range(n_v):
        x = train_tv[:, vi].astype(np.float64, copy=True)
        x -= x.mean()
        std = float(x.std())
        if std < 1e-8:
            continue
        x /= std
        n = len(x)
        fft = np.fft.rfft(x, n=2 * n)
        acf = np.fft.irfft(fft * np.conj(fft), n=2 * n)[:n].real
        if abs(acf[0]) < 1e-12:
            continue
        acf /= acf[0]
        peak = float(np.max(acf[lo:hi]))
        flags[vi] = peak >= float(acf_threshold)
    return flags


def _envelope_for_variate(
    ladder,
    vi: int,
) -> Tuple[float, float]:
    """Train ladder z-score min/max for one variate."""
    if ladder is None:
        return -8.0, 8.0
    k = int(ladder.n_unique[0, vi].item())
    uniq = ladder.values[0, vi, :k].detach().cpu().numpy().astype(np.float64)
    if k <= 0:
        return -8.0, 8.0
    if k == 1:
        v = float(uniq[0])
        return v, v
    return float(uniq[0]), float(uniq[k - 1])


def _bin_spacing(ladder, vi: int) -> float:
    lo, hi = _envelope_for_variate(ladder, vi)
    if ladder is None:
        return max(1e-3, (hi - lo) / 16.0)
    k = max(1, int(ladder.n_unique[0, vi].item()) - 1)
    return max(1e-6, (hi - lo) / float(k))


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if lo >= hi:
        return np.full_like(x, lo)
    return np.clip(x, lo, hi)


def _flatline_mask(x: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """True where timestep equals previous (consecutive flat runs)."""
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < 2:
        return mask
    same = np.abs(np.diff(x)) <= atol
    # mark both sides of equal pairs so whole plateaus are excluded
    mask[1:] |= same
    mask[:-1] |= same
    return mask


def _join_past_future(
    past: np.ndarray,
    future: np.ndarray,
    overlap: int,
) -> np.ndarray:
    """past (V,L), future (V, overlap+H) → full (V, L+H)."""
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap > 0:
        return np.concatenate([past, future[:, overlap:]], axis=-1)
    return np.concatenate([past, future], axis=-1)


def _split_full(
    full: np.ndarray,
    lookback: int,
    overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    past = full[:, :lookback]
    future = full[:, lookback - overlap :]
    return past, future


def _sync_overlap(past: np.ndarray, future: np.ndarray, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return future
    out = future.copy()
    out[:, :overlap] = past[:, -overlap:]
    return out


def aug_amplitude_shift(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    spacing = _bin_spacing(ladder, vi)
    lo, hi = _envelope_for_variate(ladder, vi)
    full = _join_past_future(past, future, overlap)
    x = full[vi]
    mean = float(x.mean())
    # stretch so typical deviation moves by ~1–3 bins
    bins = float(rng.uniform(1.2, 3.0))
    std = float(np.std(x)) + 1e-8
    factor = 1.0 + (bins * spacing) / std
    if rng.random() < 0.5:
        factor = 1.0 / max(factor, 1.05)
    y = mean + (x - mean) * factor
    full[vi] = _clamp(y, lo, hi)
    return _split_full(full, past.shape[-1], overlap)


def aug_window_shift(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    spacing = _bin_spacing(ladder, vi)
    lo, hi = _envelope_for_variate(ladder, vi)
    full = _join_past_future(past, future, overlap)
    x = full[vi]
    delta = float(rng.uniform(1.0, 3.0) * spacing) * float(rng.choice([-1.0, 1.0]))
    # shrink if needed so clamped series still moves
    room_hi = hi - float(np.max(x))
    room_lo = float(np.min(x)) - lo
    if delta > 0:
        delta = min(delta, max(0.0, room_hi))
    else:
        delta = -min(-delta, max(0.0, room_lo))
    full[vi] = _clamp(x + delta, lo, hi)
    return _split_full(full, past.shape[-1], overlap)


def aug_reverse(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    del ladder, rng
    full = _join_past_future(past, future, overlap)
    full[vi] = full[vi, ::-1].copy()
    return _split_full(full, past.shape[-1], overlap)


def aug_flip(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    del rng
    lo, hi = _envelope_for_variate(ladder, vi)
    full = _join_past_future(past, future, overlap)
    x = full[vi]
    mean = float(x.mean())
    full[vi] = _clamp(2.0 * mean - x, lo, hi)
    return _split_full(full, past.shape[-1], overlap)


def aug_time_stretch(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-linear time warp via random control-point remapping (not identity resample)."""
    lo, hi = _envelope_for_variate(ladder, vi)
    full = _join_past_future(past, future, overlap)
    x = full[vi]
    n = len(x)
    n_ctrl = int(rng.integers(4, 8))
    src = np.linspace(0.0, 1.0, n_ctrl)
    # jitter interior knots; keep endpoints fixed so length is preserved
    dst = src.copy()
    if n_ctrl > 2:
        jitter = rng.uniform(-0.18, 0.18, size=n_ctrl - 2)
        dst[1:-1] = np.clip(src[1:-1] + jitter, 1e-3, 1.0 - 1e-3)
        dst = np.sort(dst)
        dst[0], dst[-1] = 0.0, 1.0
        # enforce strict increase
        for i in range(1, n_ctrl):
            if dst[i] <= dst[i - 1]:
                dst[i] = min(1.0, dst[i - 1] + 1e-3)
        dst[-1] = 1.0
    t_out = np.linspace(0.0, 1.0, n)
    # For each output time, sample from warped source coordinate.
    t_in = np.interp(t_out, dst, src)
    y = np.interp(t_in, np.linspace(0.0, 1.0, n), x)
    full[vi] = _clamp(y, lo, hi)
    return _split_full(full, past.shape[-1], overlap)


def aug_linear_trend(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add/subtract a linear ramp across lookback+horizon, clamped to ladder envelope.

    Amplitude is a fraction of the per-variate z-score span (not raw bin spacing),
    otherwise high-cardinality ordinal ladders make the ramp invisible.
    """
    lo, hi = _envelope_for_variate(ladder, vi)
    span = max(hi - lo, 1e-6)
    full = _join_past_future(past, future, overlap)
    x = full[vi].astype(np.float64, copy=False)
    n = len(x)
    t = np.linspace(-0.5, 0.5, n)

    # Max |amp| such that x + amp*t stays inside [lo, hi] for all t.
    # amp*t <= hi-x  and amp*t >= lo-x  for every i.
    max_pos = np.inf
    max_neg = np.inf
    for ti, xi in zip(t, x):
        if abs(ti) < 1e-12:
            continue
        if ti > 0:
            max_pos = min(max_pos, (hi - xi) / ti)
            max_neg = min(max_neg, (xi - lo) / ti)
        else:
            max_pos = min(max_pos, (xi - lo) / (-ti))
            max_neg = min(max_neg, (hi - xi) / (-ti))
    max_pos = float(max(0.0, max_pos if np.isfinite(max_pos) else 0.0))
    max_neg = float(max(0.0, max_neg if np.isfinite(max_neg) else 0.0))

    sign = float(rng.choice([-1.0, 1.0]))
    room = max_pos if sign > 0 else max_neg
    # Target ~12–40% of envelope peak-to-peak; fall back to whatever fits.
    target = float(rng.uniform(0.12, 0.40) * span)
    amp = sign * min(target, room)
    if abs(amp) < 1e-8:
        # Series already hugs both bounds; nudge with a smaller centered ramp if possible.
        amp = sign * min(0.05 * span, max(max_pos, max_neg, 0.0))

    y = x + amp * t
    full[vi] = _clamp(y, lo, hi).astype(np.float32)
    return _split_full(full, past.shape[-1], overlap)


def aug_gaussian_noise(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = _envelope_for_variate(ladder, vi)
    x = past[vi].copy()
    span = max(hi - lo, 1e-6)
    sigma = float(rng.uniform(0.01, 0.10)) * span
    flat = _flatline_mask(x)
    noise = rng.normal(0.0, sigma, size=x.shape)
    noise[flat] = 0.0
    past = past.copy()
    past[vi] = _clamp(x + noise, lo, hi)
    future = _sync_overlap(past, future, overlap)
    return past, future


def aug_masking(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    del ladder
    x = past[vi].copy()
    n = len(x)
    max_len = max(4, n // 3)
    seg = int(rng.integers(4, max_len + 1))
    start = int(rng.integers(0, max(1, n - seg + 1)))
    end = start + seg
    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    if not (~mask).any():
        return past, future
    fill = float(np.mean(x[~mask]))
    x[mask] = fill
    past = past.copy()
    past[vi] = x
    future = _sync_overlap(past, future, overlap)
    return past, future


def aug_sudden_shock(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    spacing = _bin_spacing(ladder, vi)
    lo, hi = _envelope_for_variate(ladder, vi)
    x = past[vi].copy()
    n = len(x)
    frac = float(rng.uniform(0.10, 0.60))
    cut = max(1, int(round(n * frac)))
    delta = float(rng.uniform(1.0, 4.0) * spacing) * float(rng.choice([-1.0, 1.0]))
    head = x[:cut]
    if delta > 0:
        delta = min(delta, max(0.0, hi - float(np.max(head))))
    else:
        delta = -min(-delta, max(0.0, float(np.min(head)) - lo))
    x[:cut] = _clamp(head + delta, lo, hi)
    past = past.copy()
    past[vi] = x
    future = _sync_overlap(past, future, overlap)
    return past, future


def aug_irregularity(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    vi: int,
    rng: np.random.Generator,
    donor_past: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = _envelope_for_variate(ladder, vi)
    x = past[vi].copy()
    n = len(x)
    protect = min(64, n)
    writable = max(0, n - protect)
    if writable < 4 or donor_past is None:
        return past, future
    seg = int(rng.integers(4, min(128, writable) + 1))
    src = donor_past[vi]
    if len(src) < seg:
        return past, future
    src_start = int(rng.integers(0, len(src) - seg + 1))
    dst_start = int(rng.integers(0, writable - seg + 1))
    x[dst_start : dst_start + seg] = _clamp(src[src_start : src_start + seg], lo, hi)
    # last 64 stay original (already untouched if dst in writable)
    past = past.copy()
    past[vi] = x
    future = _sync_overlap(past, future, overlap)
    return past, future


_AUG_FNS = {
    "amplitude_shift": aug_amplitude_shift,
    "window_shift": aug_window_shift,
    "reverse": aug_reverse,
    "flip": aug_flip,
    "time_stretch": aug_time_stretch,
    "linear_trend": aug_linear_trend,
    "gaussian_noise": aug_gaussian_noise,
    "masking": aug_masking,
    "sudden_shock": aug_sudden_shock,
    "irregularity": aug_irregularity,
}


def apply_stacked_augs(
    past: np.ndarray,
    future: np.ndarray,
    *,
    overlap: int,
    ladder,
    periodic_flags: Sequence[bool],
    rng: np.random.Generator,
    donor_past: Optional[np.ndarray] = None,
    force_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply stacked augs independently per variate. Returns (past, future, names)."""
    past = np.asarray(past, dtype=np.float32).copy()
    future = np.asarray(future, dtype=np.float32).copy()
    n_v = past.shape[0]
    applied: List[str] = []

    for vi in range(n_v):
        periodic = bool(periodic_flags[vi]) if vi < len(periodic_flags) else False
        pool = list(NON_HEAVY_AUGS) + ([] if periodic else list(HEAVY_AUGS))
        if force_names is not None:
            names = [n for n in force_names if n in pool]
            if not names:
                continue
        else:
            k = 1 + int(rng.random() < 0.55) + int(rng.random() < 0.35)  # 1..3
            k = min(k, len(pool))
            names = list(rng.choice(pool, size=k, replace=False))

        for name in names:
            fn = _AUG_FNS[name]
            kwargs: Dict[str, Any] = {
                "overlap": overlap,
                "ladder": ladder,
                "vi": vi,
                "rng": rng,
            }
            if name == "irregularity":
                kwargs["donor_past"] = donor_past
            past, future = fn(past, future, **kwargs)
            applied.append(f"v{vi}:{name}")
    return past, future, applied


def _unwrap_timeseries(ds: Dataset):
    """Walk Subset wrappers to the underlying TimeSeriesDataset-like object."""
    cur = ds
    while hasattr(cur, "dataset"):
        cur = cur.dataset
    return cur


def set_train_window_aug_epoch(loader_or_dataset, epoch: int) -> None:
    """Best-effort: find TrainWindowAugDataset under Subset wrappers and set epoch."""
    ds = getattr(loader_or_dataset, "dataset", loader_or_dataset)
    seen = set()
    while id(ds) not in seen:
        seen.add(id(ds))
        if hasattr(ds, "set_epoch") and hasattr(ds, "periodic_flags"):
            ds.set_epoch(epoch)
            return
        if hasattr(ds, "dataset"):
            ds = ds.dataset
            continue
        break


def maybe_wrap_train_window_aug(
    train_ds: Dataset,
    *,
    enabled: bool,
    apply_prob: float = 0.5,
    seed: int = 42,
    ladder=None,
    acf_threshold: float = 0.35,
) -> Dataset:
    """Wrap train split with online augs when enabled; otherwise return as-is."""
    if not enabled:
        return train_ds
    base = _unwrap_timeseries(train_ds)
    if not hasattr(base, "data"):
        logger.warning("train_window_aug: dataset has no .data; skipping wrap")
        return train_ds

    train_np = base.data.detach().cpu().numpy() if torch.is_tensor(base.data) else np.asarray(base.data)
    flags = estimate_variate_periodicity(train_np, acf_threshold=acf_threshold)
    logger.info(
        "train_window_aug: enabled apply_prob=%.2f periodic_variates=%s/%s flags=%s",
        apply_prob,
        int(flags.sum()),
        len(flags),
        flags.tolist(),
    )
    return TrainWindowAugDataset(
        train_ds,
        apply_prob=apply_prob,
        seed=seed,
        ladder=ladder,
        periodic_flags=flags,
    )


class TrainWindowAugDataset(Dataset):
    """Online aug wrapper. Reads raw z-scored windows; re-encodes ordinal ranks."""

    def __init__(
        self,
        base: Dataset,
        *,
        apply_prob: float = 0.5,
        seed: int = 42,
        ladder=None,
        periodic_flags: Optional[np.ndarray] = None,
    ):
        self.base = base
        self.apply_prob = float(apply_prob)
        self.seed = int(seed)
        self.ladder = ladder
        self._ts = _unwrap_timeseries(base)
        self.lookback = int(self._ts.lookback)
        self.horizon = int(self._ts.horizon)
        self.overlap = int(self._ts.lookback_overlap)
        self.stride = int(self._ts.stride)
        n_v = int(self._ts.data.shape[1]) if self._ts.data.ndim == 2 else 1
        if periodic_flags is None:
            periodic_flags = np.zeros(n_v, dtype=bool)
        self.periodic_flags = np.asarray(periodic_flags, dtype=bool)
        self.yields_ordinal_ranks = ladder is not None
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Vary online augs across epochs (called from train loops)."""
        self._epoch = max(0, int(epoch))

    def __len__(self) -> int:
        return len(self.base)

    def _map_index(self, idx: int) -> int:
        """Resolve Subset indices to underlying TimeSeriesDataset index."""
        cur = self.base
        i = int(idx)
        while hasattr(cur, "indices") and hasattr(cur, "dataset"):
            i = int(cur.indices[i])
            cur = cur.dataset
        return i

    def _raw_window(self, ts_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = ts_idx * self.stride
        data = self._ts.data
        if torch.is_tensor(data):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)
        past = data_np[start : start + self.lookback].T.astype(np.float32)
        target_start = start + self.lookback - self.overlap
        target_end = start + self.lookback + self.horizon
        future = data_np[target_start:target_end].T.astype(np.float32)
        return past, future

    def _encode(self, past: np.ndarray, future: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ladder is None:
            return (
                torch.as_tensor(past, dtype=torch.float32),
                torch.as_tensor(future, dtype=torch.float32),
            )
        from models.diffusion_tsf.ordinal_window_norm import encode_with_ladder

        # past/future are (V, T); encode_with_ladder expects (B, V, T).
        past_t = torch.as_tensor(past, dtype=torch.float32).unsqueeze(0)
        fut_t = torch.as_tensor(future, dtype=torch.float32).unsqueeze(0)
        past_r = encode_with_ladder(past_t, self.ladder).squeeze(0)
        fut_r = encode_with_ladder(fut_t, self.ladder).squeeze(0)
        return past_r, fut_r

    def __getitem__(self, idx: int):
        ts_idx = self._map_index(idx)
        past, future = self._raw_window(ts_idx)
        rng = np.random.default_rng(
            self.seed + 1_000_003 * int(self._epoch) + 10007 * int(idx) + 17 * ts_idx
        )

        if rng.random() < self.apply_prob:
            donor_idx = int(rng.integers(0, len(self)))
            donor_ts = self._map_index(donor_idx)
            donor_past, _ = self._raw_window(donor_ts)
            past, future, _ = apply_stacked_augs(
                past,
                future,
                overlap=self.overlap,
                ladder=self.ladder,
                periodic_flags=self.periodic_flags,
                rng=rng,
                donor_past=donor_past,
            )

        return self._encode(past, future)
