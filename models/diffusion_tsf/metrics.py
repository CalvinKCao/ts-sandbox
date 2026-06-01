"""
Evaluation metrics for Time Series Forecasting.

Includes:
- Standard metrics: MSE, MAE
- Shape-Preservation Metric: Compares first-order derivatives
"""

import math
import torch
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error.
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        
    Returns:
        Scalar MSE value
    """
    return F.mse_loss(pred, target)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error.
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        
    Returns:
        Scalar MAE value
    """
    return F.l1_loss(pred, target)


def monotonicity_loss(pred_x0: torch.Tensor) -> torch.Tensor:
    """Penalize monotonicity violations along the height axis of a CDF map.
    
    Args:
        pred_x0: Tensor of shape (batch, channels, height, width) representing
                 a denoised occupancy/CDF map.
                 
    Returns:
        Scalar loss = mean positive increase between consecutive height rows.
    """
    # diff[y] = value[y+1] - value[y]; positive values are violations
    diff = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]
    violations = F.relu(diff)
    return violations.mean()


def first_order_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute first-order differences (discrete derivative).
    
    Args:
        x: Time series of shape (batch, seq_len)
        
    Returns:
        Gradients of shape (batch, seq_len - 1)
    """
    return x[:, 1:] - x[:, :-1]


def shape_preservation_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 50
) -> Dict[str, torch.Tensor]:
    """Shape-Preservation Metric.
    
    Compares the distribution of first-order derivatives (gradients) between
    predictions and ground truth. This captures whether high-frequency textures
    (jagged edges, W/V shapes) are preserved.
    
    The metric computes:
    1. Gradient MAE: Direct comparison of derivatives
    2. Gradient Distribution Divergence: KL divergence between histogram distributions
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        num_bins: Number of bins for histogram
        
    Returns:
        Dictionary with:
        - 'gradient_mae': MAE of first-order derivatives
        - 'gradient_correlation': Correlation between gradients
        - 'shape_score': Combined shape preservation score (lower is better)
    """
    # Compute first-order gradients
    pred_grad = first_order_gradient(pred)
    target_grad = first_order_gradient(target)
    
    # 1. Direct gradient comparison (MAE)
    gradient_mae = F.l1_loss(pred_grad, target_grad)
    
    # 2. Gradient correlation (Pearson correlation)
    pred_grad_flat = pred_grad.flatten()
    target_grad_flat = target_grad.flatten()
    
    # Center the data
    pred_centered = pred_grad_flat - pred_grad_flat.mean()
    target_centered = target_grad_flat - target_grad_flat.mean()
    
    # Compute correlation
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
    gradient_corr = numerator / denominator
    
    # 3. Sign agreement (captures direction of changes)
    pred_sign = torch.sign(pred_grad)
    target_sign = torch.sign(target_grad)
    sign_agreement = (pred_sign == target_sign).float().mean()
    
    # Combined shape score (lower is better)
    # Weight: MAE is penalized, correlation and sign agreement are rewarded
    shape_score = gradient_mae - 0.1 * gradient_corr - 0.1 * sign_agreement + 0.2
    
    logger.debug(f"Shape metrics: grad_mae={gradient_mae:.4f}, grad_corr={gradient_corr:.4f}, "
                 f"sign_agree={sign_agreement:.4f}")
    
    return {
        'gradient_mae': gradient_mae,
        'gradient_correlation': gradient_corr,
        'sign_agreement': sign_agreement,
        'shape_score': shape_score
    }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute all evaluation metrics.
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mse': mse(pred, target),
        'mae': mae(pred, target),
    }
    
    # Add shape preservation metrics
    shape_metrics = shape_preservation_score(pred, target)
    metrics.update(shape_metrics)
    
    return metrics


def log_metrics(metrics: Dict[str, torch.Tensor], prefix: str = "") -> str:
    """Format metrics for logging.
    
    Args:
        metrics: Dictionary of metric name -> value
        prefix: Optional prefix for metric names
        
    Returns:
        Formatted string
    """
    parts = []
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        full_name = f"{prefix}{name}" if prefix else name
        parts.append(f"{full_name}={value:.4f}")
    
    return " | ".join(parts)


# ============================================================================
# Texture Metrics
# ============================================================================

def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def ordinal_jsd(a: np.ndarray, b: np.ndarray, order: int = 4) -> float:
    from itertools import permutations

    perms = list(permutations(range(order)))
    lookup = {p: i for i, p in enumerate(perms)}

    def dist(x: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(perms), dtype=np.float64)
        if len(x) < order:
            counts += 1.0
        else:
            for i in range(len(x) - order + 1):
                ranks = tuple(np.argsort(np.argsort(x[i : i + order], kind="mergesort"), kind="mergesort"))
                counts[lookup[ranks]] += 1.0
        counts += 1e-12
        return counts / counts.sum()

    p = dist(a)
    q = dist(b)
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(jsd)


def _line_lengths_bool(arr: np.ndarray) -> List[int]:
    lengths: List[int] = []
    run = 0
    for value in arr:
        if value:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def rqa_features(x: np.ndarray, eps: float = 0.2, min_len: int = 2) -> np.ndarray:
    x = zscore_1d(x)
    R = np.abs(x[:, None] - x[None, :]) < eps
    np.fill_diagonal(R, False)
    recurrence = R.sum() + 1e-8

    diag_points = 0
    for offset in range(-len(x) + 1, len(x)):
        if offset == 0:
            continue
        for length in _line_lengths_bool(np.diagonal(R, offset=offset)):
            if length >= min_len:
                diag_points += length

    vert_points = 0
    for col in range(R.shape[1]):
        for length in _line_lengths_bool(R[:, col]):
            if length >= min_len:
                vert_points += length

    det = diag_points / recurrence
    lam = vert_points / recurrence
    return np.array([lam, det], dtype=np.float64)


def variogram(x: np.ndarray, max_lag: int = 24) -> np.ndarray:
    x = zscore_1d(x)
    lags = []
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        diff = x[lag:] - x[:-lag]
        lags.append(0.5 * np.mean(diff * diff))
    return np.asarray(lags, dtype=np.float64)


def _fallback_signature_features(path: np.ndarray) -> np.ndarray:
    t = path[:, 0]
    x = path[:, 1]
    dx = np.diff(x)
    dt = np.diff(t)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(x, t)
    else:
        area = np.sum((x[1:] + x[:-1]) * 0.5 * dt) if len(dt) else 0.0
    return np.array(
        [
            x[-1] - x[0],
            np.sum(np.abs(dx)),
            np.mean(dx) if len(dx) else 0.0,
            np.std(dx) if len(dx) else 0.0,
            area,
            np.sum(dt * dx) if len(dx) else 0.0,
        ],
        dtype=np.float64,
    )


def path_signature_distance(a: np.ndarray, b: np.ndarray, window: int = 12, depth: int = 3) -> float:
    try:
        import iisignature  # type: ignore
    except Exception:
        iisignature = None

    a = zscore_1d(a)
    b = zscore_1d(b)
    distances = []
    for start in range(0, len(a) - window + 1, window):
        aa = a[start : start + window]
        bb = b[start : start + window]
        t = np.linspace(0.0, 1.0, window)
        pa = np.column_stack([t, aa])
        pb = np.column_stack([t, bb])
        if iisignature is not None:
            fa = np.asarray(iisignature.sig(pa, depth), dtype=np.float64)
            fb = np.asarray(iisignature.sig(pb, depth), dtype=np.float64)
        else:
            fa = _fallback_signature_features(pa)
            fb = _fallback_signature_features(pb)
        distances.append(np.linalg.norm(fa - fb) / math.sqrt(max(1, fa.size)))
    if not distances:
        return 0.0
    return float(np.mean(distances))


def aggregate_texture_per_sample(
    y_true: np.ndarray,
    samples: np.ndarray,
    max_draws: Optional[int] = None,
) -> Dict[str, float]:
    """Mean texture over stochastic draws (not texture of the sample mean).

    Args:
        y_true: ``[batch, variates, length]`` (or compatible).
        samples: ``[n_draws, batch, variates, length]``.
        max_draws: If set, use only the first ``max_draws`` draws.
    """
    per_draw: Dict[str, List[float]] = {}
    n_draws = samples.shape[0] if max_draws is None else min(samples.shape[0], max_draws)
    for i in range(n_draws):
        m = texture_metrics(y_true, samples[i])
        for k, v in m.items():
            per_draw.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in per_draw.items()}


def _as_float(x: np.ndarray) -> float:
    return float(np.asarray(x, dtype=np.float64).mean())


def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> float:
    """CRPS used by the MMPD eval harness.

    Args:
        y_true: ``[batch, variates, length]``.
        samples: ``[batch, variates, n_samples, length]``.
    """
    expected_abs = np.abs(samples - y_true[:, :, None, :]).mean(axis=2)
    sample_count = samples.shape[2]
    total = np.zeros_like(y_true, dtype=np.float64)
    chunk = max(1, 256 // max(1, sample_count))
    for start in range(0, samples.shape[0], chunk):
        end = min(samples.shape[0], start + chunk)
        s = samples[start:end].astype(np.float64)
        total[start:end] = np.abs(
            s[:, :, :, None, :] - s[:, :, None, :, :]
        ).mean(axis=(2, 3))
    return _as_float(expected_abs - 0.5 * total)


def topk_from_modes(
    y_true: np.ndarray,
    mode_center: np.ndarray,
    mode_prob: np.ndarray,
    max_k: int = 3,
) -> Dict[str, float]:
    """Top-k MSE/MAE from mode centers sorted by descending probability.

    Report only top-1 and top-3 for parity with the MMPD matrix.
    """
    order = np.argsort(-mode_prob, axis=2)
    out: Dict[str, float] = {}
    max_k = min(max_k, mode_center.shape[2])
    for k in sorted({1, max_k}):
        if k < 1 or k > max_k:
            continue
        gathered = np.take_along_axis(mode_center, order[:, :, :k, None], axis=2)
        mse_vals = ((gathered - y_true[:, :, None, :]) ** 2).mean(axis=-1).min(axis=2)
        mae_vals = np.abs(gathered - y_true[:, :, None, :]).mean(axis=-1).min(axis=2)
        out[f"top{k}_mse"] = _as_float(mse_vals)
        out[f"top{k}_mae"] = _as_float(mae_vals)
    return out


def empirical_modes_from_samples(
    samples: np.ndarray,
    max_components: int = 9,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster probabilistic trajectories into modes for MMPD-style top-k."""
    from sklearn.mixture import GaussianMixture

    batch_size, n_variates, sample_count, horizon = samples.shape
    mode_count = min(max_components, sample_count)
    centers = np.zeros((batch_size, n_variates, mode_count, horizon), dtype=np.float64)
    probs = np.zeros((batch_size, n_variates, mode_count), dtype=np.float64)

    for b in range(batch_size):
        for v in range(n_variates):
            trajectories = samples[b, v]
            if sample_count == 1:
                centers[b, v, 0] = trajectories[0]
                probs[b, v, 0] = 1.0
                continue
            n_comp = min(mode_count, sample_count)
            gmm = GaussianMixture(
                n_components=n_comp,
                random_state=seed + b * 131 + v,
                covariance_type="diag",
                reg_covar=1e-4,
                max_iter=50,
            )
            try:
                gmm.fit(trajectories)
                centers[b, v, :n_comp] = gmm.means_
                weights = gmm.weights_
                probs[b, v, :n_comp] = weights / weights.sum()
            except ValueError:
                centers[b, v, :n_comp] = trajectories[:n_comp]
                probs[b, v, :n_comp] = 1.0 / n_comp
    return centers, probs


def probabilistic_forecast_metrics(
    y_true: np.ndarray,
    samples: np.ndarray,
    *,
    gmm_components: int = 10,
    topk_max: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """MMPD-compatible probabilistic metrics for ensemble forecasts.

    ``samples`` must be ``[batch, variates, n_samples, length]``.
    """
    mode_center, mode_prob = empirical_modes_from_samples(
        samples,
        max_components=gmm_components,
        seed=seed,
    )
    out: Dict[str, float] = {
        "crps": crps_ensemble(y_true, samples),
        "n_samples": float(samples.shape[2]),
    }
    out.update(topk_from_modes(y_true, mode_center, mode_prob, max_k=topk_max))
    return out


def texture_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    vals = {
        "texture_ordinal_jsd": [],
        "texture_rqa_distance": [],
        "texture_variogram_distance": [],
        "texture_pathsig_distance": [],
    }
    flat_true = y_true.reshape(-1, y_true.shape[-1])
    flat_pred = y_pred.reshape(-1, y_pred.shape[-1])
    for gt, pred in zip(flat_true, flat_pred):
        gt_z = zscore_1d(gt)
        pred_z = zscore_1d(pred)
        vals["texture_ordinal_jsd"].append(ordinal_jsd(gt_z, pred_z))
        vals["texture_rqa_distance"].append(float(np.linalg.norm(rqa_features(gt_z) - rqa_features(pred_z))))
        va = variogram(gt_z)
        vb = variogram(pred_z)
        vals["texture_variogram_distance"].append(float(np.linalg.norm(va - vb) / math.sqrt(max(1, va.size))))
        vals["texture_pathsig_distance"].append(path_signature_distance(gt_z, pred_z))
    return {key: float(np.mean(value)) for key, value in vals.items()}

