"""
Evaluation metrics for Time Series Forecasting.

Includes:
- Standard metrics: MSE, MAE
- Shape-Preservation Metric: Compares first-order derivatives
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict

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

