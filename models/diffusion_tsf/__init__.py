"""
Diffusion-based Time Series Forecasting using 2D Image Representation

This module implements a supervised TSF model that:
1. Maps 1D time-series to 2D "stripe" images (inspired by ViTime)
2. Uses a conditional 2D U-Net with DDPM/DDIM diffusion
3. Decodes the generated 2D representation back to 1D forecasts

The goal is to preserve high-frequency geometric patterns (jagged edges, W/V shapes)
that are typically lost in MSE-based regression.
"""

from .config import DiffusionTSFConfig
from .preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
from .unet import ConditionalUNet2D
from .diffusion import DiffusionScheduler
from .model import DiffusionTSF
from .metrics import compute_metrics, shape_preservation_score

__all__ = [
    "DiffusionTSFConfig",
    "Standardizer",
    "TimeSeriesTo2D", 
    "VerticalGaussianBlur",
    "ConditionalUNet2D",
    "DiffusionScheduler",
    "DiffusionTSF",
    "compute_metrics",
    "shape_preservation_score",
]

