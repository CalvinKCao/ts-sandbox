"""
Guidance models for Hybrid "Visual Guide" Forecasting.

This module provides Stage 1 predictors that generate coarse forecasts.
The forecasts are converted to 2D "ghost images" and fed to the diffusion
model as additional conditioning, allowing it to focus on refining
texture/residuals rather than predicting the global trend from scratch.

Supported guidance models:
- LastValueGuidance: Repeats the last observed value (naive baseline)
- LinearRegressionGuidance: Fits a linear trend on the lookback window
- iTransformerGuidance: Pre-trained iTransformer model (plug-in interface)
"""

import torch
import torch.nn as nn
from typing import Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod


@runtime_checkable
class GuidanceModel(Protocol):
    """Protocol for Stage 1 guidance models.
    
    Any guidance model must implement get_forecast() which takes
    a past window and returns a coarse forecast for the future.
    """
    
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Generate coarse forecast from past context.
        
        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            forecast_length: Number of future steps to predict
            
        Returns:
            Predicted future of shape (batch, [num_vars,] forecast_length)
        """
        ...


class BaseGuidance(nn.Module, ABC):
    """Base class for guidance models.
    
    Provides common interface and utilities for guidance implementations.
    """
    
    @abstractmethod
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Generate coarse forecast from past context."""
        pass
    
    def forward(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Alias for get_forecast for nn.Module compatibility."""
        return self.get_forecast(past, forecast_length)


class LastValueGuidance(BaseGuidance):
    """Naive baseline: repeats the last observed value.
    
    This is the simplest possible guidance - just assume the future
    is a flat line at the last observed value. Useful for testing
    the guidance pipeline.
    """
    
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Repeat last value for forecast_length steps.
        
        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            forecast_length: Number of future steps to predict
            
        Returns:
            Predicted future of shape (batch, [num_vars,] forecast_length)
        """
        # Handle both univariate (batch, seq) and multivariate (batch, vars, seq)
        if past.dim() == 2:
            # Univariate: (batch, past_len)
            last_val = past[:, -1:]  # (batch, 1)
            return last_val.expand(-1, forecast_length)
        else:
            # Multivariate: (batch, num_vars, past_len)
            last_val = past[:, :, -1:]  # (batch, num_vars, 1)
            return last_val.expand(-1, -1, forecast_length)


class LinearRegressionGuidance(BaseGuidance):
    """Fits a linear trend on the lookback window and extrapolates.
    
    Uses least-squares to fit y = a*t + b on the past window,
    then extrapolates to get the future forecast. This captures
    simple trends but not seasonality or complex patterns.
    """
    
    def __init__(self, use_last_n: Optional[int] = None):
        """
        Args:
            use_last_n: If specified, only use last N points for fitting.
                       If None, uses entire past window.
        """
        super().__init__()
        self.use_last_n = use_last_n
    
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Fit linear regression and extrapolate.
        
        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            forecast_length: Number of future steps to predict
            
        Returns:
            Predicted future of shape (batch, [num_vars,] forecast_length)
        """
        device = past.device
        dtype = past.dtype
        
        # Handle univariate case
        is_univariate = (past.dim() == 2)
        if is_univariate:
            past = past.unsqueeze(1)  # (batch, 1, past_len)
        
        batch_size, num_vars, past_len = past.shape
        
        # Optionally use only last N points
        if self.use_last_n is not None and self.use_last_n < past_len:
            past = past[:, :, -self.use_last_n:]
            past_len = self.use_last_n
        
        # Create time indices for past: [0, 1, ..., past_len-1]
        t_past = torch.arange(past_len, device=device, dtype=dtype)
        
        # Least squares: fit y = a*t + b
        # Using normal equations: [sum(t^2), sum(t)] [a]   [sum(t*y)]
        #                        [sum(t),   n     ] [b] = [sum(y)  ]
        
        t_sum = t_past.sum()
        t2_sum = (t_past ** 2).sum()
        n = float(past_len)
        
        # For each batch and variable
        # Reshape past for vectorized computation: (batch * num_vars, past_len)
        y = past.reshape(-1, past_len)
        
        ty_sum = (t_past.unsqueeze(0) * y).sum(dim=-1)  # (batch * num_vars,)
        y_sum = y.sum(dim=-1)  # (batch * num_vars,)
        
        # Solve 2x2 system
        det = n * t2_sum - t_sum * t_sum
        det = det.clamp(min=1e-8)  # Avoid division by zero
        
        a = (n * ty_sum - t_sum * y_sum) / det
        b = (t2_sum * y_sum - t_sum * ty_sum) / det
        
        # Extrapolate: future time indices are [past_len, past_len+1, ..., past_len+forecast_length-1]
        t_future = torch.arange(
            past_len, past_len + forecast_length,
            device=device, dtype=dtype
        )  # (forecast_length,)
        
        # Compute forecasts: y = a*t + b
        # a, b: (batch * num_vars,) -> (batch * num_vars, 1)
        # t_future: (forecast_length,) -> (1, forecast_length)
        forecast = a.unsqueeze(-1) * t_future.unsqueeze(0) + b.unsqueeze(-1)
        
        # Reshape back to (batch, num_vars, forecast_length)
        forecast = forecast.reshape(batch_size, num_vars, forecast_length)
        
        # Return univariate shape if input was univariate
        if is_univariate:
            forecast = forecast.squeeze(1)
        
        return forecast


class iTransformerGuidance(BaseGuidance):
    """Wrapper for pre-trained iTransformer model.
    
    This class wraps an iTransformer checkpoint to be used as
    a Stage 1 guidance model. The iTransformer provides accurate
    point forecasts that the diffusion model can refine.
    
    IMPORTANT: The iTransformer is kept frozen in eval mode at all times.
    When the parent DiffusionTSF model's train() method is called, this
    module stays in eval mode to ensure consistent guidance predictions.
    
    Usage:
        # Load pre-trained iTransformer
        itrans_model = load_itransformer_checkpoint(...)
        guidance = iTransformerGuidance(itrans_model)
        
        # Use in diffusion pipeline
        forecast = guidance.get_forecast(past, forecast_length=96)
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_norm: bool = True,
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        freeze: bool = True,
    ):
        """
        Args:
            model: Pre-trained iTransformer model instance.
            use_norm: Whether the model uses internal normalization.
            seq_len: If set, validates/slices past to this length; otherwise uses ``model.seq_len``.
            pred_len: If set, validates forecast_length against this; otherwise uses ``model.pred_len``.
            freeze: If True (default), keep iTransformer frozen and in eval mode at all
                times (legacy "Stage 1 frozen guidance" behavior). If False, the
                iTransformer is jointly trained with the diffusion backbone — its
                params remain trainable and it follows the parent module's train/eval
                state. Used by the end-to-end joint training path.
        """
        super().__init__()
        self.model = model
        self.use_norm = use_norm
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.freeze = freeze

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.training = False
            self.model.eval()

    def train(self, mode: bool = True):
        """Override train() so the frozen variant stays in eval mode.

        When ``freeze=True`` (default) the guidance model is pre-trained Stage-1
        and must not be affected by the parent DiffusionTSF's train/eval flips
        (prevents dropout / batch norm from switching modes).

        When ``freeze=False`` (joint training) we delegate to the normal
        ``nn.Module.train`` so the iTransformer participates in training.
        """
        if self.freeze:
            self.training = False
            self.model.eval()
            return self
        return super().train(mode)

    def eval(self):
        """Set the module to evaluation mode."""
        if self.freeze:
            self.training = False
            self.model.eval()
            return self
        return super().eval()

    def _past_seq_len(self) -> Optional[int]:
        if self.seq_len is not None:
            return self.seq_len
        return getattr(self.model, 'seq_len', None)

    def _past_pred_len(self) -> Optional[int]:
        if self.pred_len is not None:
            return self.pred_len
        return getattr(self.model, 'pred_len', None)

    def _slice_past_to_model_len(self, past: torch.Tensor) -> torch.Tensor:
        """Take the last L steps so iTransformer matches its inverted embedding width."""
        Lwant = self._past_seq_len()
        if Lwant is None:
            return past
        if past.dim() == 2:
            if past.shape[-1] < Lwant:
                raise ValueError(
                    f"past length {past.shape[-1]} < iTransformer seq_len {Lwant}"
                )
            if past.shape[-1] > Lwant:
                return past[:, -Lwant:]
            return past
        if past.dim() != 3:
            raise ValueError(f"past must be (B,L) or (B,V,L), got shape {tuple(past.shape)}")
        if past.shape[-1] < Lwant:
            raise ValueError(
                f"past length {past.shape[-1]} < iTransformer seq_len {Lwant}"
            )
        if past.shape[-1] > Lwant:
            return past[..., -Lwant:]
        return past
    
    def get_encoder_tokens(self, past: torch.Tensor) -> torch.Tensor:
        """Return iTransformer encoder output before the linear projector.

        Mirrors the normalization inside iTransformer.forecast() so the
        tokens are consistent with what the model uses internally.

        When ``freeze=True`` (legacy), the call runs under ``torch.no_grad()``.
        When ``freeze=False`` (joint training), gradients flow into the
        iTransformer encoder.

        Args:
            past: (B, V, L) raw (un-normalized) past sequence

        Returns:
            (B, V, d_model) — cross-variate-aware variate tokens
        """
        if self.freeze:
            with torch.no_grad():
                return self._encoder_tokens_impl(past)
        return self._encoder_tokens_impl(past)

    def _encoder_tokens_impl(self, past: torch.Tensor) -> torch.Tensor:
        past = self._slice_past_to_model_len(past)
        is_univariate = past.dim() == 2
        if is_univariate:
            x_enc = past.unsqueeze(-1)      # (B, L, 1)
        else:
            x_enc = past.permute(0, 2, 1)   # (B, V, L) -> (B, L, V)

        if self.model.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        enc_out = self.model.enc_embedding(x_enc, None)       # (B, V, d_model)
        enc_out, _ = self.model.encoder(enc_out, attn_mask=None)  # (B, V, d_model)
        return enc_out

    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Generate forecast using iTransformer.

        When ``freeze=True`` (legacy), the call runs under ``torch.no_grad()``.
        When ``freeze=False`` (joint training), gradients flow back into the
        iTransformer parameters.

        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            forecast_length: Number of future steps to predict (must match model's pred_len)

        Returns:
            Predicted future of shape (batch, [num_vars,] forecast_length)
        """
        if self.freeze:
            with torch.no_grad():
                return self._forecast_impl(past, forecast_length)
        return self._forecast_impl(past, forecast_length)

    def _forecast_impl(self, past: torch.Tensor, forecast_length: int) -> torch.Tensor:
        pred_expect = self._past_pred_len()
        if pred_expect is not None and forecast_length != pred_expect:
            raise ValueError(
                f"iTransformer was trained for pred_len={pred_expect}, "
                f"but got forecast_length={forecast_length}"
            )

        past = self._slice_past_to_model_len(past)

        # Handle univariate case - iTransformer expects (batch, seq_len, num_vars)
        is_univariate = (past.dim() == 2)
        if is_univariate:
            past = past.unsqueeze(-1)  # (batch, seq_len) -> (batch, seq_len, 1)
        else:
            # Convert from (batch, num_vars, seq_len) to (batch, seq_len, num_vars)
            past = past.permute(0, 2, 1)
        
        batch_size, seq_len, num_vars = past.shape

        # iTransformer forward expects: x_enc, x_mark_enc, x_dec, x_mark_dec
        # For our purposes, we don't use time marks (pass None/zeros)
        x_enc = past
        x_mark_enc = None  # Time marks not used
        
        # Decoder input is typically zeros or last values
        x_dec = torch.zeros(
            batch_size, forecast_length, num_vars,
            device=past.device, dtype=past.dtype
        )
        x_mark_dec = None
        
        # Forward pass
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Handle tuple output (with attention) vs tensor output
        if isinstance(output, tuple):
            output = output[0]
        
        # Output shape: (batch, forecast_length, num_vars)
        # Convert back to our format
        if is_univariate:
            # (batch, forecast_length, 1) -> (batch, forecast_length)
            output = output.squeeze(-1)
        else:
            # (batch, forecast_length, num_vars) -> (batch, num_vars, forecast_length)
            output = output.permute(0, 2, 1)
        
        return output


def create_guidance_model(
    guidance_type: str = "linear",
    **kwargs
) -> BaseGuidance:
    """Factory function to create guidance models.
    
    Args:
        guidance_type: Type of guidance model:
            - "last_value": LastValueGuidance (naive baseline)
            - "linear": LinearRegressionGuidance
            - "itransformer": Requires 'model' kwarg with pre-trained iTransformer
        **kwargs: Additional arguments passed to the guidance constructor
        
    Returns:
        Initialized guidance model
    """
    if guidance_type == "last_value":
        return LastValueGuidance()
    elif guidance_type == "linear":
        return LinearRegressionGuidance(**kwargs)
    elif guidance_type == "itransformer":
        if "model" not in kwargs:
            raise ValueError("iTransformerGuidance requires 'model' kwarg")
        return iTransformerGuidance(**kwargs)
    else:
        raise ValueError(f"Unknown guidance_type: {guidance_type}")

