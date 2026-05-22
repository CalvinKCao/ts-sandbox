import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as signal
from typing import Optional, Tuple, Dict, Union
import numpy as np
import logging
from dataclasses import dataclass

from .config import DiffusionTSFConfig
from .diffusion_model import DiffusionTSF
from .guidance import GuidanceModel

logger = logging.getLogger(__name__)

def apply_zero_phase_lowpass(x: torch.Tensor, cutoff_freq: float) -> torch.Tensor:
    """Apply a zero-phase Butterworth lowpass filter to the last dimension of x."""
    b, a = signal.butter(4, cutoff_freq, btype='low')
    
    # x shape: (B, V, SeqLen) or (B, SeqLen)
    x_np = x.detach().cpu().numpy()
    
    # Check if seqlen is large enough for the filter order
    # filtfilt requires len(x) > 3 * max(len(a), len(b))
    min_len = 3 * max(len(a), len(b))
    seq_len = x.shape[-1]
    
    if seq_len <= min_len:
        # If sequence is too short, just return the mean as the trend
        trend_np = np.mean(x_np, axis=-1, keepdims=True)
        trend_np = np.repeat(trend_np, seq_len, axis=-1)
    else:
        trend_np = signal.filtfilt(b, a, x_np, axis=-1)
        
    return torch.from_numpy(trend_np.copy()).to(x.device).float()

@dataclass
class ExperimentalDiffusionTSFConfig(DiffusionTSFConfig):
    # Experiment A: Deterministic Trend Residual Diffusion
    use_residual_diffusion: bool = False
    residual_cutoff_freq: float = 0.12  # Keeping lowest 12% frequencies
    guidance_forecast_is_lowpass: bool = False
    
    # Experiment B: Normalization Independence
    independent_norm: bool = False

    @property
    def visual_cond_channels(self) -> int:
        """Override to add an extra channel for lookback noise when using residual diffusion."""
        base = 1 + (1 if self.use_value_channel else 0)
        if self.use_residual_diffusion:
            return base + 1 # Add "ONLY lookback noise" channel
        return base


class ExperimentalDiffusionTSF(DiffusionTSF):
    def __init__(
        self,
        config: ExperimentalDiffusionTSFConfig,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None
    ):
        # We need to make sure the super().__init__ uses our overridden visual_cond_channels
        super().__init__(config, guidance_model)
        self.config = config
        
        # Experiment B: Learnable affine rescaling for past normalization path
        if self.config.independent_norm:
            # We add learnable scale and shift per variate
            self.affine_weight = nn.Parameter(torch.ones(1, self.config.num_variables, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, self.config.num_variables, 1))
            
    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Override normalization to support independent future norm."""
        if not getattr(self.config, 'independent_norm', False):
            return super()._normalize_sequence(past, future)
            
        # Independent norm: past is normalized by past stats
        past_mean = past.mean(dim=-1, keepdim=True)
        past_std = torch.clamp(
            past.std(dim=-1, keepdim=True),
            min=float(self.config.normalization_std_floor),
        )
        past_norm = (past - past_mean) / past_std
        
        # Apply learnable affine transform to past_norm
        if past_norm.dim() == 2: # univariate
            past_norm = past_norm * self.affine_weight.squeeze(1) + self.affine_bias.squeeze(1)
        else: # multivariate
            past_norm = past_norm * self.affine_weight + self.affine_bias
            
        future_norm = None
        if future is not None:
            # Independent norm: future is normalized by its OWN stats during training
            # The prompt says: "lookback noise representation, horizon prediction noise, and 
            # lookback noise+trend and horizon noise+trend should all be independently normalized"
            future_mean = future.mean(dim=-1, keepdim=True)
            future_std = torch.clamp(
                future.std(dim=-1, keepdim=True),
                min=float(self.config.normalization_std_floor),
            )
            future_norm = (future - future_mean) / future_std
            
        # At inference, denormalization uses PAST stats
        return past_norm, future_norm, (past_mean, past_std)

    def _get_guidance_forecast_norm(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int,
    ) -> torch.Tensor:
        """Override to properly handle the independent_norm case for guidance."""
        if self.guidance_model is None:
            raise ValueError("guidance model is None but guidance channel requested")
        
        mean, std = stats
        K = self.config.lookback_overlap
        H = forecast_length - K
        
        with torch.no_grad():
            coarse = self.guidance_model.get_forecast(past, H)
            
        # Standard normalization for ghost image
        coarse_norm = (coarse - mean) / std
            
        if K > 0:
            coarse_norm = torch.cat([past_norm[..., -K:], coarse_norm], dim=-1)
            
        return coarse_norm

    def _forward_factorized(self, past: torch.Tensor, future: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Override forward to inject residual logic and multi-channel lookback conditioning."""
        if not getattr(self.config, 'use_residual_diffusion', False):
            return super()._forward_factorized(past, future, t)
            
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        # Experiment A: compute trend and residual for HORIZON
        trend = apply_zero_phase_lowpass(future, self.config.residual_cutoff_freq)
        residual = future - trend

        # Normalization for past
        past_norm, _, stats = self._normalize_sequence(past, None)
        past_mean, past_std = stats
        
        # Independent normalization for horizon residual
        # "horizon prediction noise ... should all be independently normalized"
        res_mean = residual.mean(dim=-1, keepdim=True)
        res_std = torch.clamp(
            residual.std(dim=-1, keepdim=True),
            min=float(self.config.normalization_std_floor),
        )
        future_norm = (residual - res_mean) / res_std

        # Lookback noise extraction
        lookback_trend = apply_zero_phase_lowpass(past_norm, self.config.residual_cutoff_freq)
        lookback_noise = past_norm - lookback_trend
        
        # Independent normalization for lookback noise
        # "lookback noise representation ... should all be independently normalized"
        lb_noise_mean = lookback_noise.mean(dim=-1, keepdim=True)
        lb_noise_std = torch.clamp(
            lookback_noise.std(dim=-1, keepdim=True),
            min=float(self.config.normalization_std_floor),
        )
        lookback_noise_norm = (lookback_noise - lb_noise_mean) / lb_noise_std

        # Encode all to 2D
        past_2d = self.encode_to_2d(past_norm) # Lookback noise+trend
        past_noise_2d = self.encode_to_2d(lookback_noise_norm) # ONLY lookback noise
        
        future_2d = self.encode_to_2d(future_norm)
        
        # Apply dropout to conditioning
        past_2d = self._apply_coarse_dropout(past_2d)
        past_noise_2d = self._apply_coarse_dropout(past_noise_2d)

        W_past = past_2d.shape[3]
        W_fut = future_2d.shape[3]

        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        noisy_future, noise = self.scheduler.add_noise(future_2d, t)

        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            # "horizon noise+trend ... should all be independently normalized"
            # But the ghost image is just guidance, we already normalized it in _get_guidance_forecast_norm
            # If we wanted it truly independent, we'd recalculate stats for it.
            # The prompt says "independently normalized to fit in their respective CDF 2d representations"
            # encode_to_2d already handles the binning.
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)

        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        canvas = noisy_future.reshape(BV, 1, H, W_fut)
        canvas = self._inject_coordinate_channel(canvas)
        canvas = self._inject_time_channels(canvas)

        past_flat = past_2d.reshape(BV, 1, H, W_past)
        past_noise_flat = past_noise_2d.reshape(BV, 1, H, W_past)
        
        # Concatenate past noise+trend and ONLY past noise
        cond_past = torch.cat([past_flat, past_noise_flat], dim=1)
        cond_for_unet = F.interpolate(cond_past, size=(H, W_fut), mode='bilinear', align_corners=False)

        ctx_flat = None
        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            
            cond_for_unet = torch.where(drop_mask_flat.view(BV, 1, 1, 1), torch.zeros_like(cond_for_unet), cond_for_unet)
            if ctx_flat is not None:
                ctx_flat = torch.where(drop_mask_flat.view(BV, 1, 1), torch.zeros_like(ctx_flat), ctx_flat)
            if guidance_2d is not None:
                guidance_2d_flat = guidance_2d.reshape(BV, 1, H, W_fut)
                guidance_2d_flat = torch.where(drop_mask_flat.view(BV, 1, 1, 1), torch.zeros_like(guidance_2d_flat), guidance_2d_flat)
                canvas = torch.cat([canvas, guidance_2d_flat], dim=1)
        else:
            if guidance_2d is not None:
                canvas = torch.cat([canvas, guidance_2d.reshape(BV, 1, H, W_fut)], dim=1)

        chunk_size = self.config.unet_max_chunk_size
        if chunk_size > 0 and BV > chunk_size:
            noise_pred_flat_list = []
            for i in range(0, BV, chunk_size):
                end = min(i + chunk_size, BV)
                c_canvas = canvas[i:end]
                c_t = t_flat[i:end]
                c_cond = cond_for_unet[i:end]
                c_ctx = ctx_flat[i:end] if ctx_flat is not None else None
                c_out = self.noise_predictor(c_canvas, c_t, c_cond, encoder_hidden_states=c_ctx)
                noise_pred_flat_list.append(c_out)
            noise_pred_flat = torch.cat(noise_pred_flat_list, dim=0)
        else:
            noise_pred_flat = self.noise_predictor(canvas, t_flat, cond_for_unet, encoder_hidden_states=ctx_flat)
            
        noise_pred = noise_pred_flat.reshape(B, V, H, W_fut)

        K = self.config.lookback_overlap
        if K > 0:
            nl_past = F.mse_loss(noise_pred[..., :K], noise[..., :K])
            nl_fut  = F.mse_loss(noise_pred[..., K:],  noise[..., K:])
            noise_loss = self.config.past_loss_weight * nl_past + nl_fut
        else:
            noise_loss = F.mse_loss(noise_pred, noise)

        x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
        
        pred_1d, forecast_mse_loss, soft_dtw_loss = self._compute_forecast_losses(x0_pred, future_norm)

        loss = (
            noise_loss
            + self.config.forecast_mse_weight * forecast_mse_loss
            + self.config.soft_dtw_weight * soft_dtw_loss
        )

        return {
            'loss': loss,
            'noise_loss': noise_loss,
            'forecast_mse_loss': forecast_mse_loss,
            'soft_dtw_loss': soft_dtw_loss,
            'pred_1d': pred_1d,
            'noise_pred': noise_pred, 't': t,
        }

    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: Optional[float] = None,
        verbose: bool = False,
        decoder_method: str = "mean",
        beam_width: int = 5,
        jump_penalty_scale: float = 1.0,
        search_radius: int = 10,
        sampler: str = "ddim",
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Override generate to return detailed components with multi-channel conditioning."""
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        K = self.config.lookback_overlap
        W_fut = self.config.forecast_length

        past_norm, _, stats = self._normalize_sequence(past, None)
        past_mean, past_std = stats

        # Conditioning channels
        past_2d = self.encode_to_2d(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)

        if getattr(self.config, 'use_residual_diffusion', False):
            # Extract lookback noise for conditioning
            lookback_trend = apply_zero_phase_lowpass(past_norm, self.config.residual_cutoff_freq)
            lookback_noise = past_norm - lookback_trend
            lb_noise_mean = lookback_noise.mean(dim=-1, keepdim=True)
            lb_noise_std = torch.clamp(
                lookback_noise.std(dim=-1, keepdim=True),
                min=float(self.config.normalization_std_floor),
            )
            lookback_noise_norm = (lookback_noise - lb_noise_mean) / lb_noise_std
            past_noise_2d = self.encode_to_2d(lookback_noise_norm)
            past_noise_flat = past_noise_2d.reshape(BV, 1, H, W_past)
            cond_past = torch.cat([past_flat, past_noise_flat], dim=1)
            
            # Trend for prediction
            if self.guidance_model is not None:
                H_guidance = self.config.forecast_length - self.config.lookback_overlap
                guidance_pred = self.guidance_model.get_forecast(past, H_guidance)
                if getattr(self.config, 'guidance_forecast_is_lowpass', False):
                    trend = guidance_pred
                else:
                    trend = apply_zero_phase_lowpass(guidance_pred, self.config.residual_cutoff_freq)
            else:
                trend = torch.zeros(B, V, self.config.forecast_length - K, device=device)
        else:
            cond_past = past_flat
            trend = None

        cond_for_unet = F.interpolate(cond_past, size=(H, W_fut), mode='bilinear', align_corners=False)

        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)

        shape = (BV, 1, H, W_fut)

        ctx_flat = None
        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        def model_fn(x_t, t_batch, cond_arg):
            c_canvas = self._inject_coordinate_channel(x_t)
            c_canvas = self._inject_time_channels(c_canvas)
            if guidance_2d is not None:
                c_canvas = torch.cat([c_canvas, guidance_2d.reshape(BV, 1, H, W_fut)], dim=1)

            if cfg_scale is not None and cfg_scale != 1.0:
                out_cond = self.noise_predictor(c_canvas, t_batch, cond_for_unet, encoder_hidden_states=ctx_flat)
                
                # unconditioned
                c_canvas_uncond = c_canvas.clone()
                if guidance_2d is not None:
                    c_canvas_uncond[:, -1:] = 0
                cond_uncond = torch.zeros_like(cond_for_unet)
                ctx_uncond = torch.zeros_like(ctx_flat) if ctx_flat is not None else None
                
                out_uncond = self.noise_predictor(c_canvas_uncond, t_batch, cond_uncond, encoder_hidden_states=ctx_uncond)
                return out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                return self.noise_predictor(c_canvas, t_batch, cond_for_unet, encoder_hidden_states=ctx_flat)

        steps = num_inference_steps if num_inference_steps is not None else num_ddim_steps
        if sampler == "ddim":
            x = self.scheduler.sample_ddim(model=model_fn, shape=shape, cond=cond_for_unet, num_steps=steps, eta=eta, device=device, verbose=verbose)
        elif sampler == "ddpm":
            x = self.scheduler.sample_ddpm(model=model_fn, shape=shape, cond=cond_for_unet, device=device, verbose=verbose)
        else:
            x = self.scheduler.sample_ddim(model=model_fn, shape=shape, cond=cond_for_unet, num_steps=steps, eta=eta, device=device, verbose=verbose)

        x_2d = x.reshape(B, V, H, W_fut)
        decoded = self.decode_from_2d(
            x_2d, from_diffusion=True, decoder_method=decoder_method,
            beam_width=beam_width, jump_penalty_scale=jump_penalty_scale, search_radius=search_radius
        )

        if K > 0 and not self.config.unified_time_axis:
            decoded = decoded[..., K:]
        elif self.config.unified_time_axis:
            decoded = decoded[..., self.config.lookback_length:]

        if getattr(self.config, 'use_residual_diffusion', False):
            # Decoded is normalized residual. Best guess for scale is true_past_res_std.
            true_past_trend = apply_zero_phase_lowpass(past, self.config.residual_cutoff_freq)
            true_past_res = past - true_past_trend
            true_past_res_std = torch.clamp(
                true_past_res.std(dim=-1, keepdim=True),
                min=float(self.config.normalization_std_floor),
            )
            
            residual_pred = decoded * true_past_res_std
            final_pred = trend + residual_pred
            
            result = {
                "prediction": final_pred,
                "residual": residual_pred,
                "trend": trend,
                "future_2d": x_2d,
                "past_2d": past_2d,
                "past_noise_2d": past_noise_2d if 'past_noise_2d' in locals() else None
            }
            if guidance_2d is not None:
                result['guidance_2d'] = guidance_2d
            return result
        else:
            final_pred = decoded * past_std + past_mean
            result = {
                "prediction": final_pred,
                "future_2d": x_2d,
                "past_2d": past_2d
            }
            if guidance_2d is not None:
                result['guidance_2d'] = guidance_2d
            return result
