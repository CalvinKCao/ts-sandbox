"""Hybrid 1D diffusion with soft render -> U-Net -> vertical readout -> 1D epsilon head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiffusionTSFConfig
from .diffusion_model import DiffusionTSF
from .guidance import GuidanceModel
from .soft_renderer import SoftGaussianRenderer
from .unet import ConditionalUNet2D


class VerticalAttentionReadout(nn.Module):
    """Pool U-Net features over the value axis into per-time embeddings."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (N, C, H, W) -> (N, C, W)
        weights = torch.softmax(self.score(feats), dim=2)
        return (feats * weights).sum(dim=2)


@dataclass
class HybridLearnedDiffusionConfig(DiffusionTSFConfig):
    one_d_loss_weight: float = 1.0
    x0_loss_weight: float = 1.0
    cdf_loss_weight: float = 0.2


class HybridLearnedDiffusionTSF(DiffusionTSF):
    """1D DDPM state with soft-rendered U-Net canvas; L1d + soft-render EMD."""

    def __init__(
        self,
        config: HybridLearnedDiffusionConfig,
        guidance_model: Optional[GuidanceModel] = None,
    ):
        super().__init__(config, guidance_model)
        self.hybrid_config = config
        feat_dim = config.unet_channels[0]
        self.soft_renderer = SoftGaussianRenderer(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.vertical_readout = VerticalAttentionReadout(feat_dim)
        self.noise_1d_head = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(feat_dim, 1, kernel_size=1),
        )

    def _soft_encode(self, x_1d: torch.Tensor) -> torch.Tensor:
        cdf = self.soft_renderer(x_1d)
        return self.soft_renderer.to_diffusion_range(cdf)

    def _noise_loss_weighted(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        K = self.config.lookback_overlap
        if K > 0:
            nl_past = F.mse_loss(pred[..., :K], target[..., :K])
            nl_fut = F.mse_loss(pred[..., K:], target[..., K:])
            return self.config.past_loss_weight * nl_past + nl_fut
        return F.mse_loss(pred, target)

    def _predict_eps_1d_from_canvas(
        self,
        canvas: torch.Tensor,
        t_flat: torch.Tensor,
        cond_for_unet: torch.Tensor,
        ctx_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        chunk_size = self.config.unet_max_chunk_size
        BV = canvas.shape[0]
        predictor = self.noise_predictor
        if not isinstance(predictor, ConditionalUNet2D):
            raise RuntimeError("Hybrid learned_render requires ConditionalUNet2D backbone")

        if chunk_size > 0 and BV > chunk_size:
            feat_list = []
            for i in range(0, BV, chunk_size):
                end = min(i + chunk_size, BV)
                _, feat_c = predictor(
                    canvas[i:end],
                    t_flat[i:end],
                    cond_for_unet[i:end],
                    encoder_hidden_states=ctx_flat[i:end] if ctx_flat is not None else None,
                    return_features=True,
                )
                feat_list.append(feat_c)
            feats = torch.cat(feat_list, dim=0)
        else:
            _, feats = predictor(
                canvas,
                t_flat,
                cond_for_unet,
                encoder_hidden_states=ctx_flat,
                return_features=True,
            )

        pooled = self.vertical_readout(feats)
        return self.noise_1d_head(pooled).squeeze(1)

    def _build_training_context(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        past_2d: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        W_fut: int,
        B: int,
        V: int,
        H: int,
        BV: int,
        device: torch.device,
    ):
        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(
                past, past_norm, stats, W_fut
            )
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)

        ctx = (
            None
            if getattr(self.config, "disable_cross_attention", False)
            else self._get_cross_variate_context(past)
        )

        past_flat = past_2d.reshape(BV, 1, H, past_2d.shape[3])
        cond_for_unet = F.interpolate(
            past_flat, size=(H, W_fut), mode="bilinear", align_corners=False
        )

        ctx_flat = None
        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        return cond_for_unet, ctx_flat, guidance_2d

    def _apply_cfg_dropout(
        self,
        B: int,
        V: int,
        BV: int,
        device: torch.device,
        canvas: torch.Tensor,
        cond_for_unet: torch.Tensor,
        ctx_flat: Optional[torch.Tensor],
        guidance_2d: Optional[torch.Tensor],
        H: int,
        W_fut: int,
    ) -> torch.Tensor:
        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            cond_for_unet = torch.where(
                drop_mask_flat.view(BV, 1, 1, 1),
                torch.zeros_like(cond_for_unet),
                cond_for_unet,
            )
            if ctx_flat is not None:
                ctx_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1),
                    torch.zeros_like(ctx_flat),
                    ctx_flat,
                )
            if guidance_2d is not None:
                g_flat = guidance_2d.reshape(BV, 1, H, W_fut)
                g_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1, 1),
                    torch.zeros_like(g_flat),
                    g_flat,
                )
                canvas = torch.cat([canvas, g_flat], dim=1)
        elif guidance_2d is not None:
            canvas = torch.cat([canvas, guidance_2d.reshape(BV, 1, H, W_fut)], dim=1)
        return canvas

    def _forward_factorized(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        past_2d = self.encode_to_2d(past_norm)
        past_2d = self._apply_coarse_dropout(past_2d)
        future_2d = self.encode_to_2d(future_norm)

        W_fut = future_2d.shape[3]
        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        noisy_1d, noise_1d = self.scheduler.add_noise(future_norm, t)

        cond_for_unet, ctx_flat, guidance_2d = self._build_training_context(
            past, past_norm, past_2d, stats, W_fut, B, V, H, BV, device
        )
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)

        soft_canvas = self._soft_encode(noisy_1d)
        canvas = soft_canvas.reshape(BV, 1, H, W_fut)
        canvas = self._inject_coordinate_channel(canvas)
        canvas = self._inject_time_channels(canvas)
        canvas = self._apply_cfg_dropout(
            B, V, BV, device, canvas, cond_for_unet, ctx_flat, guidance_2d, H, W_fut
        )

        noise_pred_1d_flat = self._predict_eps_1d_from_canvas(
            canvas, t_flat, cond_for_unet, ctx_flat
        )
        noise_pred_1d = noise_pred_1d_flat.reshape(B, V, W_fut)

        noise_loss_1d = self._noise_loss_weighted(noise_pred_1d, noise_1d)

        x0_1d = self.scheduler.predict_x0_from_noise(noisy_1d, t, noise_pred_1d)
        x0_clamp = self.config.max_scale
        x0_1d = torch.clamp(x0_1d, -x0_clamp, x0_clamp)
        x0_loss = self._noise_loss_weighted(x0_1d, future_norm)

        soft_x0 = self._soft_encode(x0_1d)
        emd_loss = self._compute_emd_loss(soft_x0, future_2d)

        cfg = self.hybrid_config
        loss = (
            cfg.one_d_loss_weight * noise_loss_1d
            + cfg.x0_loss_weight * x0_loss
            + cfg.cdf_loss_weight * emd_loss
        )

        return {
            "loss": loss,
            "noise_loss": noise_loss_1d,
            "noise_loss_1d": noise_loss_1d,
            "x0_loss": x0_loss,
            "emd_loss": emd_loss,
            "noise_pred": noise_pred_1d,
            "t": t,
            "learned_sigma": self.soft_renderer.sigma.detach(),
        }

    @torch.no_grad()
    def _generate_factorized(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: Optional[float] = None,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del decoder_method, kwargs
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        W_past = past_2d.shape[3]
        W_fut = self.config.forecast_length

        cond_for_unet, ctx_flat, guidance_2d = self._build_training_context(
            past, past_norm, past_2d, stats, W_fut, B, V, H, BV, device
        )
        cond_flat = cond_for_unet
        null_cond = torch.zeros_like(cond_flat) if cfg_scale > 1.0 else None

        guide_flat = None
        if guidance_2d is not None:
            guide_flat = guidance_2d.reshape(BV, 1, H, W_fut)
        null_guide = (
            torch.zeros_like(guide_flat) if (guide_flat is not None and cfg_scale > 1.0) else None
        )

        null_ctx_flat = (
            torch.zeros_like(ctx_flat) if (ctx_flat is not None and cfg_scale > 1.0) else None
        )

        def _build_canvas_from_1d(x_1d_flat: torch.Tensor, use_null: bool = False) -> torch.Tensor:
            x_2d = self._soft_encode(x_1d_flat.reshape(B, V, W_fut)).reshape(BV, 1, H, W_fut)
            c = self._inject_coordinate_channel(x_2d)
            c = self._inject_time_channels(c)
            if guide_flat is not None:
                c = torch.cat([c, null_guide if use_null else guide_flat], dim=1)
            return c

        def _predict_eps_1d(x_1d_flat: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
            canvas = _build_canvas_from_1d(x_1d_flat)
            return self._predict_eps_1d_from_canvas(canvas, t_batch, cond_flat, ctx_flat)

        def model_fn(x_1d: torch.Tensor, t_batch: torch.Tensor, cond_arg) -> torch.Tensor:
            del cond_arg
            if cfg_scale <= 1.0:
                return _predict_eps_1d(x_1d, t_batch)
            eps_c = _predict_eps_1d(x_1d, t_batch)
            canvas_u = _build_canvas_from_1d(x_1d, use_null=True)
            eps_u = self._predict_eps_1d_from_canvas(canvas_u, t_batch, null_cond, null_ctx_flat)
            return eps_u + cfg_scale * (eps_c - eps_u)

        noise_shape = (BV, W_fut)

        x0_clamp = self.config.max_scale

        if sampler == "dpmpp":
            steps = num_inference_steps if num_inference_steps is not None else 20
            future_1d_flat = self.scheduler.sample_dpmpp(
                model=model_fn,
                shape=noise_shape,
                cond=cond_flat,
                num_steps=steps,
                device=device,
                verbose=verbose,
                x0_clamp=x0_clamp,
            )
        elif use_ddim:
            steps = num_inference_steps if num_inference_steps is not None else num_ddim_steps
            future_1d_flat = self.scheduler.sample_ddim_cfg(
                model=model_fn,
                shape=noise_shape,
                cond=cond_flat,
                null_cond=null_cond,
                cfg_scale=1.0,
                num_steps=steps,
                eta=eta,
                device=device,
                verbose=verbose,
                x0_clamp=x0_clamp,
            )
        else:
            future_1d_flat = self.scheduler.sample_ddpm_cfg(
                model=model_fn,
                shape=noise_shape,
                cond=cond_flat,
                null_cond=null_cond,
                cfg_scale=1.0,
                device=device,
                verbose=verbose,
            )

        future_norm = future_1d_flat.reshape(B, V, W_fut)
        future = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

        return {
            "prediction": future,
            "prediction_norm": future_norm,
            "past_2d": past_2d,
        }
