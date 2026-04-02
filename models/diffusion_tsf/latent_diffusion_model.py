"""
Latent-space diffusion time-series forecaster: frozen VAE + U-Net on scaled latents.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import LatentDiffusionConfig
    from .diffusion import DiffusionScheduler
    from .guidance import GuidanceModel, LinearRegressionGuidance
    from .preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
    from .unet import ConditionalUNet2D, TimeSeriesContextEncoder
    from .vae import TimeSeriesVAE
except ImportError:
    from config import LatentDiffusionConfig
    from diffusion import DiffusionScheduler
    from guidance import GuidanceModel, LinearRegressionGuidance
    from preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
    from unet import ConditionalUNet2D, TimeSeriesContextEncoder
    from vae import TimeSeriesVAE

logger = logging.getLogger(__name__)


class LatentDiffusionTSF(nn.Module):
    """Diffusion in VAE latent space with iTransformer ghost guidance (pixel → encode → latent)."""

    def __init__(
        self,
        config: LatentDiffusionConfig,
        vae: TimeSeriesVAE,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        if vae.latent_channels != config.latent_channels:
            raise ValueError(f"VAE latent_channels {vae.latent_channels} != config {config.latent_channels}")

        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma,
        )
        self.register_buffer(
            "decode_smoothing_kernel",
            self._build_decode_smoothing_kernel(sigma_x=3.0, sigma_y=1.0),
        )

        if config.use_guidance_channel:
            if guidance_model is not None:
                self.guidance_model = guidance_model
            else:
                self.guidance_model = LinearRegressionGuidance()
                logger.info("LatentDiffusionTSF: default LinearRegressionGuidance")
        else:
            self.guidance_model = None

        cz = config.latent_channels
        n_aux = config.num_aux_channels
        in_ch = cz + n_aux
        if config.use_guidance_channel:
            in_ch += cz
        h_lat = config.latent_image_height

        if config.use_hybrid_condition:
            self.context_encoder = TimeSeriesContextEncoder(
                input_channels=config.context_input_channels,
                embedding_dim=config.context_embedding_dim,
                num_layers=config.context_encoder_layers,
                num_heads=4,
                dropout=0.1,
                max_seq_len=max(config.lookback_length, config.forecast_length) + 256,
            )
        else:
            self.context_encoder = None

        self.noise_predictor = ConditionalUNet2D(
            in_channels=in_ch,
            out_channels=cz,
            channels=config.unet_channels,
            num_res_blocks=config.num_res_blocks,
            attention_levels=config.attention_levels,
            image_height=h_lat,
            kernel_size=config.unet_kernel_size,
            use_dilated_middle=config.use_dilated_middle,
            use_hybrid_condition=config.use_hybrid_condition,
            context_dim=config.context_embedding_dim,
            conditioning_mode=config.conditioning_mode,
            visual_cond_channels=cz,
            cond_in_channels=cz + n_aux,
        )

        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.noise_schedule,
        )

        logger.info(
            "LatentDiffusionTSF: H_lat=%s, in_ch=%s, cz=%s, aux=%s",
            h_lat,
            in_ch,
            cz,
            n_aux,
        )

    def to(self, device):
        super().to(device)
        self.scheduler = self.scheduler.to(device)
        return self

    @property
    def latent_overlap(self) -> int:
        return self.config.lookback_overlap // self.config.latent_spatial_downsample

    def _pixels_to_latent_scaled(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode_to_scaled_latent(x, sample=False)

    def _latent_scaled_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.decode_from_scaled_latent(z)

    def _build_decode_smoothing_kernel(self, sigma_x: float, sigma_y: float) -> torch.Tensor:
        size_x = int(6 * sigma_x + 1)
        size_y = int(6 * sigma_y + 1)
        if size_x % 2 == 0:
            size_x += 1
        if size_y % 2 == 0:
            size_y += 1
        x = torch.arange(size_x, dtype=torch.float32) - size_x // 2
        y = torch.arange(size_y, dtype=torch.float32) - size_y // 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size_y, size_x)

    def _get_coordinate_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        y_coords = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype)
        return y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)

    def _get_time_features(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ramp = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ramp = ramp.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        t_idx = torch.arange(width, device=device, dtype=dtype)
        sine = torch.sin(2 * math.pi * t_idx / self.config.seasonal_period)
        sine = sine.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return ramp, sine

    def _inject_coordinate_channel(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.use_coordinate_channel:
            return x
        b, _, h, w = x.shape
        coord = self._get_coordinate_grid(b, h, w, x.device, x.dtype)
        return torch.cat([x, coord], dim=1)

    def _inject_time_channels(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.use_time_ramp and not self.config.use_time_sine:
            return x
        b, _, h, w = x.shape
        ramp, sine = self._get_time_features(b, h, w, x.device, x.dtype)
        parts = [x]
        if self.config.use_time_ramp:
            parts.append(ramp)
        if self.config.use_time_sine:
            parts.append(sine)
        return torch.cat(parts, dim=1)

    def _get_value_channel(self, values_norm: torch.Tensor, height: int) -> torch.Tensor:
        if values_norm.dim() == 3:
            b, _nv, seq_len = values_norm.shape
            vc = values_norm.unsqueeze(2).expand(-1, -1, height, -1)
        else:
            b, seq_len = values_norm.shape
            vc = values_norm.unsqueeze(1).unsqueeze(2).expand(-1, -1, height, -1)
        vc = vc.clamp(-self.config.max_scale, self.config.max_scale)
        return vc / self.config.max_scale

    def _inject_value_channel(self, x: torch.Tensor, values_norm: torch.Tensor) -> torch.Tensor:
        if not self.config.use_value_channel:
            return x
        _, _, height, _ = x.shape
        vc = self._get_value_channel(values_norm, height)
        if vc.shape[1] > 1:
            vc = vc[:, 0:1, :, :]
        return torch.cat([x, vc], dim=1)

    def _inject_guidance_channel(self, x: torch.Tensor, guidance: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.config.use_guidance_channel or guidance is None:
            return x
        return torch.cat([x, guidance], dim=1)

    def _pad_to_window(self, tensor: torch.Tensor, mode: str, total_length: int) -> torch.Tensor:
        b, c, h, length = tensor.shape
        if length >= total_length:
            return tensor[..., :total_length]
        pad = total_length - length
        if mode == "past":
            return F.pad(tensor, (0, pad, 0, 0))
        if mode == "future":
            return F.pad(tensor, (pad, 0, 0, 0))
        raise ValueError(mode)

    def encode_to_2d(self, x: torch.Tensor, scale_for_diffusion: bool = True) -> torch.Tensor:
        image = self.to_2d(x)
        blurred = self.blur(image)
        if scale_for_diffusion:
            return blurred.clamp(min=0.0, max=1.0) * 2.0 - 1.0
        return blurred

    def decode_from_2d(
        self,
        image: torch.Tensor,
        from_diffusion: bool = True,
        decoder_method: str = "mean",
    ) -> torch.Tensor:
        b, num_vars, height, seq_len = image.shape
        squeeze_output = num_vars == 1
        if from_diffusion:
            cdf_map = (image + 1.0) / 2.0
        else:
            cdf_map = image
        cdf_map = torch.clamp(cdf_map, min=0.0, max=1.0)
        sharpen = (
            self.config.decode_temperature
            if decoder_method == "expectation" and from_diffusion
            else None
        )
        inv = "expectation" if decoder_method == "expectation" else "mean"
        return self.to_2d.inverse(
            cdf_map,
            cdf_decoder=inv,
            expectation_sharpen_temp=sharpen,
            squeeze_univariate=squeeze_output,
        )

    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mean = past.mean(dim=-1, keepdim=True)
        std = past.std(dim=-1, keepdim=True) + 1e-8
        past_norm = (past - mean) / std
        future_norm = (future - mean) / std if future is not None else None
        return past_norm, future_norm, (mean, std)

    def _denormalize(self, x: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, std = stats
        if x.dim() == 2:
            m = mean.reshape(mean.shape[0], 1)
            s = std.reshape(std.shape[0], 1)
            return x * s + m
        return x * std + mean

    def _generate_guidance_2d(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int,
    ) -> torch.Tensor:
        if self.guidance_model is None:
            raise ValueError("guidance_model is None")
        K = self.config.lookback_overlap
        H = forecast_length - K
        mean, std = stats
        with torch.no_grad():
            coarse = self.guidance_model.get_forecast(past, H)
        coarse_norm = (coarse - mean) / std
        if K > 0:
            overlap_norm = past_norm[..., -K:]
            coarse_norm = torch.cat([overlap_norm, coarse_norm], dim=-1)
        return self.encode_to_2d(coarse_norm, scale_for_diffusion=True)

    def _prepare_visual_conditioning(self, past_z: torch.Tensor, target_width: int) -> torch.Tensor:
        _, _, h, past_len = past_z.shape
        if past_len >= target_width:
            return past_z[:, :, :, -target_width:]
        return F.interpolate(past_z, size=(h, target_width), mode="bilinear", align_corners=False)

    def _prepare_1d_context(self, past_norm: torch.Tensor) -> torch.Tensor:
        if past_norm.dim() == 3:
            past_1d = past_norm[:, 0, :]
        else:
            past_1d = past_norm
        b, seq_len = past_1d.shape
        device, dtype = past_1d.device, past_1d.dtype
        time_idx = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
        time_idx = time_idx.unsqueeze(0).expand(b, -1)
        return torch.stack([past_1d, time_idx], dim=-1)

    def _apply_coarse_dropout(self, image: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.cutout_prob <= 0:
            return image
        if torch.rand(1, device=image.device).item() >= self.config.cutout_prob:
            return image
        b, c, h, w = image.shape
        num_masks = torch.randint(
            self.config.cutout_min_masks,
            self.config.cutout_max_masks + 1,
            (1,),
            device=image.device,
        ).item()
        for _ in range(num_masks):
            shape_idx = torch.randint(0, len(self.config.cutout_shapes), (1,), device=image.device).item()
            mask_h, mask_w = self.config.cutout_shapes[shape_idx]
            mask_h = min(mask_h, h)
            mask_w = min(mask_w, w)
            if mask_h <= 0 or mask_w <= 0:
                continue
            top_max = max(1, h - mask_h + 1)
            left_max = max(1, w - mask_w + 1)
            top = torch.randint(0, top_max, (1,), device=image.device).item()
            left = torch.randint(0, left_max, (1,), device=image.device).item()
            image[:, :, top : top + mask_h, left : left + mask_w] = -1.0
        return image

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = past.device
        b = past.shape[0]
        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        past_2d = self.encode_to_2d(past_norm)
        future_2d = self.encode_to_2d(future_norm)
        past_2d = self._apply_coarse_dropout(past_2d)

        with torch.no_grad():
            z_past = self.vae.encode_to_scaled_latent(past_2d, sample=False)
            z_future = self.vae.encode_to_scaled_latent(future_2d, sample=False)

        past_len = z_past.shape[-1]
        future_len = z_future.shape[-1]
        total_len = past_len + future_len

        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (b,), device=device)

        noisy_future, noise = self.scheduler.add_noise(z_future, t)

        encoder_hidden_states = None
        if self.context_encoder is not None:
            encoder_hidden_states = self.context_encoder(self._prepare_1d_context(past_norm))

        guidance_lat = None
        if self.config.use_guidance_channel:
            g2d = self._generate_guidance_2d(past, past_norm, stats, future_2d.shape[-1])
            with torch.no_grad():
                guidance_lat = self.vae.encode_to_scaled_latent(g2d, sample=False)

        if self.config.unified_time_axis:
            past_pad = self._pad_to_window(z_past, "past", total_len)
            noisy_pad = self._pad_to_window(noisy_future, "future", total_len)
            canvas = past_pad + noisy_pad
            canvas = self._inject_coordinate_channel(canvas)
            canvas = self._inject_time_channels(canvas)
            if self.config.use_value_channel:
                past_vals = past_norm if past_norm.dim() == 3 else past_norm.unsqueeze(1)
                vals_pad = F.pad(past_vals, (0, future_2d.shape[-1]))
                h_lat = self.config.latent_image_height
                w_lat = total_len
                val_ch = self._get_value_channel(vals_pad, self.config.image_height)
                if val_ch.shape[1] > 1:
                    val_ch = val_ch[:, 0:1, :, :]
                val_ch = F.interpolate(val_ch, size=(h_lat, w_lat), mode="bilinear", align_corners=False)
                canvas = torch.cat([canvas, val_ch], dim=1)
            if guidance_lat is not None:
                gp = self._pad_to_window(guidance_lat, "future", total_len)
                canvas = self._inject_guidance_channel(canvas, gp)

            cond = past_pad
            if self.config.use_value_channel:
                cond = torch.cat([cond, val_ch], dim=1)

            noise_pred_full = self.noise_predictor(canvas, t, cond, encoder_hidden_states=encoder_hidden_states)
            noise_pred = noise_pred_full[..., past_len:]
        else:
            canvas = noisy_future
            canvas = self._inject_coordinate_channel(canvas)
            canvas = self._inject_time_channels(canvas)
            val_ch = None
            if self.config.use_value_channel:
                if past_norm.dim() == 3:
                    last_val = past_norm[:, :, -1:]
                    last_val_exp = last_val.expand(-1, -1, future_2d.shape[-1])
                else:
                    last_val = past_norm[:, -1:]
                    last_val_exp = last_val.expand(-1, future_2d.shape[-1])
                val_ch = self._get_value_channel(last_val_exp, self.config.image_height)
                if val_ch.shape[1] > 1:
                    val_ch = val_ch[:, 0:1, :, :]
                val_ch = F.interpolate(
                    val_ch,
                    size=(self.config.latent_image_height, future_len),
                    mode="bilinear",
                    align_corners=False,
                )
                canvas = torch.cat([canvas, val_ch], dim=1)
            if guidance_lat is not None:
                canvas = self._inject_guidance_channel(canvas, guidance_lat)
            cond = self._prepare_visual_conditioning(z_past, target_width=future_len)
            if val_ch is not None:
                cond = torch.cat([cond, val_ch], dim=1)
            noise_pred = self.noise_predictor(canvas, t, cond, encoder_hidden_states=encoder_hidden_states)

        K = self.latent_overlap
        if K > 0:
            noise_loss = self.config.past_loss_weight * F.mse_loss(
                noise_pred[..., :K], noise[..., :K]
            ) + F.mse_loss(noise_pred[..., K:], noise[..., K:])
        else:
            noise_loss = F.mse_loss(noise_pred, noise)

        x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        if self.config.emd_lambda > 0:
            aux = self.config.emd_lambda * F.mse_loss(x0_pred, z_future)
        else:
            aux = torch.tensor(0.0, device=device)

        loss = noise_loss + aux
        out = {
            "loss": loss,
            "noise_loss": noise_loss,
            "noise_pred": noise_pred,
            "t": t,
        }
        if guidance_lat is not None:
            out["guidance_lat"] = guidance_lat
        return out

    def set_guidance_model(self, guidance_model: Optional[Union[GuidanceModel, nn.Module]]) -> None:
        if guidance_model is None and self.config.use_guidance_channel:
            raise ValueError("use_guidance_channel=True requires a guidance model")
        self.guidance_model = guidance_model

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
    ) -> Dict[str, torch.Tensor]:
        device = past.device
        b = past.shape[0]
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        z_past = self.vae.encode_to_scaled_latent(past_2d, sample=False)
        past_len = z_past.shape[-1]

        fp = self.config.forecast_length
        future_len = self.vae.encode_to_scaled_latent(
            self.encode_to_2d(torch.zeros(b, 1, fp, device=device, dtype=past_norm.dtype)),
            sample=False,
        ).shape[-1]
        total_len = past_len + future_len

        encoder_hidden_states = None
        null_encoder_hidden_states = None
        if self.context_encoder is not None:
            ctx = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(ctx)
            if cfg_scale > 1.0:
                null_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        guidance_lat = None
        guidance_pad = None
        null_guidance_pad = None
        if self.config.use_guidance_channel:
            g2d = self._generate_guidance_2d(past, past_norm, stats, fp)
            guidance_lat = self.vae.encode_to_scaled_latent(g2d, sample=False)
            if self.config.unified_time_axis:
                guidance_pad = self._pad_to_window(guidance_lat, "future", total_len)
                if cfg_scale > 1.0:
                    null_guidance_pad = torch.zeros_like(guidance_pad)
            else:
                guidance_pad = guidance_lat
                if cfg_scale > 1.0:
                    null_guidance_pad = torch.zeros_like(guidance_pad)

        past_pad = self._pad_to_window(z_past, "past", total_len) if self.config.unified_time_axis else None

        if self.config.unified_time_axis:
            val_ch = None
            if self.config.use_value_channel:
                past_vals = past_norm if past_norm.dim() == 3 else past_norm.unsqueeze(1)
                vals_pad = F.pad(past_vals, (0, fp))
                val_ch = self._get_value_channel(vals_pad, self.config.image_height)
                if val_ch.shape[1] > 1:
                    val_ch = val_ch[:, 0:1, :, :]
                val_ch = F.interpolate(
                    val_ch,
                    size=(self.config.latent_image_height, total_len),
                    mode="bilinear",
                    align_corners=False,
                )
            cond = past_pad
            if val_ch is not None:
                cond = torch.cat([cond, val_ch], dim=1)
        else:
            val_ch = None
            if self.config.use_value_channel:
                if past_norm.dim() == 3:
                    last_val = past_norm[:, :, -1:]
                    last_val_exp = last_val.expand(-1, -1, fp)
                else:
                    last_val = past_norm[:, -1:]
                    last_val_exp = last_val.expand(-1, fp)
                val_ch = self._get_value_channel(last_val_exp, self.config.image_height)
                if val_ch.shape[1] > 1:
                    val_ch = val_ch[:, 0:1, :, :]
                val_ch = F.interpolate(
                    val_ch,
                    size=(self.config.latent_image_height, future_len),
                    mode="bilinear",
                    align_corners=False,
                )
            cond = self._prepare_visual_conditioning(z_past, target_width=future_len)
            if val_ch is not None:
                cond = torch.cat([cond, val_ch], dim=1)

        null_cond = torch.zeros_like(cond) if cfg_scale > 1.0 else None

        cz = self.config.latent_channels
        h_lat = self.config.latent_image_height
        noise_shape = (b, cz, h_lat, future_len)

        def model_fn(x_fut, t, c, use_null_ctx=False, use_null_guide=False):
            if self.config.unified_time_axis:
                x_pad = self._pad_to_window(x_fut, "future", total_len)
                canv = past_pad + x_pad
                canv = self._inject_coordinate_channel(canv)
                canv = self._inject_time_channels(canv)
                if val_ch is not None:
                    canv = torch.cat([canv, val_ch], dim=1)
                if self.config.use_guidance_channel:
                    g = null_guidance_pad if use_null_guide else guidance_pad
                    canv = self._inject_guidance_channel(canv, g)
                ctx = null_encoder_hidden_states if use_null_ctx else encoder_hidden_states
                full = self.noise_predictor(canv, t, c, encoder_hidden_states=ctx)
                return full[..., past_len:]
            canv = x_fut
            canv = self._inject_coordinate_channel(canv)
            canv = self._inject_time_channels(canv)
            if val_ch is not None:
                canv = torch.cat([canv, val_ch], dim=1)
            if self.config.use_guidance_channel:
                g = null_guidance_pad if use_null_guide else guidance_pad
                canv = self._inject_guidance_channel(canv, g)
            ctx = null_encoder_hidden_states if use_null_ctx else encoder_hidden_states
            return self.noise_predictor(canv, t, c, encoder_hidden_states=ctx)

        def model_cfg(x, t, c, null_c, scale):
            if scale <= 1.0:
                return model_fn(x, t, c)
            o = model_fn(x, t, c, use_null_ctx=False, use_null_guide=False)
            u = model_fn(x, t, null_c, use_null_ctx=True, use_null_guide=True)
            return u + scale * (o - u)

        if use_ddim:
            z_future = self.scheduler.sample_ddim_cfg(
                model=lambda x, t, c: model_cfg(x, t, c, null_cond, cfg_scale),
                shape=noise_shape,
                cond=cond,
                null_cond=null_cond,
                cfg_scale=1.0,
                num_steps=num_ddim_steps,
                eta=eta,
                device=device,
                verbose=verbose,
            )
        else:
            z_future = self.scheduler.sample_ddpm_cfg(
                model=lambda x, t, c: model_cfg(x, t, c, null_cond, cfg_scale),
                shape=noise_shape,
                cond=cond,
                null_cond=null_cond,
                cfg_scale=1.0,
                device=device,
                verbose=verbose,
            )

        future_2d = self._latent_scaled_to_pixels(z_future)
        future_norm = self.decode_from_2d(future_2d, decoder_method=decoder_method)
        future = self._denormalize(future_norm, stats)
        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]
        if self.config.num_variables == 1:
            if future.dim() == 2:
                future = future.unsqueeze(1)
            if future_norm.dim() == 2:
                future_norm = future_norm.unsqueeze(1)

        return {
            "prediction": future,
            "prediction_norm": future_norm,
            "future_2d": future_2d,
            "past_2d": past_2d,
        }

    def get_loss(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.forward(past, future)["loss"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.guidance_model is not None:
            self.guidance_model.eval()
        return self
