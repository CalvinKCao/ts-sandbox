"""
Complete Diffusion-based Time Series Forecasting Model.

stuff in here:
- Preprocessing (norm, 2D encoding, blur)
- Conditional U-Net
- DDPM/DDIM diffusion
- 2D to 1D decoding
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, Union

from .config import DiffusionTSFConfig
from .preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
from .unet import ConditionalUNet2D, iTransformerTokenAdapter
from .diffusion import DiffusionScheduler
from .guidance import GuidanceModel, LinearRegressionGuidance
from .metrics import monotonicity_loss
from .dit import FactorizedDiT

logger = logging.getLogger(__name__)


def beam_search_decoder(
    cdf_map: torch.Tensor,
    bin_centers: torch.Tensor,
    beam_width: int = 5,
    jump_penalty_scale: float = 1.0,
    search_radius: int = 10,
    eps: float = 1e-8
) -> torch.Tensor:
    """beam search for decoding the CDF/occupancy maps.
    
    finds a path through the prob map. tries to maximize likelihood while 
    punishing big jumps between time steps so it stays smoothish.
    """
    batch_size, height, seq_len = cdf_map.shape
    device = cdf_map.device
    
    # 1. cdf to pdf conversion: drop[y] = cdf[y] - cdf[y+1]
    # occupancy is high at bottom, low at top. 
    # stick some zeros at the top so shape matches
    pdf = torch.zeros_like(cdf_map)
    pdf[:, :-1, :] = cdf_map[:, :-1, :] - cdf_map[:, 1:, :]
    pdf = torch.clamp(pdf, min=0.0)
    
    # norm the pdf per column
    pdf_sum = pdf.sum(dim=1, keepdim=True).clamp(min=eps)
    pdf = pdf / pdf_sum
    
    # log probs (clamped to avoid -inf explosion)
    log_pdf = torch.log(pdf.clamp(min=eps))  # (batch, height, seq_len)
    
    results = []
    
    for b in range(batch_size):
        log_pdf_b = log_pdf[b]  # (height, seq_len)
        
        # start with top beam_width positions at t=0
        init_scores = log_pdf_b[:, 0]  # (height,)
        topk_scores, topk_indices = init_scores.topk(min(beam_width, height))
        
        # beams: (score, [path indices])
        beam_scores = topk_scores  # (beam_width,)
        beam_paths = topk_indices.unsqueeze(1)  # (beam_width, 1)
        
        # walk through time
        for t in range(1, seq_len):
            num_beams = beam_scores.shape[0]
            
            # current ends for each beam
            prev_positions = beam_paths[:, -1]  # (num_beams,)
            
            # For each beam, compute scores for all possible next positions
            # within search_radius
            all_candidates_scores = []
            all_candidates_paths = []
            
            for beam_idx in range(num_beams):
                prev_pos = prev_positions[beam_idx].item()
                prev_score = beam_scores[beam_idx]
                
                # Define search window
                lo = max(0, prev_pos - search_radius)
                hi = min(height, prev_pos + search_radius + 1)
                
                # Candidate positions and their scores
                candidates = torch.arange(lo, hi, device=device)
                candidate_log_probs = log_pdf_b[lo:hi, t]
                
                # Jump penalties
                jumps = (candidates - prev_pos).abs().float()
                penalties = jump_penalty_scale * jumps
                
                # Total scores
                candidate_scores = prev_score + candidate_log_probs - penalties
                
                # Store candidates
                for i, pos in enumerate(candidates):
                    all_candidates_scores.append(candidate_scores[i])
                    new_path = torch.cat([beam_paths[beam_idx], pos.unsqueeze(0)])
                    all_candidates_paths.append(new_path)
            
            if len(all_candidates_scores) == 0:
                # Edge case: no valid candidates, keep current beams
                continue
            
            # Stack and prune to top beam_width
            all_scores = torch.stack(all_candidates_scores)
            topk_count = min(beam_width, len(all_scores))
            topk_scores, topk_idx = all_scores.topk(topk_count)
            
            beam_scores = topk_scores
            beam_paths = torch.stack([all_candidates_paths[i] for i in topk_idx.tolist()])
        
        # Select best path
        best_idx = beam_scores.argmax()
        best_path = beam_paths[best_idx]  # (seq_len,) indices
        
        # Convert indices to values using bin_centers
        path_values = bin_centers[best_path.long()]
        results.append(path_values)
    
    return torch.stack(results)  # (batch, seq_len)


class DiffusionTSF(nn.Module):
    """Diffusion-based Time Series Forecasting Model.
    
    Pipeline:
    1. Normalize input (past + future) using local mean/std
    2. Convert to 2D "stripe" representation
    3. Apply vertical Gaussian blur
    4. Train U-Net to denoise future conditioned on past
    5. At inference: generate future via DDPM/DDIM
    6. Decode 2D representation back to 1D
    
    Optional Hybrid "Visual Guide" mode (use_guidance_channel=True):
    - A Stage 1 predictor (e.g., iTransformer) generates a coarse forecast
    - The coarse forecast is converted to a 2D "ghost image"
    - This ghost image is concatenated to the U-Net input
    - The diffusion model focuses on refining texture/residuals
    """
    
    def __init__(
        self,
        config: DiffusionTSFConfig,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None
    ):
        """
        Args:
            config: Model configuration
            guidance_model: Optional Stage 1 predictor for hybrid forecasting.
                           If config.use_guidance_channel is True but no model
                           is provided, a LinearRegressionGuidance is used as default.
        """
        super().__init__()
        self.config = config
        
        # Preprocessing modules
        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma
        )
        
        # Guidance model for hybrid "visual guide" forecasting
        if config.use_guidance_channel:
            if guidance_model is not None:
                self.guidance_model = guidance_model
            else:
                # Default to linear regression if no model provided
                self.guidance_model = LinearRegressionGuidance()
                logger.info("Using default LinearRegressionGuidance for guidance channel")
        else:
            self.guidance_model = None
        
        # Noise prediction backbone (U-Net or Transformer)
        # Input channels: num_variables (data) + aux channels (coord, time_ramp, time_sine)
        # Use the config property for consistent calculation
        backbone_in_channels = config.backbone_in_channels
        
        if config.model_type == "dit":
            self.noise_predictor = FactorizedDiT(
                in_channels=backbone_in_channels,
                cond_channels=config.visual_cond_channels,
                out_channels=1,
                image_height=config.image_height,
                patch_size=config.dit_patch_size,
                embed_dim=config.dit_embed_dim,
                depth=config.dit_depth,
                num_heads=config.dit_num_heads,
                mlp_ratio=config.dit_mlp_ratio,
                dropout=config.dit_dropout,
                context_dim=config.context_embedding_dim,
                gradient_checkpointing=config.use_gradient_checkpointing,
            )
        else:
            self.noise_predictor = ConditionalUNet2D(
                in_channels=backbone_in_channels,
                out_channels=1,
                channels=config.unet_channels,
                num_res_blocks=config.num_res_blocks,
                attention_levels=config.attention_levels,
                image_height=config.image_height,
                kernel_size=config.unet_kernel_size,
                use_dilated_middle=config.use_dilated_middle,
                context_dim=config.context_embedding_dim,
                visual_cond_channels=config.visual_cond_channels,
                use_gradient_checkpointing=config.use_gradient_checkpointing,
            )

        # iTransformer token adapter is shared by both backbones; it produces the
        # (B, V, ctx_dim) cross-attention memory consumed at the bottleneck.
        # Use a large enough max_variates (default 512) to stay compatible with
        # pretrained checkpoints even when finetuning on fewer variables.
        self.context_encoder = iTransformerTokenAdapter(
            d_model=config.itrans_d_model,
            context_dim=config.context_embedding_dim,
            max_variates=max(config.num_variables, 512),
            dropout=0.1,
        )

        # Diffusion scheduler (not a nn.Module, managed separately)
        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.noise_schedule
        )

        logger.info(f"DiffusionTSF initialized:")
        logger.info(f"  Variables: {config.num_variables} ({'multivariate' if config.num_variables > 1 else 'univariate'})")
        logger.info(f"  Lookback: {config.lookback_length}, Forecast: {config.forecast_length}")
        logger.info(
            f"  Image size: {config.image_height} x {config.forecast_length} "
            f"(H x W; denoised future canvas)"
        )

    
    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler = self.scheduler.to(device)
        return self
    
    def set_guidance_model(self, guidance_model: Optional[Union[GuidanceModel, nn.Module]]) -> None:
        """Set or replace the guidance model for hybrid forecasting.
        
        This allows swapping the Stage 1 predictor after model initialization,
        e.g., to plug in a pre-trained iTransformer checkpoint.
        
        Args:
            guidance_model: Stage 1 predictor model. Set to None to disable
                           guidance (requires config.use_guidance_channel=False).
        """
        if guidance_model is None and self.config.use_guidance_channel:
            raise ValueError(
                "Cannot set guidance_model to None when use_guidance_channel=True. "
                "Either provide a guidance model or set config.use_guidance_channel=False."
            )
        self.guidance_model = guidance_model
        if guidance_model is not None:
            logger.info(f"Guidance model set: {type(guidance_model).__name__}")

    def _get_coordinate_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Create a vertical coordinate gradient map."""
        y_coords = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype)
        coord_grid = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        return coord_grid
    
    def _inject_coordinate_channel(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate vertical coordinate channel to input tensor."""
        if not self.config.use_coordinate_channel:
            return x
        batch_size, _, height, width = x.shape
        coord_grid = self._get_coordinate_grid(
            batch_size, height, width, x.device, x.dtype
        )
        return torch.cat([x, coord_grid], dim=1)
    
    def _get_time_features(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create horizontal time-aware coordinate channels."""
        ramp = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ramp = ramp.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        t_idx = torch.arange(width, device=device, dtype=dtype)
        sine = torch.sin(2 * math.pi * t_idx / self.config.seasonal_period)
        sine = sine.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return ramp, sine
    
    def _inject_time_channels(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate horizontal time coordinate channels to input tensor."""
        if not self.config.use_time_ramp and not self.config.use_time_sine:
            return x
        batch_size, _, height, width = x.shape
        ramp, sine = self._get_time_features(
            batch_size, height, width, x.device, x.dtype
        )
        channels_to_add = [x]
        if self.config.use_time_ramp:
            channels_to_add.append(ramp)
        if self.config.use_time_sine:
            channels_to_add.append(sine)
        return torch.cat(channels_to_add, dim=1)
    
    def _get_value_channel(
        self,
        values_norm: torch.Tensor,
        height: int
    ) -> torch.Tensor:
        """Create a 2D channel containing the normalized values broadcast across height."""
        if values_norm.dim() == 3:
            batch_size, num_vars, seq_len = values_norm.shape
            value_channel = values_norm.unsqueeze(2).expand(-1, -1, height, -1)
        else:
            batch_size, seq_len = values_norm.shape
            value_channel = values_norm.unsqueeze(1).unsqueeze(2).expand(-1, -1, height, -1)
        value_channel = value_channel.clamp(-self.config.max_scale, self.config.max_scale)
        value_channel = value_channel / self.config.max_scale
        return value_channel
    
    def _inject_value_channel(
        self,
        x: torch.Tensor,
        values_norm: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate value channel to input tensor."""
        if not self.config.use_value_channel:
            return x
        _, _, height, _ = x.shape
        value_channel = self._get_value_channel(values_norm, height)
        if value_channel.shape[1] > 1:
            value_channel = value_channel[:, 0:1, :, :]
        return torch.cat([x, value_channel], dim=1)
    
    def _generate_guidance_2d(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int
    ) -> torch.Tensor:
        """Generate 2D "ghost image" from Stage 1 guidance model."""
        coarse_norm = self._get_guidance_forecast_norm(past, past_norm, stats, forecast_length)
        return self.encode_to_2d(coarse_norm, scale_for_diffusion=True)
    
    def _inject_guidance_channel(
        self,
        x: torch.Tensor,
        guidance_2d: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate guidance 2D image to input tensor."""
        if not self.config.use_guidance_channel or guidance_2d is None:
            return x
        return torch.cat([x, guidance_2d], dim=1)
    
    def _prepare_visual_conditioning(
        self,
        past_2d: torch.Tensor,
        target_width: int
    ) -> torch.Tensor:
        """Prepare past 2D image for visual concatenation conditioning."""
        _, _, height, past_len = past_2d.shape
        if past_len >= target_width:
            visual_cond = past_2d[:, :, :, -target_width:]
        else:
            visual_cond = F.interpolate(
                past_2d, 
                size=(height, target_width), 
                mode='bilinear', 
                align_corners=False
            )
        return visual_cond
    
    def _get_guidance_forecast_norm(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int,
    ) -> torch.Tensor:
        """run the guidance model and return normalized forecast (B, V, forecast_length).

        separating this from _generate_guidance_2d so we can reuse the raw forecast
        as cross-variate context without calling get_forecast() twice.
        """
        if self.guidance_model is None:
            raise ValueError("guidance model is None but guidance channel requested")
        mean, std = stats
        K = self.config.lookback_overlap
        H = forecast_length - K
        with torch.no_grad():
            coarse = self.guidance_model.get_forecast(past, H)
        coarse_norm = (coarse - mean) / std
        if K > 0:
            coarse_norm = torch.cat([past_norm[..., -K:], coarse_norm], dim=-1)
        return coarse_norm  # (B, V, forecast_length) normalized

    def _get_cross_variate_context(self, past_raw: torch.Tensor) -> Optional[torch.Tensor]:
        """produce (B, V, ctx_dim) encoder_hidden_states for the bottleneck.

        feeds raw past through the frozen iTransformer encoder and projects to context_dim.
        """
        if self.guidance_model is None or not hasattr(self.guidance_model, 'get_encoder_tokens'):
            raise RuntimeError(
                "iTransformerTokenAdapter requires a guidance model with get_encoder_tokens(). "
                "Ensure guidance model is iTransformerGuidance."
            )
        enc_tokens = self.guidance_model.get_encoder_tokens(past_raw)   # (B, V, d_model)
        return self.context_encoder(enc_tokens)                          # (B, V, ctx_dim)

    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-window z-score using past mean/std, or identity when disabled in config."""
        if not getattr(self.config, "per_window_standardize", True):
            mean = past.new_zeros(past.shape[:-1] + (1,))
            std = past.new_ones(past.shape[:-1] + (1,))
            future_norm = future if future is not None else None
            return past, future_norm, (mean, std)
        mean = past.mean(dim=-1, keepdim=True)
        std = past.std(dim=-1, keepdim=True) + 1e-8
        past_norm = (past - mean) / std
        if future is not None:
            future_norm = (future - mean) / std
        else:
            future_norm = None
        return past_norm, future_norm, (mean, std)
    
    def _denormalize(
        self,
        x: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Denormalize using stored statistics."""
        mean, std = stats
        return x * std + mean
    
    def _compute_emd_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute column-wise Wasserstein-1 distance via CDF trick."""
        if self.config.representation_mode == "pdf":
            temperature = self.config.decode_temperature
            prob_pred = F.softmax(pred / temperature, dim=2)
            prob_target = F.softmax(target / temperature, dim=2)
            cdf_pred = prob_pred.cumsum(dim=2)
            cdf_target = prob_target.cumsum(dim=2)
        else:
            cdf_pred = (pred + 1.0) / 2.0
            cdf_target = (target + 1.0) / 2.0
        emd = (cdf_pred - cdf_target).abs().mean()
        return emd

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Filter out legacy decode_smoothing_kernel from old checkpoints."""
        key = prefix + "decode_smoothing_kernel"
        if key in state_dict:
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def encode_to_2d(self, x: torch.Tensor, scale_for_diffusion: bool = True) -> torch.Tensor:
        """Encode 1D time series to blurred 2D representation."""
        image = self.to_2d(x)
        blurred = self.blur(image)
        if scale_for_diffusion:
            if self.config.representation_mode == "pdf":
                scaled = blurred * 30.0
                scaled = scaled * 2.0 - 1.0
            else:
                scaled = blurred.clamp(min=0.0, max=1.0) * 2.0 - 1.0
            return scaled
        return blurred
    
    def decode_from_2d(
        self,
        image: torch.Tensor,
        from_diffusion: bool = True,
        decoder_method: str = "mean",
        beam_width: int = 5,
        jump_penalty_scale: float = 1.0,
        search_radius: int = 10
    ) -> torch.Tensor:
        """Decode 2D representation to 1D time series."""
        batch_size, num_vars, height, seq_len = image.shape
        squeeze_output = (num_vars == 1)
        
        if self.config.representation_mode == "pdf":
            temperature = self.config.decode_temperature if from_diffusion else None
            x = self.to_2d.inverse(
                image,
                cdf_decoder="expectation",
                expectation_sharpen_temp=temperature,
                squeeze_univariate=squeeze_output,
            )
            return x
        else:
            if from_diffusion:
                cdf_map = (image + 1.0) / 2.0
            else:
                cdf_map = image
            cdf_map = torch.clamp(cdf_map, min=0.0, max=1.0)
            if decoder_method in ("mean", "pdf_expectation"):
                temperature = getattr(self.config, "decode_temperature", None) if decoder_method == "pdf_expectation" else None
                cdf_decoder = "expectation" if decoder_method == "pdf_expectation" else "mean"
                x = self.to_2d.inverse(
                    cdf_map,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=temperature,
                    squeeze_univariate=squeeze_output,
                )
                return x
            if num_vars > 1:
                raise NotImplementedError(f"decoder_method='{decoder_method}' not yet supported for multivariate.")
            cdf_map_squeezed = cdf_map.squeeze(1)
            centers = self.to_2d.bin_centers.view(1, -1, 1).to(cdf_map_squeezed.device)
            if decoder_method == "median":
                below_half_mask = cdf_map_squeezed < 0.5
                has_below = below_half_mask.any(dim=1)
                first_below = below_half_mask.float().argmax(dim=1)
                median_idx = (first_below - 1).clamp(min=0)
                median_idx = torch.where(has_below, median_idx, torch.full_like(median_idx, self.config.image_height - 1))
                all_below = (first_below == 0) & has_below
                median_idx = torch.where(all_below, torch.zeros_like(median_idx), median_idx)
                x = torch.gather(centers.expand(cdf_map_squeezed.shape[0], -1, cdf_map_squeezed.shape[2]), 1, median_idx.unsqueeze(1)).squeeze(1)
            elif decoder_method == "mode":
                drop = -torch.diff(cdf_map_squeezed, dim=1, prepend=cdf_map_squeezed[:, :1, :])
                drop = torch.relu(drop)
                peak_idx = drop.argmax(dim=1)
                x = torch.gather(centers.expand(cdf_map_squeezed.shape[0], -1, cdf_map_squeezed.shape[2]), 1, peak_idx.unsqueeze(1)).squeeze(1)
            elif decoder_method == "beam":
                x = beam_search_decoder(
                    cdf_map_squeezed,
                    bin_centers=self.to_2d.bin_centers.to(cdf_map_squeezed.device),
                    beam_width=beam_width,
                    jump_penalty_scale=jump_penalty_scale,
                    search_radius=search_radius
                )
            else:
                raise ValueError(f"Unknown decoder_method '{decoder_method}'")
        return x
    
    def _apply_coarse_dropout(self, image: torch.Tensor) -> torch.Tensor:
        """Randomly zero rectangular regions to encourage continuity learning."""
        if not self.training or self.config.cutout_prob <= 0:
            return image
        if torch.rand(1, device=image.device).item() >= self.config.cutout_prob:
            return image
        b, c, h, w = image.shape
        num_masks = torch.randint(
            self.config.cutout_min_masks,
            self.config.cutout_max_masks + 1,
            (1,),
            device=image.device
        ).item()
        for _ in range(num_masks):
            shape_idx = torch.randint(0, len(self.config.cutout_shapes), (1,), device=image.device).item()
            mask_h, mask_w = self.config.cutout_shapes[shape_idx]
            mask_h = min(mask_h, h)
            mask_w = min(mask_w, w)
            if mask_h <= 0 or mask_w <= 0: continue
            top_max = max(1, h - mask_h + 1)
            left_max = max(1, w - mask_w + 1)
            top = torch.randint(0, top_max, (1,), device=image.device).item()
            left = torch.randint(0, left_max, (1,), device=image.device).item()
            image[:, :, top:top + mask_h, left:left + mask_w] = -1.0
        return image
    
    def _pad_to_window(
        self,
        tensor: torch.Tensor,
        mode: str,
        total_length: int
    ) -> torch.Tensor:
        """Pad tensor to total window length (Lookback + Forecast)."""
        batch, channels, height, length = tensor.shape
        if length >= total_length:
            return tensor[..., :total_length]
        padding_len = total_length - length
        if mode == 'past':
            return F.pad(tensor, (0, padding_len, 0, 0))
        elif mode == 'future':
            return F.pad(tensor, (padding_len, 0, 0, 0))
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (shared factorized path for unet and dit backbones)."""
        return self._forward_factorized(past, future, t)

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
        """Generate future predictions (shared factorized path for unet and dit backbones).

        sampler:
            'ddim' (default) — existing DDIM(+CFG) path, ``num_ddim_steps`` steps
            'ddpm'           — full T-step DDPM(+CFG)
            'dpmpp'          — DPM-Solver++(2M); CFG not supported (uses cond only)
        num_inference_steps overrides num_ddim_steps when set (used by dpmpp/ddim).
        """
        return self._generate_factorized(
            past, use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
            eta=eta, cfg_scale=cfg_scale, verbose=verbose,
            decoder_method=decoder_method, beam_width=beam_width,
            jump_penalty_scale=jump_penalty_scale, search_radius=search_radius,
            sampler=sampler, num_inference_steps=num_inference_steps,
        )
    
    # ====================================================================
    # Factorized U-Net forward/generate — per-variate shared-weight U-Net
    # with cross-variate context at the bottleneck via iTransformerTokenAdapter
    # ====================================================================

    def _forward_factorized(self, past: torch.Tensor, future: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """training forward: each variate's occupancy map denoised independently.

        the U-Net weights are shared across all V variates. cross-variate info
        is injected at the bottleneck via cross-attention on V context tokens
        from iTransformerTokenAdapter (iTransformer enc_out projected + variate identity).
        """
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        past_2d   = self.encode_to_2d(past_norm)    # (B, V, H, W_past)
        future_2d = self.encode_to_2d(future_norm)   # (B, V, H, W_fut)
        past_2d   = self._apply_coarse_dropout(past_2d)

        W_past = past_2d.shape[3]
        W_fut  = future_2d.shape[3]

        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        noisy_future, noise = self.scheduler.add_noise(future_2d, t)  # (B, V, H, W_fut)

        # compute guidance once, reuse for both the 2D ghost image and the context encoder
        guidance_forecast_norm = None
        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        # ctx: (B, V, ctx_dim) or None

        # flatten variates into batch dim for shared-weight U-Net
        # t: same timestep for all V variates of each batch element
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)  # (BV,)

        canvas = noisy_future.reshape(BV, 1, H, W_fut)
        canvas = self._inject_coordinate_channel(canvas)
        canvas = self._inject_time_channels(canvas)

        # visual cond: per-variate past bilinearly resized to match future width
        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        # broadcast context: every one of the BV U-Net forward passes sees ALL V tokens
        ctx_flat = None
        if ctx is not None:
            # (B, V, ctx_dim) → (BV, V, ctx_dim)
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        # --- Apply Classifier-Free Guidance Dropout ---
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

        x0_pred  = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        
        # Clamp x0_pred for numerical stability at high t
        # Matches ddim_step logic: range [-2, 2] provides slack beyond [-1, 1]
        x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
        
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)

        mono_loss = torch.tensor(0.0, device=device)
        if self.config.use_monotonicity_loss and self.config.representation_mode == "cdf":
            cdf_pred  = torch.clamp((x0_pred + 1.0) / 2.0, 0.0, 1.0)
            mono_loss = monotonicity_loss(cdf_pred)

        guidance_loss = torch.tensor(0.0, device=device)
        if guidance_2d is not None and self.config.guidance_penalty_weight > 0:
            guidance_loss = F.mse_loss(x0_pred, guidance_2d)

        loss = (
            noise_loss + 
            self.config.emd_lambda * emd_loss + 
            self.config.monotonicity_weight * mono_loss +
            self.config.guidance_penalty_weight * guidance_loss
        )

        result = {
            'loss': loss, 'noise_loss': noise_loss, 'emd_loss': emd_loss,
            'guidance_loss': guidance_loss,
            'noise_pred': noise_pred, 't': t,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    @torch.no_grad()
    def _generate_factorized(self, past: torch.Tensor, use_ddim: bool = True,
                              num_ddim_steps: int = 50, eta: float = 0.0,
                              cfg_scale: Optional[float] = None, verbose: bool = False,
                              decoder_method: str = "mean",
                              sampler: str = "ddim",
                              num_inference_steps: Optional[int] = None,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """inference: per-variate DDIM/DDPM sampling with cross-variate bottleneck context."""
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        W_past  = past_2d.shape[3]
        W_fut   = self.config.forecast_length

        # per-variate past visual cond, interpolated to future width
        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        cond_flat     = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
        null_cond     = torch.zeros_like(cond_flat) if cfg_scale > 1.0 else None

        # guidance: compute once before loop
        guidance_forecast_norm = None
        guidance_2d = None
        guide_flat  = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)
            guide_flat  = guidance_2d.reshape(BV, 1, H, W_fut)
        null_guide = torch.zeros_like(guide_flat) if (guide_flat is not None and cfg_scale > 1.0) else None

        # cross-variate context tokens — fixed for entire sampling trajectory
        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat      = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None
        null_ctx_flat = torch.zeros_like(ctx_flat) if (ctx_flat is not None and cfg_scale > 1.0) else None

        def _build_canvas(x_noisy, use_null=False):
            c = self._inject_coordinate_channel(x_noisy)
            c = self._inject_time_channels(c)
            if guide_flat is not None:
                c = torch.cat([c, null_guide if use_null else guide_flat], dim=1)
            return c

        def _chunked_model_fn(x_chunk, t_batch_chunk, cond_arg_chunk, ctx_flat_chunk):
            chunk_size = self.config.unet_max_chunk_size
            BV_curr = x_chunk.shape[0]
            if chunk_size > 0 and BV_curr > chunk_size:
                outs = []
                for i in range(0, BV_curr, chunk_size):
                    end = min(i + chunk_size, BV_curr)
                    c_x = x_chunk[i:end]
                    c_t = t_batch_chunk[i:end] if t_batch_chunk.shape[0] == BV_curr else t_batch_chunk
                    c_cond = cond_arg_chunk[i:end] if cond_arg_chunk is not None else None
                    c_ctx = ctx_flat_chunk[i:end] if ctx_flat_chunk is not None else None
                    outs.append(self.noise_predictor(c_x, c_t, c_cond, encoder_hidden_states=c_ctx))
                return torch.cat(outs, dim=0)
            else:
                return self.noise_predictor(x_chunk, t_batch_chunk, cond_arg_chunk, encoder_hidden_states=ctx_flat_chunk)

        def model_fn(x, t_batch, cond_arg):
            if cfg_scale <= 1.0:
                return _chunked_model_fn(_build_canvas(x), t_batch, cond_arg, ctx_flat)
            # CFG: cond vs uncond pass
            out_c = _chunked_model_fn(_build_canvas(x, use_null=False), t_batch, cond_flat, ctx_flat)
            out_u = _chunked_model_fn(_build_canvas(x, use_null=True),  t_batch, null_cond, null_ctx_flat)
            return out_u + cfg_scale * (out_c - out_u)

        noise_shape = (BV, 1, H, W_fut)

        if sampler == "dpmpp":
            steps = num_inference_steps if num_inference_steps is not None else 20
            future_2d_flat = self.scheduler.sample_dpmpp(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                num_steps=steps, device=device, verbose=verbose,
            )
        elif use_ddim:
            steps = num_inference_steps if num_inference_steps is not None else num_ddim_steps
            future_2d_flat = self.scheduler.sample_ddim_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0,
                num_steps=steps, eta=eta, device=device, verbose=verbose,
            )
        else:
            future_2d_flat = self.scheduler.sample_ddpm_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0, device=device, verbose=verbose,
            )

        future_2d  = future_2d_flat.reshape(B, V, H, W_fut)
        future_norm = self.decode_from_2d(future_2d, decoder_method=decoder_method, **kwargs)
        future      = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future      = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            'prediction': future, 'prediction_norm': future_norm,
            'future_2d': future_2d, 'past_2d': past_2d,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training."""
        outputs = self.forward(past, future)
        return outputs['loss']
