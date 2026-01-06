"""
Complete Diffusion-based Time Series Forecasting Model.

This module integrates:
- Preprocessing (normalization, 2D encoding, blur)
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

# Handle imports for both module and script execution
try:
    from .config import DiffusionTSFConfig
    from .preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
    from .unet import ConditionalUNet2D, TimeSeriesContextEncoder
    from .transformer import DiffusionTransformer
    from .diffusion import DiffusionScheduler
    from .guidance import GuidanceModel, LinearRegressionGuidance, create_guidance_model
except ImportError:
    from config import DiffusionTSFConfig
    from preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
    from unet import ConditionalUNet2D, TimeSeriesContextEncoder
    from transformer import DiffusionTransformer
    from diffusion import DiffusionScheduler
    from guidance import GuidanceModel, LinearRegressionGuidance, create_guidance_model

logger = logging.getLogger(__name__)


def beam_search_decoder(
    cdf_map: torch.Tensor,
    bin_centers: torch.Tensor,
    beam_width: int = 5,
    jump_penalty_scale: float = 1.0,
    search_radius: int = 10,
    eps: float = 1e-8
) -> torch.Tensor:
    """Beam search decoder for CDF/occupancy maps with temporal continuity.
    
    Finds a continuous path through the probability map that maximizes
    likelihood while penalizing large vertical jumps between time steps.
    
    Args:
        cdf_map: Occupancy map of shape (batch, height, seq_len), values in [0, 1].
                 High at bottom, low at top.
        bin_centers: Tensor of shape (height,) mapping bin indices to values.
        beam_width: Number of candidate paths to keep at each step.
        jump_penalty_scale: Multiplier for jump penalty (higher = smoother paths).
        search_radius: Max vertical pixels to search from previous position.
        eps: Small constant for numerical stability in log.
        
    Returns:
        Decoded values of shape (batch, seq_len).
    """
    batch_size, height, seq_len = cdf_map.shape
    device = cdf_map.device
    
    # 1. Convert CDF to PDF: drop[y] = cdf[y] - cdf[y+1] (value drops going up)
    # For occupancy: high at bottom, low at top, so cdf[y] - cdf[y+1] > 0 at transition
    # Append zeros at top for shape consistency
    pdf = torch.zeros_like(cdf_map)
    pdf[:, :-1, :] = cdf_map[:, :-1, :] - cdf_map[:, 1:, :]
    pdf = torch.clamp(pdf, min=0.0)
    
    # Normalize PDF per column to get valid probabilities
    pdf_sum = pdf.sum(dim=1, keepdim=True).clamp(min=eps)
    pdf = pdf / pdf_sum
    
    # Log probabilities (with floor to avoid -inf)
    log_pdf = torch.log(pdf.clamp(min=eps))  # (batch, height, seq_len)
    
    results = []
    
    for b in range(batch_size):
        log_pdf_b = log_pdf[b]  # (height, seq_len)
        
        # Initialize: top beam_width starting positions at t=0
        init_scores = log_pdf_b[:, 0]  # (height,)
        topk_scores, topk_indices = init_scores.topk(min(beam_width, height))
        
        # Each beam: (score, [path indices])
        # Store as tensors for efficiency
        beam_scores = topk_scores  # (beam_width,)
        beam_paths = topk_indices.unsqueeze(1)  # (beam_width, 1)
        
        # Step through time
        for t in range(1, seq_len):
            num_beams = beam_scores.shape[0]
            
            # Current ending positions for each beam
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
            representation_mode=config.representation_mode
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma
        )
        self.register_buffer(
            "decode_smoothing_kernel",
            self._build_decode_smoothing_kernel(sigma_x=3.0, sigma_y=1.0)
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
        
        if config.model_type == "transformer":
            self.noise_predictor = DiffusionTransformer(
                image_height=config.image_height,
                patch_height=config.transformer_patch_height,
                patch_width=config.transformer_patch_width,
                embed_dim=config.transformer_embed_dim,
                depth=config.transformer_depth,
                num_heads=config.transformer_num_heads,
                dropout=config.transformer_dropout,
                in_channels=backbone_in_channels,
                out_channels=config.num_variables,  # Output one channel per variable
            )
            # Note: Transformer backbone does not yet support hybrid conditioning
            self.context_encoder = None
        else:
            self.noise_predictor = ConditionalUNet2D(
                in_channels=backbone_in_channels,
                out_channels=config.num_variables,  # Output one channel per variable
                channels=config.unet_channels,
                num_res_blocks=config.num_res_blocks,
                attention_levels=config.attention_levels,
                image_height=config.image_height,
                kernel_size=config.unet_kernel_size,
                use_dilated_middle=config.use_dilated_middle,
                use_hybrid_condition=config.use_hybrid_condition,
                context_dim=config.context_embedding_dim,
                conditioning_mode=config.conditioning_mode,
                visual_cond_channels=config.visual_cond_channels
            )
            
            # Create TimeSeriesContextEncoder for hybrid conditioning
            if config.use_hybrid_condition:
                self.context_encoder = TimeSeriesContextEncoder(
                    input_channels=config.context_input_channels,
                    embedding_dim=config.context_embedding_dim,
                    num_layers=config.context_encoder_layers,
                    num_heads=4,
                    dropout=0.1,
                    max_seq_len=max(config.lookback_length, config.forecast_length) + 256
                )
            else:
                self.context_encoder = None
        
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
        logger.info(f"  Image size: {config.image_height} x W")
        logger.info(f"  Input channels: {config.backbone_in_channels} (data: {config.num_variables}, aux: {config.num_aux_channels}, guidance: {config.guidance_channels})")
        logger.info(f"  Conditioning mode: {config.conditioning_mode}")
        logger.info(f"  Diffusion steps: {config.num_diffusion_steps}")
        logger.info(f"  Coordinate channel: {config.use_coordinate_channel}")
        logger.info(f"  Time ramp channel: {config.use_time_ramp}")
        logger.info(f"  Time sine channel: {config.use_time_sine}")
        logger.info(f"  Value channel: {config.use_value_channel}")
        if config.use_guidance_channel:
            logger.info(f"  Guidance channel: enabled (Stage 1 → 2D ghost image)")
        if config.use_time_sine:
            logger.info(f"  Seasonal period: {config.seasonal_period}")
        if config.model_type == "unet":
            logger.info(f"  U-Net kernel size: {config.unet_kernel_size}")
            if config.use_dilated_middle:
                logger.info(f"  Dilated middle block: enabled (dilations=1,2,4,8)")
            if config.use_hybrid_condition:
                logger.info(f"  Hybrid 1D conditioning: enabled (context_dim={config.context_embedding_dim}, layers={config.context_encoder_layers})")
    
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
        """Create a vertical coordinate gradient map.
        
        The gradient goes from +1.0 at the top (index 0) to -1.0 at the bottom.
        This provides the backbone with explicit vertical position information,
        helping it distinguish between high and low value regions.
        
        Args:
            batch_size: Batch size for the output tensor
            height: Height of the coordinate map
            width: Width of the coordinate map
            device: Device to place the tensor on
            dtype: Data type of the tensor
            
        Returns:
            Tensor of shape (batch_size, 1, height, width) with vertical gradient
        """
        # Create vertical gradient: +1 at top (y=0), -1 at bottom (y=height-1)
        # This matches the convention where top = high values, bottom = low values
        y_coords = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype)
        
        # Expand to (1, 1, height, 1) then broadcast to (batch_size, 1, height, width)
        coord_grid = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        return coord_grid
    
    def _inject_coordinate_channel(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate vertical coordinate channel to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Tensor of shape (batch, channels+1, height, width) with coordinate channel
        """
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
        """Create horizontal time-aware coordinate channels.
        
        Provides the backbone with explicit temporal position information:
        1. Linear Ramp (Progress Bar): -1.0 at start, +1.0 at end
        2. Sine Wave (Clock): sin(2*pi*t/period) for seasonal awareness
        
        Args:
            batch_size: Batch size for the output tensor
            height: Height of the coordinate map
            width: Width of the coordinate map (sequence length)
            device: Device to place the tensor on
            dtype: Data type of the tensor
            
        Returns:
            Tuple of (ramp, sine) tensors, each of shape (batch_size, 1, height, width)
        """
        # Linear ramp: -1.0 at start (col 0), +1.0 at end (col width-1)
        # This is the "progress bar" showing position in the window
        ramp = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ramp = ramp.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        # Sine wave: sin(2*pi*t/period) where t is column index
        # This is the "clock" providing seasonal/periodic awareness
        t_idx = torch.arange(width, device=device, dtype=dtype)
        sine = torch.sin(2 * math.pi * t_idx / self.config.seasonal_period)
        sine = sine.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        return ramp, sine
    
    def _inject_time_channels(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate horizontal time coordinate channels to input tensor.
        
        Conditionally adds up to two channels based on config:
        - Time Ramp (if use_time_ramp): Linear gradient from -1 to +1 across width (progress bar)
        - Time Sine (if use_time_sine): Sinusoidal wave for seasonal awareness (clock)
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Tensor of shape (batch, channels + num_enabled, height, width) with time channels
        """
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
        """Create a 2D channel containing the normalized values broadcast across height.
        
        Each column (timestep) of the output will have the same value repeated
        down the entire height dimension. This provides the backbone with explicit
        access to the raw numerical values at each timestep.
        
        Args:
            values_norm: Normalized 1D values of shape (batch, [num_vars,] seq_len)
            height: Height of the 2D image to match
            
        Returns:
            Value channel of shape (batch, 1, height, seq_len) for univariate,
            or (batch, num_vars, height, seq_len) for multivariate.
        """
        # Handle multivariate: (batch, num_vars, seq_len)
        if values_norm.dim() == 3:
            batch_size, num_vars, seq_len = values_norm.shape
            # Expand: (batch, num_vars, seq_len) -> (batch, num_vars, 1, seq_len) -> (batch, num_vars, height, seq_len)
            value_channel = values_norm.unsqueeze(2).expand(-1, -1, height, -1)
        else:
            # Univariate: (batch, seq_len)
            batch_size, seq_len = values_norm.shape
            # Expand: (batch, seq_len) -> (batch, 1, 1, seq_len) -> (batch, 1, height, seq_len)
            value_channel = values_norm.unsqueeze(1).unsqueeze(2).expand(-1, -1, height, -1)
        
        # Scale to roughly [-1, 1] range for consistency with other channels
        # The values are already normalized (mean 0, std 1), so we clamp to max_scale
        # and then rescale to [-1, 1]
        value_channel = value_channel.clamp(-self.config.max_scale, self.config.max_scale)
        value_channel = value_channel / self.config.max_scale  # Now in [-1, 1]
        
        return value_channel
    
    def _inject_value_channel(
        self,
        x: torch.Tensor,
        values_norm: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate value channel to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            values_norm: Normalized 1D values of shape (batch, [num_vars,] seq_len)
            
        Returns:
            Tensor with value channel appended
        """
        if not self.config.use_value_channel:
            return x
        
        _, _, height, _ = x.shape
        value_channel = self._get_value_channel(values_norm, height)
        
        # For multivariate, we only use the first variable's values for the aux channel
        # to keep channel count consistent (otherwise it would vary with num_vars)
        if value_channel.shape[1] > 1:
            value_channel = value_channel[:, 0:1, :, :]  # Take first var only
        
        return torch.cat([x, value_channel], dim=1)
    
    def _generate_guidance_2d(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int
    ) -> torch.Tensor:
        """Generate 2D "ghost image" from Stage 1 guidance model.
        
        This converts the coarse 1D forecast from the guidance model into
        a 2D image representation that can be concatenated to the U-Net input.
        
        Args:
            past: Original (unnormalized) past sequence (batch, [num_vars,] past_len)
            past_norm: Normalized past sequence (batch, [num_vars,] past_len)
            stats: Tuple of (mean, std) from normalization
            forecast_length: Number of future steps to predict
            
        Returns:
            Guidance 2D image of shape (batch, num_vars, height, forecast_length)
        """
        if self.guidance_model is None:
            raise ValueError("Guidance model is None but guidance channel is requested")
        
        # Get coarse forecast from Stage 1 model (in original scale)
        with torch.no_grad():
            coarse_forecast = self.guidance_model.get_forecast(past, forecast_length)
        
        # Normalize using same stats as past
        mean, std = stats
        coarse_norm = (coarse_forecast - mean) / std
        
        # Convert to 2D representation
        guidance_2d = self.encode_to_2d(coarse_norm, scale_for_diffusion=True)
        
        return guidance_2d
    
    def _inject_guidance_channel(
        self,
        x: torch.Tensor,
        guidance_2d: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate guidance 2D image to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            guidance_2d: Guidance image of shape (batch, num_vars, height, width)
                        or None if guidance is disabled
            
        Returns:
            Tensor with guidance channels appended (if enabled)
        """
        if not self.config.use_guidance_channel or guidance_2d is None:
            return x
        
        return torch.cat([x, guidance_2d], dim=1)
    
    def _prepare_visual_conditioning(
        self,
        past_2d: torch.Tensor,
        target_width: int
    ) -> torch.Tensor:
        """Prepare past 2D image for visual concatenation conditioning.
        
        For visual_concat mode, the past image is cropped or interpolated to match
        the target width (future length) and concatenated directly as conditioning.
        This allows the model to explicitly "see" the past trajectory pixels.
        
        Takes the END of the past context (most relevant for forecasting), similar
        to ConditioningEncoder behavior.
        
        Args:
            past_2d: Past 2D image of shape (batch, num_vars, height, past_len)
            target_width: Target width (typically forecast_length)
            
        Returns:
            Visual conditioning of shape (batch, num_vars, height, target_width)
        """
        _, _, height, past_len = past_2d.shape
        
        if past_len >= target_width:
            # Take the last target_width timesteps (most relevant for forecasting)
            visual_cond = past_2d[:, :, :, -target_width:]
        else:
            # Past is shorter than target - interpolate to match
            visual_cond = F.interpolate(
                past_2d, 
                size=(height, target_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return visual_cond
    
    def _prepare_1d_context(
        self,
        past_norm: torch.Tensor
    ) -> torch.Tensor:
        """Prepare 1D context input for the TimeSeriesContextEncoder.
        
        Creates a tensor with two channels:
        - Channel 0: Normalized time series values
        - Channel 1: Normalized time index (0.0 to 1.0 ramp)
        
        Supports both univariate and multivariate inputs.
        For multivariate, we flatten all variables into a single sequence.
        
        Args:
            past_norm: Normalized past sequence of shape (batch, [num_vars,] past_len)
            
        Returns:
            Context input of shape (batch, seq_len, context_input_channels)
            For univariate: seq_len = past_len
            For multivariate: seq_len = past_len (each variable processed separately)
        """
        # Handle multivariate: shape is (batch, num_vars, seq_len)
        # We use only the first variable for context encoding for now
        # TODO: Consider encoding all variables (concatenate or separate attention)
        if past_norm.dim() == 3:
            # Use first variable for context (or could average across variables)
            past_1d = past_norm[:, 0, :]  # (batch, past_len)
        else:
            past_1d = past_norm  # (batch, past_len)
        
        batch_size, seq_len = past_1d.shape
        device = past_1d.device
        dtype = past_1d.dtype
        
        # Create time index channel: linear ramp from 0.0 to 1.0
        time_idx = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
        time_idx = time_idx.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)
        
        # Stack to create (batch, seq_len, 2)
        context_input = torch.stack([past_1d, time_idx], dim=-1)
        
        return context_input
    
    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize sequences using past statistics.
        
        Supports both univariate and multivariate inputs:
        - Univariate: (batch, seq_len)
        - Multivariate: (batch, num_vars, seq_len)
        
        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            future: Optional future sequence of shape (batch, [num_vars,] future_len)
            
        Returns:
            (past_norm, future_norm, (mean, std))
        """
        # Compute statistics from past only (along the sequence dimension)
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
    
    def _build_decode_smoothing_kernel(
        self,
        sigma_x: float,
        sigma_y: float
    ) -> torch.Tensor:
        """Create anisotropic Gaussian kernel for decode-time smoothing."""
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
    
    def _apply_decode_smoothing(self, prob: torch.Tensor) -> torch.Tensor:
        """Apply horizontal-heavy Gaussian smoothing to probability map.
        
        This is only used at inference to connect vertical streaks along time.
        """
        kernel = self.decode_smoothing_kernel.to(device=prob.device, dtype=prob.dtype)
        pad_y = kernel.shape[2] // 2
        pad_x = kernel.shape[3] // 2
        prob_2d = prob.unsqueeze(1)  # (batch, 1, height, width)
        prob_padded = F.pad(prob_2d, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        smoothed = F.conv2d(prob_padded, kernel)
        return smoothed.squeeze(1)
    
    def _compute_emd_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute column-wise Wasserstein-1 distance via CDF trick.
        
        Supports multivariate: (batch, num_vars, height, seq_len)
        EMD is computed over height dimension for each variable.
        
        For PDF mode:
            Treats 2D maps as logits, converts to probabilities via softmax,
            then to CDFs, then takes L1.
        For CDF mode:
            The 2D maps ARE the CDFs (occupancy), so we take L1 directly.
        """
        # pred and target: (batch, num_vars, height, seq_len)
        
        if self.config.representation_mode == "pdf":
            temperature = self.config.decode_temperature
            # Softmax along height (dim=2)
            prob_pred = F.softmax(pred / temperature, dim=2)
            prob_target = F.softmax(target / temperature, dim=2)
            
            # Cumsum along height (dim=2)
            cdf_pred = prob_pred.cumsum(dim=2)
            cdf_target = prob_target.cumsum(dim=2)
        else:
            # In CDF mode, the image is the occupancy map (the CDF).
            # We bring it from diffusion range [-1, 1] to [0, 1].
            # L1 distance between CDFs is exactly the EMD (Wasserstein-1).
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
        """Backward compatibility for checkpoints without decode_smoothing_kernel."""
        key = prefix + "decode_smoothing_kernel"
        if key not in state_dict:
            # Insert the default kernel so strict loading succeeds.
            state_dict[key] = self.decode_smoothing_kernel
            if key in missing_keys:
                missing_keys.remove(key)
            logger.warning("decode_smoothing_kernel missing in checkpoint; using default anisotropic kernel.")
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
        """Encode 1D time series to blurred 2D representation.
        
        Supports both univariate and multivariate inputs:
        - Univariate: (batch, seq_len) -> (batch, 1, height, seq_len)
        - Multivariate: (batch, num_vars, seq_len) -> (batch, num_vars, height, seq_len)
        
        Args:
            x: Normalized time series of shape (batch, [num_vars,] seq_len)
            scale_for_diffusion: If True, scale output to [-1, 1] range for diffusion
            
        Returns:
            Blurred 2D image of shape (batch, num_vars, height, seq_len)
        """
        image = self.to_2d(x)
        blurred = self.blur(image)
        
        if scale_for_diffusion:
            if self.config.representation_mode == "pdf":
                # Scale from [0, ~0.03] to [-1, 1] for proper diffusion SNR
                # The blur already creates a pseudo-probability, but values are small
                scaled = blurred * 30.0  # Now roughly [0, 1]
                scaled = scaled * 2.0 - 1.0  # Now [-1, 1]
            else:
                # Occupancy map already dense in [0, 1]; just shift to [-1, 1]
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
        """Decode 2D representation to 1D time series.
        
        Uses expectation over the probability distribution at each time step.
        
        Supports both univariate and multivariate:
        - Univariate: (batch, 1, height, seq_len) -> (batch, seq_len)
        - Multivariate: (batch, num_vars, height, seq_len) -> (batch, num_vars, seq_len)
        
        Args:
            image: 2D image of shape (batch, num_vars, height, seq_len)
            from_diffusion: If True, image is in [-1, 1] range from diffusion
            decoder_method: For CDF/occupancy mode, select 'mean' (sum),
                            'median' (first crossing of 0.5),
                            'mode' (peak of PDF via vertical diff), or
                            'beam' (beam search with temporal continuity).
            beam_width: For 'beam' method, number of candidate paths to keep.
            jump_penalty_scale: For 'beam' method, penalty for vertical jumps.
            search_radius: For 'beam' method, max pixels to search from prev pos.
            
        Returns:
            Time series of shape (batch, [num_vars,] seq_len)
        """
        batch_size, num_vars, height, seq_len = image.shape
        squeeze_output = (num_vars == 1)
        
        if self.config.representation_mode == "pdf":
            if from_diffusion:
                # Image is in [-1, 1] range. Higher values = higher probability.
                # We need to convert to proper probabilities.
                # Softmax along height dimension (dim=2)
                temperature = self.config.decode_temperature
                prob = F.softmax(image / temperature, dim=2)
            else:
                prob = F.softmax(image, dim=2)
            
            # Compute expectation: sum_j P(j) * center(j)
            # Shape: (1, 1, height, 1) for broadcasting with (batch, num_vars, height, seq_len)
            centers = self.to_2d.bin_centers.view(1, 1, -1, 1).to(image.device)
            x = (prob * centers).sum(dim=2)  # -> (batch, num_vars, seq_len)
            
            # Squeeze for univariate backwards compatibility
            if squeeze_output:
                x = x.squeeze(1)
        else:
            if from_diffusion:
                cdf_map = (image + 1.0) / 2.0
            else:
                cdf_map = image
            
            cdf_map = torch.clamp(cdf_map, min=0.0, max=1.0)
            
            # Mean: column-sum approach (default) - works for multivariate
            if decoder_method == "mean":
                # Sum along height dimension (dim=2)
                column_sum = cdf_map.sum(dim=2)  # -> (batch, num_vars, seq_len)
                column_sum = torch.clamp(column_sum, 0.0, float(self.config.image_height))
                normalized = column_sum / float(self.config.image_height)
                x = normalized * (2 * self.config.max_scale) - self.config.max_scale
                
                # Squeeze for univariate backwards compatibility
                if squeeze_output:
                    x = x.squeeze(1)
                return x

            # For other decoders, only support univariate for now
            if num_vars > 1:
                raise NotImplementedError(f"decoder_method='{decoder_method}' not yet supported for multivariate. Use 'mean'.")
            
            # Squeeze to (batch, height, seq_len) for legacy decoder methods
            cdf_map_squeezed = cdf_map.squeeze(1)
            
            if not self.training and self.config.decode_smoothing and self.decode_smoothing_kernel is not None:
                cdf_map_squeezed = self._apply_decode_smoothing(cdf_map_squeezed)
            
            centers = self.to_2d.bin_centers.view(1, -1, 1).to(cdf_map_squeezed.device)

            if decoder_method == "median":
                # Occupancy map: column looks like [1,1,1,0.8,0.5,0.2,0,0] from bottom to top.
                # We want the transition point where intensity drops below 0.5.
                # Find the LAST index where value >= 0.5 (i.e., the highest filled bin).
                
                # below_half_mask: True where intensity < 0.5
                below_half_mask = cdf_map_squeezed < 0.5  # (batch, height, seq_len)
                
                # For each column, find if there's any crossing (any value below 0.5)
                has_below = below_half_mask.any(dim=1)  # (batch, seq_len)
                
                # Find first index from bottom where value < 0.5
                # argmax on float gives first True (first index below 0.5)
                first_below = below_half_mask.float().argmax(dim=1)  # (batch, seq_len)
                
                # The median bin is just before the first below-0.5 bin
                # (i.e., the last bin that's >= 0.5)
                median_idx = (first_below - 1).clamp(min=0)
                
                # If no value is below 0.5 (column all >= 0.5), use top bin
                median_idx = torch.where(
                    has_below,
                    median_idx,
                    torch.full_like(median_idx, self.config.image_height - 1)
                )
                
                # If entire column is below 0.5 (first_below == 0), use bottom bin
                all_below = (first_below == 0) & has_below
                median_idx = torch.where(
                    all_below,
                    torch.zeros_like(median_idx),
                    median_idx
                )
                
                x = torch.gather(
                    centers.expand(cdf_map_squeezed.shape[0], -1, cdf_map_squeezed.shape[2]),
                    1,
                    median_idx.unsqueeze(1)
                ).squeeze(1)
            elif decoder_method == "mode":
                # Occupancy map goes from 1.0 (bottom) to 0.0 (top).
                # The "PDF" is the drop in value as we go up: -diff = cdf[y-1] - cdf[y].
                # We want to find where the drop is largest (the transition edge).
                
                # Compute drop: how much value decreases going from y-1 to y
                # drop[y] = cdf[y-1] - cdf[y] (positive where value decreases)
                drop = -torch.diff(
                    cdf_map_squeezed,
                    dim=1,
                    prepend=cdf_map_squeezed[:, :1, :]  # prepend first row so shapes match
                )
                drop = torch.relu(drop)  # only care about decreases, not increases
                
                peak_idx = drop.argmax(dim=1)
                x = torch.gather(
                    centers.expand(cdf_map_squeezed.shape[0], -1, cdf_map_squeezed.shape[2]),
                    1,
                    peak_idx.unsqueeze(1)
                ).squeeze(1)
            elif decoder_method == "beam":
                # Beam search decoder for temporal continuity
                x = beam_search_decoder(
                    cdf_map_squeezed,
                    bin_centers=self.to_2d.bin_centers.to(cdf_map_squeezed.device),
                    beam_width=beam_width,
                    jump_penalty_scale=jump_penalty_scale,
                    search_radius=search_radius
                )
            else:
                raise ValueError(f"Unknown decoder_method '{decoder_method}' (expected 'mean', 'median', 'mode', or 'beam').")
        
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
            if mask_h <= 0 or mask_w <= 0:
                continue
            
            top_max = max(1, h - mask_h + 1)
            left_max = max(1, w - mask_w + 1)
            top = torch.randint(0, top_max, (1,), device=image.device).item()
            left = torch.randint(0, left_max, (1,), device=image.device).item()
            
            image[:, :, top:top + mask_h, left:left + mask_w] = -1.0
        
        return image
    
    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.
        
        Computes the diffusion loss:
        1. Normalize and encode to 2D
        2. Add noise to future image
        3. Predict noise with U-Net conditioned on past
        4. Return loss between true and predicted noise
        
        Supports both univariate and multivariate inputs.
        
        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            future: Future sequence of shape (batch, [num_vars,] future_len)
            t: Optional diffusion timesteps (sampled randomly if None)
            
        Returns:
            Dictionary with 'loss' and intermediate values for debugging
        """
        batch_size = past.shape[0]
        device = past.device
        
        # Normalize using past statistics
        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        
        # Encode to 2D
        past_2d = self.encode_to_2d(past_norm)  # (batch, num_vars, H, past_len)
        future_2d = self.encode_to_2d(future_norm)  # (batch, num_vars, H, future_len)
        
        # Coarse dropout / cutout augmentation on CONDITIONING ONLY
        # Never apply to future_2d - that's the ground truth target!
        past_2d = self._apply_coarse_dropout(past_2d)
        
        logger.debug(f"past_2d shape: {past_2d.shape}, future_2d shape: {future_2d.shape}")
        
        # Prepare 1D context for hybrid conditioning (if enabled)
        encoder_hidden_states = None
        if self.context_encoder is not None:
            context_input = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(context_input)
            
            # Apply CFG dropout to context as well
            if self.training and self.config.cfg_dropout > 0:
                drop_mask = torch.rand(batch_size, device=device) < self.config.cfg_dropout
                if drop_mask.any():
                    null_context = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        drop_mask.view(-1, 1, 1).expand_as(encoder_hidden_states),
                        null_context,
                        encoder_hidden_states
                    )
        
        # Classifier-Free Guidance: randomly drop conditioning during training
        # This teaches the model to work both with and without conditioning
        if self.training and self.config.cfg_dropout > 0:
            # Create a mask for which samples should have conditioning dropped
            drop_mask = torch.rand(batch_size, device=device) < self.config.cfg_dropout
            if drop_mask.any():
                # Replace dropped conditions with zeros (null conditioning)
                null_cond = torch.zeros_like(past_2d)
                past_2d = torch.where(
                    drop_mask.view(-1, 1, 1, 1).expand_as(past_2d),
                    null_cond,
                    past_2d
                )
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        
        # Add noise to future
        noisy_future, noise = self.scheduler.add_noise(future_2d, t)
        
        # Inject vertical coordinate channel for spatial awareness
        noisy_future_with_coords = self._inject_coordinate_channel(noisy_future)
        
        # Inject horizontal time channels for temporal awareness
        noisy_future_full = self._inject_time_channels(noisy_future_with_coords)
        
        # Inject value channels (normalized values broadcast across height)
        noisy_future_full = self._inject_value_channel(noisy_future_full, future_norm)
        
        # Generate and inject guidance channel (Stage 1 coarse forecast as "ghost image")
        guidance_2d = None
        if self.config.use_guidance_channel:
            forecast_length = future.shape[-1]
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, forecast_length)
            noisy_future_full = self._inject_guidance_channel(noisy_future_full, guidance_2d)
        
        # Prepare conditioning based on mode
        if self.config.conditioning_mode == "visual_concat":
            # Visual concat mode: pass raw past image channels (no aux channels) as conditioning
            # The past visual is cropped to future length for direct pixel-level visibility
            target_width = noisy_future_full.shape[3]  # future_len
            cond_for_unet = self._prepare_visual_conditioning(past_2d, target_width)
        else:
            # Vector embedding mode: pass full past with aux channels to ConditioningEncoder
            past_2d_with_coords = self._inject_coordinate_channel(past_2d)
            past_2d_full = self._inject_time_channels(past_2d_with_coords)
            past_2d_full = self._inject_value_channel(past_2d_full, past_norm)
            cond_for_unet = past_2d_full
        
        # Predict noise (with optional encoder_hidden_states for hybrid conditioning)
        noise_pred = self.noise_predictor(
            noisy_future_full, t, cond_for_unet,
            encoder_hidden_states=encoder_hidden_states
        )
        
        # L2 loss on noise
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Estimate clean image x0_hat from predicted noise
        x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        
        # EMD term between estimated x0 and ground truth x0 (future_2d)
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)
        
        loss = noise_loss + self.config.emd_lambda * emd_loss
        
        result = {
            'loss': loss,
            'noise_loss': noise_loss,
            'emd_loss': emd_loss,
            'noise': noise,
            'noise_pred': noise_pred,
            'past_2d': past_2d,
            'future_2d': future_2d,
            'noisy_future': noisy_future,
            't': t
        }
        
        # Include guidance image in output for visualization/debugging
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        
        return result
    
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
        search_radius: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions given past context.
        
        Supports both univariate and multivariate inputs.
        
        Args:
            past: Past sequence of shape (batch, [num_vars,] past_len)
            use_ddim: Whether to use accelerated DDIM sampling
            num_ddim_steps: Number of DDIM steps (ignored if use_ddim=False)
            eta: DDIM stochasticity parameter
            cfg_scale: Classifier-free guidance scale. If None, uses config value.
                       1.0 = no guidance, >1 = stronger conditioning adherence
            verbose: Whether to log progress
            decoder_method: 'mean', 'median', 'mode', or 'beam' (for CDF mode)
            beam_width: For 'beam', number of candidate paths to keep
            jump_penalty_scale: For 'beam', penalty for vertical jumps (higher = smoother)
            search_radius: For 'beam', max pixels to search from previous position
            
        Returns:
            Dictionary with predictions and intermediate values
        """
        batch_size = past.shape[0]
        device = past.device
        
        # Use config cfg_scale if not specified
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale
        
        # Normalize past
        past_norm, _, stats = self._normalize_sequence(past)
        
        # Encode past to 2D
        past_2d = self.encode_to_2d(past_norm)
        
        # Prepare conditioning based on mode
        if self.config.conditioning_mode == "visual_concat":
            # Visual concat mode: crop past to forecast length for direct pixel visibility
            target_width = self.config.forecast_length
            cond_for_unet = self._prepare_visual_conditioning(past_2d, target_width)
            
            # Null conditioning for CFG: zeros with same shape as visual conditioning
            if cfg_scale > 1.0:
                null_cond_for_unet = torch.zeros_like(cond_for_unet)
            else:
                null_cond_for_unet = None
        else:
            # Vector embedding mode: inject aux channels for ConditioningEncoder
            past_2d_with_coords = self._inject_coordinate_channel(past_2d)
            past_2d_full = self._inject_time_channels(past_2d_with_coords)
            past_2d_full = self._inject_value_channel(past_2d_full, past_norm)
            cond_for_unet = past_2d_full
            
            # Null conditioning for CFG: zeros with all aux channels
            if cfg_scale > 1.0:
                null_cond = torch.zeros_like(past_2d)
                null_cond_with_coords = self._inject_coordinate_channel(null_cond)
                null_cond_full = self._inject_time_channels(null_cond_with_coords)
                # For value channel, use zeros (null conditioning means no value info)
                if self.config.use_value_channel:
                    _, _, height, width = null_cond_full.shape
                    zero_value_channel = torch.zeros(batch_size, 1, height, width, device=device, dtype=null_cond_full.dtype)
                    null_cond_full = torch.cat([null_cond_full, zero_value_channel], dim=1)
                null_cond_for_unet = null_cond_full
            else:
                null_cond_for_unet = None
        
        # Prepare 1D context for hybrid conditioning (if enabled)
        encoder_hidden_states = None
        null_encoder_hidden_states = None
        if self.context_encoder is not None:
            context_input = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(context_input)
            
            # Create null context for CFG
            if cfg_scale > 1.0:
                null_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
        
        # Shape for future image (data channels only - aux channels get added by wrapper)
        future_shape = (
            batch_size,
            self.config.num_variables,
            self.config.image_height,
            self.config.forecast_length
        )
        
        # Generate guidance 2D image if enabled (Stage 1 coarse forecast as "ghost image")
        guidance_2d = None
        null_guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(
                past, past_norm, stats, self.config.forecast_length
            )
            # Null guidance for CFG: zeros with same shape
            if cfg_scale > 1.0:
                null_guidance_2d = torch.zeros_like(guidance_2d)
        
        # Create a wrapper that injects coordinate and time channels before calling the backbone
        # This closure captures the encoder_hidden_states and guidance_2d for conditioning
        def model_with_channels(x, t, cond, use_null_context=False, use_null_guidance=False):
            # x is the noisy future (B, num_vars, H, W), inject coords and time
            x_with_coords = self._inject_coordinate_channel(x)
            x_full = self._inject_time_channels(x_with_coords)
            
            # For value channel during generation, use zeros (we don't know future values)
            if self.config.use_value_channel:
                curr_batch_size = x_full.shape[0]
                _, _, height, width = x_full.shape
                zero_values = torch.zeros(curr_batch_size, 1, height, width, device=x_full.device, dtype=x_full.dtype)
                x_full = torch.cat([x_full, zero_values], dim=1)
            
            # Inject guidance channel (Stage 1 ghost image)
            if self.config.use_guidance_channel:
                guide = null_guidance_2d if use_null_guidance else guidance_2d
                x_full = self._inject_guidance_channel(x_full, guide)
            
            # cond is already prepared based on conditioning_mode:
            # - visual_concat: raw past visual channels at target width
            # - vector_embedding: past with aux channels (processed by ConditioningEncoder in U-Net)
            
            # Determine which context to use
            ctx = null_encoder_hidden_states if use_null_context else encoder_hidden_states
            
            return self.noise_predictor(x_full, t, cond, encoder_hidden_states=ctx)
        
        # Create CFG-aware model wrapper
        def model_cfg(x, t, cond, null_cond=None, cfg_scale=1.0):
            """Model wrapper that handles CFG internally."""
            if cfg_scale <= 1.0 or null_cond is None:
                return model_with_channels(x, t, cond, use_null_context=False, use_null_guidance=False)
            
            # Run conditional and unconditional in parallel via batching
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)
            cond_double = torch.cat([cond, null_cond], dim=0)
            
            # For hybrid conditioning, also double the context
            if encoder_hidden_states is not None:
                ctx_double = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
            else:
                ctx_double = None
            
            # Inject channels
            x_with_coords = self._inject_coordinate_channel(x_double)
            x_full = self._inject_time_channels(x_with_coords)
            
            # For value channel during generation, use zeros (we don't know future values)
            if self.config.use_value_channel:
                batch_double = x_full.shape[0]
                _, _, height, width = x_full.shape
                zero_values = torch.zeros(batch_double, 1, height, width, device=x_full.device, dtype=x_full.dtype)
                x_full = torch.cat([x_full, zero_values], dim=1)
            
            # Inject guidance channel for both conditional and unconditional branches
            # For CFG batching: [conditional, unconditional] -> [guidance_2d, null_guidance_2d]
            if self.config.use_guidance_channel:
                guidance_double = torch.cat([guidance_2d, null_guidance_2d], dim=0)
                x_full = self._inject_guidance_channel(x_full, guidance_double)
            
            # Run batched prediction
            noise_pred_double = self.noise_predictor(
                x_full, t_double, cond_double, encoder_hidden_states=ctx_double
            )
            
            # Split and apply CFG
            noise_cond, noise_uncond = noise_pred_double.chunk(2, dim=0)
            return noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        
        # Generate via diffusion with CFG
        if use_ddim:
            future_2d = self.scheduler.sample_ddim_cfg(
                model=lambda x, t, cond: model_cfg(x, t, cond, null_cond_for_unet, cfg_scale),
                shape=future_shape,
                cond=cond_for_unet,
                null_cond=null_cond_for_unet,
                cfg_scale=1.0,  # CFG is handled inside model_cfg
                num_steps=num_ddim_steps,
                eta=eta,
                device=device,
                verbose=verbose
            )
        else:
            future_2d = self.scheduler.sample_ddpm_cfg(
                model=lambda x, t, cond: model_cfg(x, t, cond, null_cond_for_unet, cfg_scale),
                shape=future_shape,
                cond=cond_for_unet,
                null_cond=null_cond_for_unet,
                cfg_scale=1.0,  # CFG is handled inside model_cfg
                device=device,
                verbose=verbose
            )
        
        # Decode to 1D (normalized)
        future_norm = self.decode_from_2d(
            future_2d,
            decoder_method=decoder_method,
            beam_width=beam_width,
            jump_penalty_scale=jump_penalty_scale,
            search_radius=search_radius
        )
        
        # Denormalize
        future = self._denormalize(future_norm, stats)
        
        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'future_2d': future_2d,
            'past_2d': past_2d
        }
        
        # Include guidance image and Stage 1 forecast in output for analysis
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
            # Also decode the guidance to 1D for comparison
            guidance_norm = self.decode_from_2d(guidance_2d, decoder_method=decoder_method)
            result['guidance_1d'] = self._denormalize(guidance_norm, stats)
        
        return result
    
    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Future sequence of shape (batch, future_len)
            
        Returns:
            Scalar loss value
        """
        outputs = self.forward(past, future)
        return outputs['loss']

