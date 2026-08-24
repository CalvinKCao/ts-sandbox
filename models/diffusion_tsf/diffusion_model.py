"""
Complete Diffusion-based Time Series Forecasting Model.

Binary CDF images, FactorizedDiT denoiser, iTransformer guidance channel.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import DiffusionTSFConfig
from .preprocessing import TimeSeriesTo2D
from .diffusion import BinaryDiffusionScheduler
from .ordinal_window_norm import OrdinalLadder, ordinal_decode, ordinal_encode
from .guidance import GuidanceModel, iTransformerTokenAdapter
from .dit import FactorizedDiT
from .pipeline.eval_bench import (
    note as eval_bench_note,
    span as eval_bench_span,
)

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


@dataclass(frozen=True)
class BinaryLossBreakdown:
    regular: torch.Tensor
    x0: torch.Tensor
    zt: torch.Tensor


class BinaryLossFunction:
    """Stateless BCE, min-SNR, and deterministic-anchor loss calculator."""

    def __init__(
        self,
        config: DiffusionTSFConfig,
        scheduler: BinaryDiffusionScheduler,
        weight_tensor: Callable[..., torch.Tensor],
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self._weight_tensor = weight_tensor

    def bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        t_flat: Optional[torch.Tensor] = None,
        weight_source: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
        apply_min_snr: bool,
    ) -> torch.Tensor:
        per_elem = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        per_elem = per_elem * self._weight_tensor(
            target, weight_source=weight_source,
        ).to(dtype=per_elem.dtype)
        if element_mask is not None:
            if element_mask.shape != per_elem.shape:
                raise ValueError(
                    f"element_mask shape {tuple(element_mask.shape)} != loss shape {tuple(per_elem.shape)}"
                )
            per_elem = per_elem * element_mask
            reduction_denom = element_mask.sum().clamp_min(1.0)
        else:
            reduction_denom = torch.tensor(float(per_elem.numel()), device=per_elem.device)
        if not apply_min_snr or t_flat is None or self.config.loss_weighting == "none":
            return per_elem.sum() / reduction_denom
        beta_t = self.scheduler.betas[t_flat].clamp(1e-5, 1.0 - 1e-5)
        snr = ((1.0 - beta_t) ** 2) / (beta_t ** 2)
        weight = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma)) / snr
        view_shape = (-1,) + (1,) * (per_elem.dim() - 1)
        return (per_elem * weight.view(view_shape)).sum() / reduction_denom

    def regular(
        self,
        primary_logits: torch.Tensor,
        secondary_logits: torch.Tensor,
        clean_target: torch.Tensor,
        noise_target: torch.Tensor,
        t_flat: torch.Tensor,
        *,
        element_mask: Optional[torch.Tensor] = None,
    ) -> BinaryLossBreakdown:
        if self.config.prediction_target == "epsilon":
            x0_target, zt_target = noise_target, clean_target
        else:
            x0_target, zt_target = clean_target, noise_target
        x0 = self.bce(
            primary_logits, x0_target, t_flat=t_flat,
            weight_source=clean_target, element_mask=element_mask,
            apply_min_snr=True,
        )
        zt = self.bce(
            secondary_logits, zt_target, t_flat=t_flat,
            weight_source=clean_target, element_mask=element_mask,
            apply_min_snr=True,
        )
        return BinaryLossBreakdown(regular=x0 + zt, x0=x0, zt=zt)

    def combine_anchor(self, regular: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        lam = float(self.config.deterministic_anchor_lambda)
        return lam * regular + (1.0 - lam) * anchor


class BinaryDataPreparer:
    """Pure normalized-series to binary-CDF conversion operations."""

    def __init__(self, config: DiffusionTSFConfig, to_2d: TimeSeriesTo2D) -> None:
        self.config = config
        self.to_2d = to_2d

    def _subsample(self, x: torch.Tensor) -> torch.Tensor:
        stride = max(1, int(getattr(self.config, "representation_time_stride", 1)))
        return x if stride == 1 else x[..., ::stride]

    def encode_binary(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_2d(self._subsample(x))

    def encode_staged(
        self,
        x: torch.Tensor,
        *,
        ordinal_rank_max: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self._subsample(x)
        coarse_h = int(getattr(self.config, "coarse_image_height", self.config.image_height))
        fine_h = int(getattr(self.config, "fine_image_height", self.config.image_height))
        if ordinal_rank_max is not None:
            coarse, fine = self.to_2d.encode_dual_heights_bounded(
                x, coarse_height=coarse_h, fine_height=fine_h, value_min=0.0,
                value_max_per_variate=ordinal_rank_max,
            )
        else:
            coarse, fine = self.to_2d.encode_dual_heights(
                x, coarse_height=coarse_h, fine_height=fine_h,
            )
        return {"coarse": coarse, "fine": fine}

    @staticmethod
    def resize_cdf_height(image: torch.Tensor, target_height: int) -> torch.Tensor:
        if image.shape[2] == target_height:
            return image
        flat = image.reshape(-1, 1, image.shape[2], image.shape[3])
        resized = F.interpolate(flat, size=(target_height, image.shape[3]), mode="bilinear", align_corners=False)
        return resized.reshape(image.shape[0], image.shape[1], target_height, image.shape[3])

    def coarse_cdf_to_height(self, coarse_map: torch.Tensor, target_height: int) -> torch.Tensor:
        if coarse_map.shape[2] == target_height:
            return coarse_map
        values = self.to_2d._decode_occupancy_in_range(
            coarse_map, value_range=self.config.max_scale, cdf_decoder="mean",
        )
        return self.to_2d._encode_values_in_range(
            values, value_range=self.config.max_scale, height=target_height,
        )


class DiffusionStageStrategy:
    name: str
    uses_patch_abs_embedding = False

    def forward(self, model: "DiffusionTSF", past, future, t, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def generate(self, model: "DiffusionTSF", past, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def capture_diagnostics(self, model: "DiffusionTSF", past, future, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class CoarseStageStrategy(DiffusionStageStrategy):
    name = "coarse"

    def forward(self, model, past, future, t, *, include_anchor: bool = True,
                cross_variate_context=None, context_token_variate_ids=None, **_kwargs):
        return model._forward_binary_coarse(
            past, future, t, include_anchor=include_anchor,
            cross_variate_context=cross_variate_context,
            context_token_variate_ids=context_token_variate_ids,
        )

    def generate(self, model, past, **kwargs):
        return model._generate_binary_coarse(past, **kwargs)

    def capture_diagnostics(self, model, past, future, **kwargs):
        return model._diagnostic_capture_coarse(past, future, **kwargs)


class PatchRefineStageStrategy(DiffusionStageStrategy):
    name = "patch_refine"
    uses_patch_abs_embedding = True

    def forward(
        self,
        model,
        past,
        future,
        t,
        *,
        patch_col0=None,
        loss_mode: str = "combined",
        include_anchor: bool = True,
        cross_variate_context=None,
        context_token_variate_ids=None,
        **_kwargs,
    ):
        return model._forward_binary_patch_refine(
            past, future, t, expand_t_per_window=t is not None,
            patch_col0=patch_col0, loss_mode=loss_mode,
            include_anchor=include_anchor,
            cross_variate_context=cross_variate_context,
            context_token_variate_ids=context_token_variate_ids,
        )

    def generate(self, model, past, **kwargs):
        return model._generate_binary_patch_refine(past, **kwargs)

    def capture_diagnostics(self, model, past, future, **kwargs):
        return model._diagnostic_capture_patch_refine(past, future, **kwargs)


def build_diffusion_stage_strategy(stage: str) -> DiffusionStageStrategy:
    strategies = {"coarse": CoarseStageStrategy, "patch_refine": PatchRefineStageStrategy}
    try:
        return strategies[stage]()
    except KeyError as exc:
        raise ValueError(
            f"unsupported diffusion_stage={stage!r}; only coarse and patch_refine remain"
        ) from exc


class DiffusionTSF(nn.Module):
    """Binary diffusion TSF with FactorizedDiT and optional iTransformer guidance."""

    def __init__(
        self,
        config: DiffusionTSFConfig,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        # Persisted in every diffusion state dict: old guided-channel weights are
        # intentionally incompatible with this cross-attention-only backbone.
        self.register_buffer(
            "conditioning_architecture_version", torch.tensor(1, dtype=torch.int8),
        )
        self.stage_strategy = build_diffusion_stage_strategy(config.diffusion_stage)

        needs_guidance_model = not config.disable_cross_attention
        if needs_guidance_model and guidance_model is None:
            raise ValueError(
                "Cross-attention requires a frozen patch-decoder guidance model; none was provided."
            )

        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.data_prep = BinaryDataPreparer(config, self.to_2d)
        self.guidance_model = guidance_model if needs_guidance_model else None

        backbone_in_channels = config.backbone_in_channels
        dit_patch = config.dit_patch_size
        cond_patch = config.dit_cond_patch_size or (8, 8)
        self.noise_predictor = FactorizedDiT(
            in_channels=backbone_in_channels,
            cond_channels=config.visual_cond_channels,
            out_channels=config.dit_out_channels,
            image_height=config.image_height,
            patch_size=dit_patch,
            embed_dim=config.dit_embed_dim,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            dropout=config.dit_dropout,
            context_dim=config.context_embedding_dim,
            gradient_checkpointing=config.use_gradient_checkpointing,
            cond_patch_size=cond_patch,
            use_scale_embedding=False,
            enable_cross_scale_attention=False,
            use_variate_embedding=(
                config.use_variate_embedding
                and config.variate_factorized
                and config.num_variables > 1
            ),
            max_variates=max(config.num_variables, 512),
            cross_variate_context_bias=config.cross_variate_context_bias,
            use_patch_abs_embedding=self.stage_strategy.uses_patch_abs_embedding,
            max_coarse_bins=max(16, int(config.coarse_image_height)),
            max_horizon_steps=max(
                1024,
                int(config.dataset_forecast_length or 0),
                int(config.forecast_length),
            ),
            use_horizon_chunk_embedding=bool(config.horizon_stitch),
        )

        self._ctx_token_variate_ids: Optional[torch.Tensor] = None
        if config.guidance_type == "patch_decoder":
            self.context_encoder = None
        else:
            self.context_encoder = iTransformerTokenAdapter(
                d_model=config.itrans_d_model,
                context_dim=config.context_embedding_dim,
                max_variates=max(config.num_variables, 512),
                dropout=0.1,
            )

        self.binary_scheduler = BinaryDiffusionScheduler(
            num_steps=config.binary_num_steps,
            beta_start=config.binary_beta_start,
            beta_end=config.binary_beta_end,
            schedule_type=config.binary_noise_schedule,
            length_mode=getattr(config, "binary_length_mode", "none"),
            length_g=float(getattr(config, "binary_length_g", 1.0)),
            length_scale=float(getattr(config, "binary_length_scale", 1.0)),
        )
        self.loss_function = BinaryLossFunction(
            config,
            self.binary_scheduler,
            self._binary_bce_weight_tensor,
        )

        self._ordinal_apply_ood_shift: bool = False
        self._ordinal_input_is_ranked: bool = False

        logger.debug("DiffusionTSF initialized:")
        logger.debug(
            "  Variables: %d (%s)",
            config.num_variables,
            "multivariate" if config.num_variables > 1 else "univariate",
        )
        logger.debug(
            "  Lookback: %d, Forecast: %d",
            config.lookback_length,
            config.forecast_length,
        )
        logger.debug(
            "  Image size: %d x %d (H x W)",
            config.image_height,
            config.forecast_length,
        )

    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        if self.binary_scheduler is not None:
            self.binary_scheduler = self.binary_scheduler.to(device)
        return self

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

    def _resample_1d_time_series(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linearly resample the trailing time axis to exactly target_len."""
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if x.shape[-1] == target_len:
            return x
        if x.shape[-1] == 1:
            return x.expand(*x.shape[:-1], target_len)
        flat = x.reshape(-1, 1, x.shape[-1])
        out = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
        return out.reshape(*x.shape[:-1], target_len)

    def _representation_time_stride(self) -> int:
        return max(1, int(getattr(self.config, "representation_time_stride", 1)))

    def _subsample_repr_time(self, x: torch.Tensor) -> torch.Tensor:
        stride = self._representation_time_stride()
        if stride <= 1:
            return x
        return x[..., ::stride]

    def _repr_time_len(self, raw_len: int) -> int:
        stride = self._representation_time_stride()
        if stride <= 1:
            return raw_len
        return (int(raw_len) + stride - 1) // stride

    def _repr_forecast_width(self, raw_horizon_width: Optional[int] = None) -> int:
        raw = int(raw_horizon_width if raw_horizon_width is not None else self.config.forecast_length)
        return self._repr_time_len(raw)

    def _overlap_repr_cols(self) -> int:
        k = int(self.config.lookback_overlap)
        if k <= 0:
            return 0
        stride = self._representation_time_stride()
        return k // stride if stride > 1 else k

    def _raw_dataset_horizon(self) -> int:
        dhz = int(self.config.dataset_forecast_length or 0)
        if dhz > 0:
            return dhz
        return max(1, int(self.config.forecast_length) - int(self.config.lookback_overlap))

    def _upsample_repr_to_raw_horizon(self, x_repr: torch.Tensor) -> torch.Tensor:
        if self._representation_time_stride() <= 1:
            return x_repr
        return self._resample_1d_time_series(x_repr, self._raw_dataset_horizon())

    def _raw_canvas_length(self) -> int:
        return int(self.config.lookback_overlap) + self._raw_dataset_horizon()

    def _strip_overlap_and_upsample_repr(self, future_norm: torch.Tensor) -> torch.Tensor:
        """Window-norm decode path: drop overlap prefix then upsample forecast to raw hz."""
        k = self._overlap_repr_cols()
        if k > 0:
            future_norm = future_norm[..., k:]
        return self._upsample_repr_to_raw_horizon(future_norm)

    def _flatten_ctx_for_factorized_dit(
        self,
        ctx: Optional[torch.Tensor],
        B: int,
        V: int,
    ) -> Optional[torch.Tensor]:
        if ctx is None:
            return None
        return ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(B * V, ctx.shape[1], -1)

    def _get_cross_variate_context(
        self,
        past_raw: torch.Tensor,
        past_norm: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Produce (B, M, ctx_dim) context tokens for DiT bottleneck cross-attention."""
        if self.guidance_model is None or not hasattr(self.guidance_model, "get_encoder_tokens"):
            raise RuntimeError(
                "Cross-attention requires a guidance model with get_encoder_tokens()."
            )
        self._ctx_token_variate_ids = None
        if getattr(self.config, "guidance_type", "itransformer") == "patch_decoder":
            if past_norm is None:
                past_norm, _, _ = self._normalize_sequence(past_raw, None)
            with eval_bench_span("token_retrieval"):
                enc_tokens = self.guidance_model.get_encoder_tokens(past_norm)
            self._ctx_token_variate_ids = getattr(
                self.guidance_model, "token_variate_ids", None
            )
            return enc_tokens
        with eval_bench_span("token_retrieval"):
            enc_tokens = self.guidance_model.get_encoder_tokens(past_raw)
        if self.context_encoder is None:
            raise RuntimeError("itransformer guidance requires context_encoder.")
        return self.context_encoder(enc_tokens)

    def _resolve_cross_variate_context(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        cached_context: Optional[torch.Tensor] = None,
        cached_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Use phase-cached frozen tokens when provided, otherwise encode live."""
        if self.config.disable_cross_attention:
            return None
        if cached_context is not None:
            self._ctx_token_variate_ids = cached_token_variate_ids
            return cached_context
        return self._get_cross_variate_context(past, past_norm)

    def _window_norm_center(self, past: torch.Tensor) -> torch.Tensor:
        if self.config.window_norm_center == "last":
            return past[..., -1:]
        if self.config.window_norm_center != "mean":
            raise ValueError(
                f"unknown window_norm_center {self.config.window_norm_center!r}"
            )
        return past.mean(dim=-1, keepdim=True)

    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        *,
        apply_ood_shift: Optional[bool] = None,
        data_is_ranked: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Optional[OrdinalLadder]]]:
        """Prepares raw input sequences for diffusion modeling by applying normalization.

        This method is the first step in every forward pass and sample generation loop.
        It supports three distinct normalization pathways:
          1. Ordinal Window Normalization: Converts raw continuous values into discrete
             unit-interval ordinal ranks using a pre-computed OrdinalLadder.
          2. No Normalization: Passes raw inputs through without modification.
          3. Standard Window Normalization: Performs per-window z-score normalization
             ((x - center) / std) with low-variance fallback protection.

        Args:
            past: Lookback window tensor of shape (Batch, Variates, Time_lookback).
            future: Optional ground-truth forecast tensor of shape (Batch, Variates, Time_forecast).
            apply_ood_shift: Override flag for applying out-of-distribution shift correction.
            data_is_ranked: Override flag indicating if incoming tensors are pre-ranked.

        Returns:
            A tuple of (past_norm, future_norm, stats):
              - past_norm: Normalized lookback tensor.
              - future_norm: Normalized forecast tensor (or None if future was None).
              - stats: Tuple containing (center, std, ladder, [ood_shift]) used for denormalization.
        """
        # STEP 1: Fall back to model instance attributes if per-call flags were not provided.
        if apply_ood_shift is None:
            apply_ood_shift = self._ordinal_apply_ood_shift
        if data_is_ranked is None:
            data_is_ranked = self._ordinal_input_is_ranked

        # =========================================================================
        # PATHWAY 1: ORDINAL WINDOW NORMALIZATION
        # =========================================================================
        # Quantizes numerical values into discrete ordinal CDF bins [0, 1].
        if self.config.use_ordinal_window_norm:
            ladder = self.config.ordinal_ladder
            if ladder is None:
                raise ValueError("ordinal_ladder is required when use_ordinal_window_norm=True")
            
            # CASE A: The dataset loader already pre-converted input values into ordinal unit ranks.
            if data_is_ranked:
                batch_size = past.shape[0] if past.dim() == 3 else 1
                ladder_b = ladder.expand_batch(batch_size)
                # Create dummy center=0 and std=1 tensors so tensor shapes match standard stats.
                center = torch.zeros_like(past[..., :1])
                std = torch.ones_like(past[..., :1])
                return past, future, (center, std, ladder_b)

            # CASE B: Raw continuous values need to be encoded into ordinal unit ranks now.
            past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(
                past,
                future,
                ladder=ladder,
                apply_ood_shift=apply_ood_shift,
                causal_only=bool(self.config.ordinal_ood_shift_causal_only),
            )
            # Create dummy center=0 and std=1 tensors for ordinal representation.
            center = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past_ord, future_ord, (center, std, ladder_b, ood_shift)

        # =========================================================================
        # PATHWAY 2: NO NORMALIZATION (IDENTITY)
        # =========================================================================
        if not self.config.use_window_normalization:
            mean = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past, future, (mean, std, None)

        # =========================================================================
        # PATHWAY 3: STANDARD WINDOW NORMALIZATION (PER-WINDOW Z-SCORE)
        # =========================================================================
        # Step 3A: Compute the center for each window (either lookback mean or last value).
        center = self._window_norm_center(past)
        
        # Step 3B: Compute standard deviation across time for each window (shape: B, V, 1).
        past_std = past.std(dim=-1, keepdim=True)
        threshold = float(self.config.window_norm_low_var_threshold)

        # Step 3C: Handle low-variance / flat time series windows to avoid division by zero.
        if threshold > 0.0:
            # Floor standard deviation to avoid zero division (e.g., 1e-8).
            std_floor = past_std.clamp_min(self.config.window_norm_std_floor)
            
            # Determine replacement unit std for low-variance windows (either per-variate or scalar).
            per_v = self.config.window_norm_low_var_unit_std_per_variate
            default_unit = float(self.config.window_norm_low_var_unit_std)
            if per_v is not None:
                if len(per_v) != past.shape[1]:
                    raise ValueError(
                        "window_norm_low_var_unit_std_per_variate length "
                        f"{len(per_v)} != num_variables {past.shape[1]}"
                    )
                unit = torch.tensor(
                    per_v, device=past.device, dtype=past.dtype,
                ).view(1, -1, 1).expand_as(past_std)
            else:
                unit = torch.full_like(past_std, default_unit)
                
            # Identify windows with variance below threshold or completely flat.
            low_var = past_std < threshold
            flat = past_std <= self.config.window_norm_std_floor
            
            # If flat/low variance, use replacement unit std; otherwise use clamped past_std.
            std = torch.where(flat | low_var, unit, std_floor)
        else:
            # Standard clamping without low-variance threshold override.
            std = past_std.clamp_min(self.config.window_norm_std_floor)

        # Hybrid flat variates: already in coverage-scaled dataset space — no instance norm.
        skip_mask = getattr(self.config, "skip_window_norm_variate_mask", None)
        if bool(getattr(self.config, "hybrid_flat_dataset_norm", False)) and skip_mask is None:
            raise RuntimeError(
                "hybrid_flat_dataset_norm=True but skip_window_norm_variate_mask is unset; "
                "call load_dataset (or restore mask from metadata) before encode"
            )
        if skip_mask is not None:
            from utils.hybrid_flat_dataset_norm import apply_skip_window_norm_mask

            center, std = apply_skip_window_norm_mask(center, std, skip_mask)

        # Step 3D: Apply z-score normalization: (x - center) / std.
        past_norm = (past - center) / std
        if future is not None:
            future_norm = (future - center) / std
        else:
            future_norm = None

        return past_norm, future_norm, (center, std, None)
    
    def _denormalize(
        self,
        x: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Denormalize using stored statistics."""
        mean, std = stats
        return x * std + mean

    def _denormalize_future(
        self,
        future_norm: torch.Tensor,
        past: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor, Optional[OrdinalLadder], ...],
        *,
        trim_overlap: bool = True,
    ) -> torch.Tensor:
        """Map decoded future back to global z-score (ordinal inverse or window denorm)."""
        mean, std, ladder, *rest = stats
        ood_shift = rest[0] if rest else None
        if ladder is not None:
            _, future_val = ordinal_decode(
                torch.zeros_like(past[..., :1]).expand_as(past),
                future_norm,
                ladder,
                ood_shift=ood_shift,
            )
            future_norm = future_val
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std)

        K_raw = int(self.config.lookback_overlap)
        K_repr = self._overlap_repr_cols()
        center_shift = (
            ladder is None
            and K_raw > 0
            and getattr(self.config, "lookback_overlap_center_shift", False)
            and future_norm.shape[-1] >= max(K_repr, 1)
        )
        if center_shift:
            overlap_repr = future_norm[..., :K_repr]
            if self._representation_time_stride() > 1 and K_repr > 0:
                overlap_norm = self._resample_1d_time_series(overlap_repr, K_raw)
            else:
                overlap_norm = future_norm[..., :K_raw]
            past_tail = past[..., -K_raw:]
            overlap_raw = overlap_norm * std + mean
            shift = (past_tail - overlap_raw).mean(dim=-1, keepdim=True)
            future = future_norm * std + mean + shift
        else:
            future = self._denormalize(future_norm, (mean, std))

        if self._representation_time_stride() > 1:
            future = self._resample_1d_time_series(future, self._raw_canvas_length())
        if trim_overlap and K_raw > 0:
            future = future[..., K_raw:]
        elif trim_overlap and self._representation_time_stride() > 1:
            future = self._upsample_repr_to_raw_horizon(future)
        return future
    
    def _predict_noise_chunked(
        self,
        canvas: torch.Tensor,
        t_flat: torch.Tensor,
        cond_for_unet: Optional[torch.Tensor],
        ctx_flat: Optional[torch.Tensor],
        scale_indices: Optional[torch.Tensor] = None,
        variate_indices: Optional[torch.Tensor] = None,
        token_variate_ids: Optional[torch.Tensor] = None,
        context_window_indices: Optional[torch.Tensor] = None,
        patch_coarse_bin: Optional[torch.Tensor] = None,
        patch_time0: Optional[torch.Tensor] = None,
        horizon_chunk_emb: Optional[torch.Tensor] = None,
        return_cross_attn_weights: bool = False,
    ) -> torch.Tensor:
        """Run the denoiser with the same chunking rule used by training/eval."""
        chunk_size = self.config.unet_max_chunk_size
        n_items = canvas.shape[0]
        eval_bench_note("dit_n_items", n_items)
        eval_bench_note("dit_chunk_size", int(chunk_size) if chunk_size else n_items)
        if chunk_size > 0 and n_items > chunk_size:
            outs = []
            for i in range(0, n_items, chunk_size):
                end = min(i + chunk_size, n_items)
                c_canvas = canvas[i:end]
                c_t = t_flat[i:end] if t_flat.shape[0] == n_items else t_flat
                c_cond = cond_for_unet[i:end] if cond_for_unet is not None else None
                c_ctx = ctx_flat
                c_context_windows = (
                    context_window_indices[i:end]
                    if context_window_indices is not None
                    else None
                )
                if c_ctx is not None and c_context_windows is None:
                    c_ctx = c_ctx[i:end]
                c_scale = scale_indices[i:end] if scale_indices is not None else None
                c_var = variate_indices[i:end] if variate_indices is not None else None
                c_bin = patch_coarse_bin[i:end] if patch_coarse_bin is not None else None
                c_t0 = patch_time0[i:end] if patch_time0 is not None else None
                c_h = horizon_chunk_emb[i:end] if horizon_chunk_emb is not None else None
                kwargs = {
                    "encoder_hidden_states": c_ctx,
                    "token_variate_ids": token_variate_ids,
                    "context_window_indices": c_context_windows,
                    "return_cross_attn_weights": return_cross_attn_weights and i == 0,
                }
                if c_scale is not None:
                    kwargs["scale_indices"] = c_scale
                if c_var is not None:
                    kwargs["variate_indices"] = c_var
                if c_bin is not None:
                    kwargs["patch_coarse_bin"] = c_bin
                if c_t0 is not None:
                    kwargs["patch_time0"] = c_t0
                if c_h is not None:
                    kwargs["horizon_chunk_emb"] = c_h
                outs.append(self.noise_predictor(c_canvas, c_t, c_cond, **kwargs))
            return torch.cat(outs, dim=0)
        kwargs = {
            "encoder_hidden_states": ctx_flat,
            "token_variate_ids": token_variate_ids,
            "context_window_indices": context_window_indices,
            "return_cross_attn_weights": return_cross_attn_weights,
        }
        if scale_indices is not None:
            kwargs["scale_indices"] = scale_indices
        if variate_indices is not None:
            kwargs["variate_indices"] = variate_indices
        if patch_coarse_bin is not None:
            kwargs["patch_coarse_bin"] = patch_coarse_bin
        if patch_time0 is not None:
            kwargs["patch_time0"] = patch_time0
        if horizon_chunk_emb is not None:
            kwargs["horizon_chunk_emb"] = horizon_chunk_emb
        return self.noise_predictor(canvas, t_flat, cond_for_unet, **kwargs)

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

    def encode_to_2d_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 1D series to a hard binary CDF image without blur."""
        return self.data_prep.encode_binary(x)

    def _uses_global_ordinal_encoding(self) -> bool:
        return bool(
            self.config.use_ordinal_window_norm
            and self.config.ordinal_ladder is not None
        )

    def _ordinal_rank_max_tensor(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ladder = self.config.ordinal_ladder
        if ladder is None:
            raise ValueError("ordinal_ladder is required for global ordinal encoding")
        return ladder.rank_max_per_variate().reshape(-1).to(device=device, dtype=dtype)

    def _staged_fine_residual_value_range(self) -> float:
        if self._uses_global_ordinal_encoding():
            coarse_h = int(getattr(self.config, "coarse_image_height", self.config.image_height))
            vmax = float(self._ordinal_rank_max_tensor(self.to_2d.bin_centers.device).max().item())
            span = max(vmax, 0.0)
            return span / float(coarse_h) * 0.5 if span > 0.0 else 0.0
        coarse_h = int(getattr(self.config, "coarse_image_height", self.config.image_height))
        return float(self.config.max_scale) / float(coarse_h)

    def _decode_staged_combined_1d(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        *,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
    ) -> torch.Tensor:
        if self._uses_global_ordinal_encoding():
            vmax = self._ordinal_rank_max_tensor(coarse_map.device, dtype=coarse_map.dtype)
            return self.to_2d.decode_dual_heights_bounded(
                coarse_map,
                fine_map,
                value_min=0.0,
                value_max_per_variate=vmax,
                cdf_decoder=cdf_decoder,
                expectation_sharpen_temp=expectation_sharpen_temp,
                squeeze_univariate=False,
            )
        return self.to_2d.decode_dual(
            coarse_map,
            fine_map,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
            squeeze_univariate=False,
        )

    def _decode_coarse_1d_from_map(
        self,
        coarse_map: torch.Tensor,
        *,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
    ) -> torch.Tensor:
        if self._uses_global_ordinal_encoding():
            vmax = self._ordinal_rank_max_tensor(coarse_map.device, dtype=coarse_map.dtype)
            vals = []
            for vi in range(coarse_map.shape[1]):
                span = float(vmax[vi].item())
                vals.append(
                    self.to_2d._decode_occupancy_bounded(
                        coarse_map[:, vi : vi + 1],
                        value_min=0.0,
                        value_max=span,
                        cdf_decoder=cdf_decoder,
                        expectation_sharpen_temp=expectation_sharpen_temp,
                    )
                )
            out = torch.cat(vals, dim=1)
            if coarse_map.shape[1] == 1:
                return out.squeeze(1)
            return out
        return self.to_2d._decode_occupancy_in_range(
            coarse_map,
            value_range=self.config.max_scale,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )

    def _decode_fine_1d_from_map(
        self,
        fine_map: torch.Tensor,
        *,
        coarse_height: int,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
    ) -> torch.Tensor:
        fine_range = self._staged_fine_residual_value_range()
        if self._uses_global_ordinal_encoding():
            vals = []
            for vi in range(fine_map.shape[1]):
                vals.append(
                    self.to_2d._decode_occupancy_bounded(
                        fine_map[:, vi : vi + 1],
                        value_min=-fine_range,
                        value_max=fine_range,
                        cdf_decoder=cdf_decoder,
                        expectation_sharpen_temp=expectation_sharpen_temp,
                    )
                )
            out = torch.cat(vals, dim=1)
            if fine_map.shape[1] == 1:
                return out.squeeze(1)
            return out
        return self.to_2d._decode_occupancy_in_range(
            fine_map,
            value_range=fine_range,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )

    def _encode_staged_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ordinal_rank_max = None
        if self._uses_global_ordinal_encoding():
            ordinal_rank_max = self._ordinal_rank_max_tensor(x.device, dtype=x.dtype)
        return self.data_prep.encode_staged(x, ordinal_rank_max=ordinal_rank_max)








    def _binary_anchor_canvas_like(self, like: torch.Tensor) -> torch.Tensor:
        """Bernoulli(0.5) stationary anchor: flat 0.5 mean or random bits."""
        if self.config.binary_anchor_input_mode == "random_bits":
            return torch.bernoulli(torch.full_like(like, 0.5))
        return torch.full_like(like, 0.5)

    def _binary_anchor_canvas_shape(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        template = torch.empty(shape, device=device, dtype=dtype)
        return self._binary_anchor_canvas_like(template)






    def _resize_cdf_height(self, image: torch.Tensor, target_height: int) -> torch.Tensor:
        return self.data_prep.resize_cdf_height(image, target_height)

    def _coarse_cdf_to_height(self, coarse_map: torch.Tensor, target_height: int) -> torch.Tensor:
        return self.data_prep.coarse_cdf_to_height(coarse_map, target_height)

    def decode_dual_from_2d(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        from_diffusion: bool = False,
        decoder_method: str = "mean",
        past_seed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode dual-scale CDF maps to normalized 1D values."""
        if from_diffusion:
            coarse_map = (coarse_map + 1.0) / 2.0
            fine_map = (fine_map + 1.0) / 2.0
        cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
        temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
        if self._uses_global_ordinal_encoding():
            return self._decode_staged_combined_1d(
                coarse_map,
                fine_map,
                cdf_decoder=cdf_decoder,
                expectation_sharpen_temp=temperature,
            )
        return self.to_2d.decode_dual(
            coarse_map,
            fine_map,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=temperature,
            squeeze_univariate=(coarse_map.shape[1] == 1),
        )

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

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        patch_col0: Optional[torch.Tensor] = None,
        loss_mode: str = "combined",
        include_anchor: bool = True,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (binary factorized DiT path)."""
        return self.stage_strategy.forward(
            self,
            past,
            future,
            t,
            patch_col0=patch_col0,
            loss_mode=loss_mode,
            include_anchor=include_anchor,
            cross_variate_context=cross_variate_context,
            context_token_variate_ids=context_token_variate_ids,
        )

    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        verbose: bool = False,
        decoder_method: str = "mean",
        beam_width: int = 5,
        jump_penalty_scale: float = 1.0,
        search_radius: int = 10,
        sampler: str = "ddim",
        num_inference_steps: Optional[int] = None,
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
        horizon_chunk_t0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions via binary reverse sampling.

        sampler: 'ddim' (default), 'anchor' / 'deterministic_anchor' for one-shot anchor decode.
        num_inference_steps overrides binary_sample_steps when set.
        """
        steps = num_inference_steps if num_inference_steps is not None else self.config.binary_sample_steps
        gen_common = dict(
            num_steps=steps,
            verbose=verbose,
            decoder_method=decoder_method,
            beam_width=beam_width,
            jump_penalty_scale=jump_penalty_scale,
            search_radius=search_radius,
            sampler=sampler,
            yield_intermediates=yield_intermediates,
            reverse_step_indices=reverse_step_indices,
            snapshot_timesteps=snapshot_timesteps,
            future_coarse_2d=future_coarse_2d,
            future_fine_2d=future_fine_2d,
            horizon_chunk_t0=horizon_chunk_t0,
        )
        return self.stage_strategy.generate(self, past, **gen_common)


    def _cdf_distance_weight_tensor(
        self,
        target: torch.Tensor,
        *,
        weight_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Distance weights matching target map shape (BV,1,H,W) or (B,V,H,W).

        Always derive |r−k| from a clean CDF map (weight_source or target). Never
        from a random flip mask — that would scramble the distance geometry.
        """
        alpha = float(getattr(self.config, "binary_cdf_distance_alpha", 1.0))
        src = target if weight_source is None else weight_source
        return self.to_2d.cdf_distance_weights(src, alpha)

    def _binary_bce_weight_tensor(
        self,
        target: torch.Tensor,
        *,
        weight_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine the supported CDF-distance weights."""
        weights = torch.ones_like(target)
        if self.config.binary_use_boundary_weighted_bce:
            weights = weights * self._cdf_distance_weight_tensor(
                target, weight_source=weight_source,
            ).to(dtype=weights.dtype)
        return weights

    def _binary_plain_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        weight_source: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unweighted BCE for binary CDF images (optional distance weights)."""
        return self.loss_function.bce(
            logits,
            target,
            weight_source=weight_source,
            element_mask=element_mask,
            apply_min_snr=False,
        )

    def _binary_weighted_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        t_flat: Optional[torch.Tensor] = None,
        *,
        weight_source: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BCE with optional CDF-distance + min-SNR timestep weighting."""
        return self.loss_function.bce(
            logits,
            target,
            t_flat=t_flat,
            weight_source=weight_source,
            element_mask=element_mask,
            apply_min_snr=True,
        )

    def _occupancy_channels(self) -> int:
        return int(getattr(self.config, "data_occupancy_channels", 1))

    def _split_binary_heads(self, out_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split DiT binary outputs into x0 and zt heads (C occupancy channels each)."""
        c = self._occupancy_channels()
        if out_flat.shape[1] < 2 * c:
            raise ValueError(
                f"binary head expected >= {2 * c} channels, got {out_flat.shape[1]}"
            )
        return out_flat[:, :c], out_flat[:, c : 2 * c]
    def _x0_logits_from_prediction(
        self,
        primary_logits: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_target == "epsilon":
            return torch.where(xt > 0.5, -primary_logits, primary_logits)
        return primary_logits

    def _flat_variate_indices(self, bv: int, num_variates: int, device: torch.device) -> torch.Tensor:
        if bv % num_variates != 0:
            raise ValueError(f"bv={bv} not divisible by num_variates={num_variates}")
        batch_size = bv // num_variates
        return torch.arange(num_variates, device=device).unsqueeze(0).expand(batch_size, -1).reshape(bv)

    def _past_cond_resize_to_horizon(self) -> bool:
        return bool(getattr(self.config, "past_cond_resize_to_horizon", True))

    def _past_cond_tail_len(self, past_len: int, target_width: int) -> int:
        cap = int(self.config.diffusion_lookback_cap or 0)
        if cap > 0:
            return min(past_len, cap)
        if not self._past_cond_resize_to_horizon():
            return past_len
        return min(past_len, target_width)

    def _resize_past_cond_to_width(self, cond: torch.Tensor, width: int) -> torch.Tensor:
        H = cond.shape[-2]
        cur = cond.shape[-1]
        if cur == width:
            return cond
        if self._past_cond_resize_to_horizon():
            return F.interpolate(
                cond, size=(H, width), mode="bilinear", align_corners=False,
            )
        if cur > width:
            raise ValueError(
                f"past visual cond width {cur} exceeds target {width} "
                "with past_cond_resize_to_horizon=False"
            )
        return F.pad(cond, (0, width - cur))

    def _expand_horizon_cond_to_past_width(
        self,
        horizon_cond: torch.Tensor,
        past_width: int,
    ) -> torch.Tensor:
        """Right-pad horizon scaffold to native lookback width (no temporal squash)."""
        cur = int(horizon_cond.shape[-1])
        if cur == past_width:
            return horizon_cond
        if cur > past_width:
            raise ValueError(
                f"horizon cond width {cur} exceeds native past width {past_width}"
            )
        return F.pad(horizon_cond, (0, past_width - cur))

    def _cat_past_and_horizon_cond(
        self,
        past_cond: torch.Tensor,
        horizon_cond: torch.Tensor,
    ) -> torch.Tensor:
        # resize=true: warp past down onto the horizon grid (classic dual-scale).
        # resize=false: keep native lookback width and expand the horizon scaffold
        # to match — same contract as joint/patch_refine native-past cond.
        if self._past_cond_resize_to_horizon():
            past_cond = self._resize_past_cond_to_width(
                past_cond, horizon_cond.shape[-1],
            )
        else:
            horizon_cond = self._expand_horizon_cond_to_past_width(
                horizon_cond, past_cond.shape[-1],
            )
        return torch.cat((past_cond, horizon_cond), dim=1)

    def _horizon_chunk_inner(self) -> int:
        return int(self.config.horizon_chunk_inner)

    def _horizon_canvas_width(self) -> int:
        return int(self.config.lookback_overlap) + self._horizon_chunk_inner()

    def _dit_module(self) -> FactorizedDiT:
        pred = self.noise_predictor
        return getattr(pred, "_orig_mod", pred)

    def _horizon_chunk_emb_for_rows(
        self,
        t0: torch.Tensor,
        horizon: torch.Tensor,
        n_rows: int,
        row_window_index: Optional[torch.Tensor] = None,
        n_variates: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not self.config.horizon_stitch:
            return None
        dit = self._dit_module()
        window_emb = dit.encode_horizon_chunk(
            t0, horizon, self._horizon_chunk_inner(),
        )
        if row_window_index is not None:
            if row_window_index.numel() != n_rows:
                raise ValueError(
                    f"row_window_index length {row_window_index.numel()} != n_rows {n_rows}"
                )
            return window_emb.index_select(0, row_window_index)
        v = int(n_variates or 0)
        if v <= 0:
            if window_emb.shape[0] != n_rows:
                raise ValueError(
                    f"horizon_chunk_emb batch {window_emb.shape[0]} != n_rows {n_rows}"
                )
            return window_emb
        if window_emb.shape[0] * v != n_rows:
            raise ValueError(
                f"cannot expand horizon_chunk_emb ({window_emb.shape[0]},) over "
                f"V={v} to n_rows={n_rows}"
            )
        return window_emb.unsqueeze(1).expand(-1, v, -1).reshape(n_rows, -1)

    def _slice_horizon_stitch_future(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep original past; slice a fixed canvas from a long future when stitch is on."""
        from .horizon_chunks import chunk_starts, slice_future_canvas

        k = int(self.config.lookback_overlap)
        inner = self._horizon_chunk_inner()
        canvas_w = k + inner
        b = future.shape[0]
        device = future.device
        h = int(self.config.dataset_forecast_length or 0)
        if h <= 0:
            h = max(1, int(self.config.forecast_length) - k)
        if not self.config.horizon_stitch:
            t0_out = torch.zeros(b, device=device, dtype=torch.long)
            h_t = torch.full((b,), h, device=device, dtype=torch.long)
            return past, future, t0_out, h_t
        expected = k + h
        if future.shape[-1] != expected:
            raise ValueError(
                f"horizon_stitch expects future width {expected} "
                f"(overlap {k} + H {h}), got {future.shape[-1]}"
            )
        starts = chunk_starts(h, inner=inner, overlap=k)
        starts_t = torch.tensor(starts, device=device, dtype=torch.long)
        if t0 is None:
            idx = torch.randint(0, len(starts), (b,), device=device)
            t0_out = starts_t[idx]
        else:
            t0_out = t0.to(device=device, dtype=torch.long).reshape(b)
            if not bool(torch.isin(t0_out, starts_t).all()):
                raise ValueError(f"t0 must be in {starts}, got {t0_out.tolist()}")
        future_c = slice_future_canvas(future, t0_out, inner=inner, overlap=k)
        if future_c.shape[-1] != canvas_w:
            raise ValueError(
                f"sliced canvas width {future_c.shape[-1]} != {canvas_w}"
            )
        h_t = torch.full((b,), h, device=device, dtype=torch.long)
        return past, future_c, t0_out, h_t

    def _window_horizon_ids(
        self,
        n_windows: int,
        device: torch.device,
        t0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = int(self.config.dataset_forecast_length or 0)
        if h <= 0:
            h = max(1, int(self.config.forecast_length) - int(self.config.lookback_overlap))
        if t0 is None:
            t0_out = torch.zeros(n_windows, device=device, dtype=torch.long)
        else:
            t0_out = t0.to(device=device, dtype=torch.long).reshape(n_windows)
        h_t = torch.full((n_windows,), h, device=device, dtype=torch.long)
        return t0_out, h_t

    def _append_raw_lookback_cond_channel(
        self,
        cond: torch.Tensor,
        past_raw: torch.Tensor,
        past_tail_len: int,
        target_width: int,
    ) -> torch.Tensor:
        if not self.config.use_raw_lookback_cond_channel:
            return cond
        if past_raw is None:
            raise ValueError("past_raw is required when use_raw_lookback_cond_channel=True")
        B, V = past_raw.shape[:2]
        H = self.config.image_height
        BV = B * V
        past_tail_raw = past_raw[..., -past_tail_len:]
        raw_maps = self._encode_staged_maps(past_tail_raw)
        raw_coarse = self._resize_cdf_height(raw_maps["coarse"], H)
        past_repr_w = raw_maps["coarse"].shape[-1]
        raw_cond = raw_coarse.reshape(BV, 1, H, past_repr_w)
        if self._past_cond_resize_to_horizon():
            raw_cond = F.interpolate(
                raw_cond,
                size=(H, target_width),
                mode="bilinear",
                align_corners=True,
            )
        elif raw_cond.shape[-1] != cond.shape[-1]:
            raw_cond = self._resize_past_cond_to_width(raw_cond, cond.shape[-1])
        return torch.cat([cond, raw_cond], dim=1)

    def _staged_past_condition(
        self,
        past_norm: torch.Tensor,
        target_width: int,
        past_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build GT lookback conditioning maps for staged denoisers.

        Uses only the trailing diffusion_lookback_cap (default 96) timesteps for 2D
        past CDFs; iTrans cross-attn / guidance ghost use the full past separately.
        """
        B, V = past_norm.shape[:2]
        H = self.config.image_height
        BV = B * V
        past_tail_len = self._past_cond_tail_len(past_norm.shape[-1], target_width)
        past_tail_norm = past_norm[..., -past_tail_len:]
        with eval_bench_span("encode_past"):
            past_maps = self._encode_staged_maps(past_tail_norm)
        past_repr_w = past_maps["coarse"].shape[-1]
        cond_maps = [
            self._resize_cdf_height(past_maps["coarse"], H),
            self._resize_cdf_height(past_maps["fine"], H),
        ]
        cond = torch.cat(
            [m.reshape(BV, 1, H, past_repr_w) for m in cond_maps], dim=1,
        )
        if self._past_cond_resize_to_horizon():
            cond = F.interpolate(cond, size=(H, target_width), mode='bilinear', align_corners=False)
        cond = self._append_raw_lookback_cond_channel(
            cond, past_raw, past_tail_len, target_width,
        )
        return cond, past_maps

    def _patch_refine_geometry_knobs(self) -> Tuple[int, int, int, int]:
        return (
            int(self.config.patch_refine_canvas_height),
            int(self.config.patch_refine_patch_height),
            int(self.config.patch_refine_patch_width),
            int(self.config.patch_refine_col_stride),
        )

    def _encode_absolute_future_hir(
        self,
        future_norm: torch.Tensor,
        canvas_height: int,
    ) -> torch.Tensor:
        from .patch_refine import encode_absolute_hir_cdf

        ordinal_max = None
        if self._uses_global_ordinal_encoding():
            ordinal_max = self._ordinal_rank_max_tensor(future_norm.device, dtype=future_norm.dtype)
        return encode_absolute_hir_cdf(
            future_norm,
            canvas_height=canvas_height,
            max_scale=float(self.config.max_scale),
            ordinal_rank_max=ordinal_max,
        )

    def _decode_absolute_future_hir(self, hir_cdf: torch.Tensor) -> torch.Tensor:
        from .patch_refine import decode_absolute_hir_cdf

        ordinal_max = None
        if self._uses_global_ordinal_encoding():
            ordinal_max = self._ordinal_rank_max_tensor(hir_cdf.device, dtype=hir_cdf.dtype)
        return decode_absolute_hir_cdf(
            hir_cdf,
            max_scale=float(self.config.max_scale),
            ordinal_rank_max=ordinal_max,
        )

    def _patch_refine_lookback_cond(
        self,
        past_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full native-width stacked past coarse∥fine, never resized to the crop."""
        from .patch_refine import stack_past_coarse_fine

        past_tail_len = int(past_norm.shape[-1])
        cap = int(self.config.diffusion_lookback_cap or 0)
        if cap > 0:
            past_tail_len = min(past_tail_len, cap)
        past_tail = past_norm[..., -past_tail_len:]
        with eval_bench_span("encode_past"):
            past_maps = self._encode_staged_maps(past_tail)
        cond = stack_past_coarse_fine(past_maps["coarse"], past_maps["fine"])
        return cond, past_maps

    def _prepare_binary_patch_refine(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        expand_t_per_window: bool = False,
        patch_col0: Optional[torch.Tensor] = None,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Build the graph-free tensors shared by regular and anchor losses."""
        from .patch_refine import (
            build_patch_aux_channels_layout,
            expand_ctx_for_layout,
            expand_lookback_cond_for_layout,
            naive_upscale_coarse_cdf,
        )
        from .patch_refine_geometry import (
            PatchLayout,
            coarse_edges_from_cdf,
            extract_patch_batch_layout,
            patch_layout_for_fixed_col0,
            select_patch_locations,
            subsample_unique_seg_layout,
        )

        assert self.binary_scheduler is not None
        past, future, t0, horizon = self._slice_horizon_stitch_future(past, future)
        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        canvas_h, patch_h, patch_w, col_stride = self._patch_refine_geometry_knobs()
        coarse_h = int(self.config.coarse_image_height)
        unique = bool(getattr(self.config, "patch_refine_unique_segments", False))

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        hir_gt = self._encode_absolute_future_hir(future_norm, canvas_h)
        naive = naive_upscale_coarse_cdf(future_maps["coarse"], canvas_h)
        edges = coarse_edges_from_cdf(future_maps["coarse"], canvas_height=canvas_h)
        if unique:
            if patch_col0 is None:
                max_c0 = int(edges.shape[-1]) - patch_w
                patch_col0 = torch.randint(0, max_c0 + 1, (B,), device=device)
            else:
                patch_col0 = patch_col0.to(device=device, dtype=torch.long).view(B)
            layout = patch_layout_for_fixed_col0(
                edges,
                patch_col0,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
                hir_canvas=hir_gt,
            )
            layout = subsample_unique_seg_layout(
                layout,
                float(getattr(self.config, "patch_refine_finetune_patch_fraction", 1.0)),
                unique_segments=True,
                training=bool(self.training),
            )
        else:
            # Non-unique inference-compatible geometry still chooses coverage
            # repairs with the legacy policy. Downstream crop/condition work is
            # tensorized through PatchLayout, and the live canvas128 training
            # path uses the fully vectorized unique-segment branch above.
            locations = select_patch_locations(
                edges,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
                col_stride=col_stride,
            )
            layout = PatchLayout.from_locations(locations, device=device)

        target_patches = extract_patch_batch_layout(
            hir_gt, layout, patch_height=patch_h, patch_width=patch_w,
        )
        target_occupancy = target_patches.sum(dim=-2, keepdim=True)
        target_visible = (target_occupancy > 0) & (target_occupancy < patch_h)
        target_visible_mask = target_visible.expand_as(target_patches).to(target_patches.dtype)
        # This is a data-validity guard, not layout validation: a fully
        # saturated batch would otherwise make the masked BCE a silent zero
        # loss update. Keep it before the denoiser/compiled region.
        if not bool(target_visible.any()):
            raise RuntimeError("patch_refine batch has no visible GT transitions")
        n_patches = target_patches.shape[0]
        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (n_patches,), device=device)
        elif expand_t_per_window:
            if t.numel() != B:
                raise ValueError(
                    "expand_t_per_window requires one timestep per window "
                    f"(got {t.numel()}, B={B})"
                )
            t = t.to(device=device, dtype=torch.long).index_select(0, layout.batch_index)
        elif t.numel() != n_patches:
            raise ValueError(
                f"timestep batch {t.numel()} incompatible with {n_patches} patches "
                "(pass expand_t_per_window=True to broadcast per-window timesteps)"
            )

        xt, zt = self.binary_scheduler.add_noise(target_patches, t)
        lookback_cond, past_maps = self._patch_refine_lookback_cond(past_norm)
        cond = expand_lookback_cond_for_layout(lookback_cond, layout)

        aux, patch_coarse_bin, patch_time0 = build_patch_aux_channels_layout(
            naive,
            edges,
            layout,
            patch_height=patch_h,
            patch_width=patch_w,
            canvas_height=canvas_h,
            coarse_height=coarse_h,
            horizon_width=int(hir_gt.shape[-1]),
        )

        ctx = self._resolve_cross_variate_context(
            past, past_norm, cross_variate_context, context_token_variate_ids,
        )
        context_window_indices = None
        if ctx is not None:
            if self.training and self.config.cfg_dropout > 0.0:
                # Per-crop CFG dropout needs genuinely distinct zero contexts.
                # Preserve the legacy expanded path for this uncommon mode.
                ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
                ctx_for_patches = expand_ctx_for_layout(ctx_flat, layout)
                drop = torch.rand(n_patches, device=device) < self.config.cfg_dropout
                ctx_for_patches = torch.where(
                    drop.view(n_patches, 1, 1),
                    torch.zeros_like(ctx_for_patches),
                    ctx_for_patches,
                )
            else:
                ctx_for_patches = ctx
                context_window_indices = layout.batch_index
        else:
            ctx_for_patches = None

        return {
            "target_patches": target_patches,
            "target_visible": target_visible,
            "target_visible_mask": target_visible_mask,
            "t": t,
            "xt": xt,
            "zt": zt,
            "cond": cond,
            "aux": aux,
            "ctx": ctx_for_patches,
            "context_window_indices": context_window_indices,
            "variate_indices": layout.variate_index,
            "window_index": layout.batch_index,
            "patch_coarse_bin": patch_coarse_bin,
            "patch_time0": patch_time0,
            "horizon_chunk_t0": t0,
            "horizon_chunk_h": horizon,
            "hir_gt": hir_gt,
            "future_maps": future_maps,
            "past_maps": past_maps,
            "n_patches": n_patches,
        }

    def prepare_patch_refine_loss_inputs(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        *,
        patch_col0: Optional[torch.Tensor] = None,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare one patch batch for separate regular/anchor backwards."""
        if self.stage_strategy.name != "patch_refine":
            raise RuntimeError("prepared patch losses are only valid for patch_refine")
        return self._prepare_binary_patch_refine(
            past,
            future,
            patch_col0=patch_col0,
            cross_variate_context=cross_variate_context,
            context_token_variate_ids=context_token_variate_ids,
        )

    def patch_refine_loss_from_prepared(
        self,
        prepared: Dict[str, Any],
        *,
        loss_mode: str = "combined",
        include_anchor: bool = True,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one or both patch losses from a shared prepared patch batch."""
        if loss_mode not in {"combined", "regular", "anchor"}:
            raise ValueError(f"unknown patch-refine loss_mode={loss_mode!r}")

        target_patches = prepared["target_patches"]
        target_visible = prepared["target_visible"]
        target_visible_mask = prepared["target_visible_mask"]
        t = prepared["t"]
        xt = prepared["xt"]
        zt = prepared["zt"]
        cond = prepared["cond"]
        aux = prepared["aux"]
        ctx = prepared["ctx"]
        context_window_indices = prepared["context_window_indices"]
        variate_indices = prepared["variate_indices"]
        patch_coarse_bin = prepared["patch_coarse_bin"]
        patch_time0 = prepared["patch_time0"]
        n_patches = int(prepared["n_patches"])
        device = target_patches.device
        horizon_emb = self._horizon_chunk_emb_for_rows(
            prepared["horizon_chunk_t0"],
            prepared["horizon_chunk_h"],
            n_patches,
            row_window_index=prepared["window_index"],
        )

        regular_loss = torch.tensor(0.0, device=device)
        loss_x0 = torch.tensor(0.0, device=device)
        loss_zt = torch.tensor(0.0, device=device)
        x0_pred = torch.empty_like(target_patches)
        if loss_mode != "anchor":
            canvas = self._inject_coordinate_channel(xt.float())
            canvas = self._inject_time_channels(canvas)
            canvas = torch.cat([canvas, aux], dim=1)
            out = self._predict_noise_chunked(
                canvas,
                t,
                cond,
                ctx,
                context_window_indices=context_window_indices,
                variate_indices=variate_indices,
                token_variate_ids=self._ctx_token_variate_ids,
                patch_coarse_bin=patch_coarse_bin,
                patch_time0=patch_time0,
                horizon_chunk_emb=horizon_emb,
            )
            primary_logits, zt_logits = self._split_binary_heads(out)
            x0_logits = self._x0_logits_from_prediction(primary_logits, xt)
            loss_breakdown = self.loss_function.regular(
                primary_logits,
                zt_logits,
                target_patches,
                zt,
                t,
                element_mask=target_visible_mask,
            )
            loss_x0 = loss_breakdown.x0
            loss_zt = loss_breakdown.zt
            regular_loss = loss_breakdown.regular
            x0_pred = torch.sigmoid(x0_logits)

        anchor_loss = torch.tensor(0.0, device=device)
        combined_loss = regular_loss if loss_mode != "anchor" else anchor_loss
        if (
            include_anchor
            and loss_mode != "regular"
            and self.config.use_deterministic_anchor_loss
        ):
            anchor_t = torch.full(
                (n_patches,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t.dtype,
            )
            neutral = self._binary_anchor_canvas_like(target_patches)
            anchor_canvas = self._inject_coordinate_channel(neutral)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            anchor_canvas = torch.cat([anchor_canvas, aux], dim=1)
            anchor_out = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t,
                cond,
                ctx,
                context_window_indices=context_window_indices,
                variate_indices=variate_indices,
                token_variate_ids=self._ctx_token_variate_ids,
                patch_coarse_bin=patch_coarse_bin,
                patch_time0=patch_time0,
                horizon_chunk_emb=horizon_emb,
            )
            anchor_primary, _ = self._split_binary_heads(anchor_out)
            anchor_x0 = self._x0_logits_from_prediction(anchor_primary, neutral)
            anchor_loss = self.loss_function.bce(
                anchor_x0,
                target_patches,
                weight_source=target_patches,
                element_mask=target_visible_mask,
                apply_min_snr=False,
            )
            combined_loss = (
                self.loss_function.combine_anchor(regular_loss, anchor_loss)
                if loss_mode == "combined"
                else anchor_loss
            )

        return {
            "loss": combined_loss,
            "noise_loss": regular_loss,
            "combined_mse_loss": combined_loss,
            "anchor_loss": anchor_loss,
            "loss_x0": loss_x0,
            "loss_zt": loss_zt,
            "emd_loss": torch.tensor(0.0, device=device),
            "guidance_loss": torch.tensor(0.0, device=device),
            "noise_pred": x0_pred,
            "x0_pred": x0_pred,
            "future_2d": prepared["hir_gt"],
            "future_2d_coarse": prepared["future_maps"]["coarse"],
            "future_2d_fine": prepared["future_maps"]["fine"],
            "past_2d_coarse": prepared["past_maps"]["coarse"],
            "past_2d_fine": prepared["past_maps"]["fine"],
            "t": t,
            "diffusion_stage": "patch_refine",
            "n_patches": torch.tensor(float(n_patches), device=device),
            "patch_visible_column_fraction": target_visible.float().mean(),
        }

    def _forward_binary_patch_refine(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        expand_t_per_window: bool = False,
        patch_col0: Optional[torch.Tensor] = None,
        loss_mode: str = "combined",
        include_anchor: bool = True,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train boundary-centered 32x8 patches on absolute hi-res CDF crops."""
        prepared = self._prepare_binary_patch_refine(
            past,
            future,
            t=t,
            expand_t_per_window=expand_t_per_window,
            patch_col0=patch_col0,
            cross_variate_context=cross_variate_context,
            context_token_variate_ids=context_token_variate_ids,
        )
        return self.patch_refine_loss_from_prepared(
            prepared,
            loss_mode=loss_mode,
            include_anchor=include_anchor,
        )

    @torch.no_grad()
    def _generate_binary_patch_refine(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Refine a hi-res CDF scaffold with boundary patches, then decode."""
        from .patch_refine import (
            build_patch_aux_channels_layout,
            expand_lookback_cond_for_layout,
            naive_upscale_coarse_cdf,
        )
        from .patch_refine_geometry import (
            PatchLayout,
            blend_patch_bins_layout,
            coarse_edges_from_cdf,
            patch_layout_for_fixed_col0,
            primary_stride_col0s,
            select_patch_locations,
        )
        from .patch_refine_segments import (
            coverage_gap_layout,
        )

        assert self.binary_scheduler is not None
        if future_coarse_2d is None:
            raise ValueError("patch_refine generation requires future_coarse_2d from the coarse model")

        with eval_bench_span("patch_refine"):
            B = past.shape[0]
            V = self.config.num_variables
            device = past.device
            canvas_h, patch_h, patch_w, col_stride = self._patch_refine_geometry_knobs()
            coarse_h = int(self.config.coarse_image_height)
            raw_hz_w = int(self.config.forecast_length)
            W_fut = self._repr_forecast_width(raw_hz_w)
            t0, h_t = self._window_horizon_ids(
                B, device, t0=kwargs.get("horizon_chunk_t0"),
            )
            eval_bench_note("refine_B", B)
            eval_bench_note("refine_V", V)
            eval_bench_note("refine_W", W_fut)
            eval_bench_note("unique_segments", int(bool(getattr(self.config, "patch_refine_unique_segments", False))))

            with eval_bench_span("normalize"):
                past_norm, _, stats = self._normalize_sequence(past)
            coarse = future_coarse_2d.to(device)
            if coarse.shape[:2] != (B, V) or coarse.shape[3] != W_fut:
                raise ValueError(
                    "future_coarse_2d must have shape "
                    f"(B={B}, V={V}, Hc, W={W_fut}), got {tuple(coarse.shape)}"
                )

            with eval_bench_span("geometry"):
                naive = naive_upscale_coarse_cdf(coarse, canvas_h)
                edges = coarse_edges_from_cdf(coarse, canvas_height=canvas_h)
            unique = bool(getattr(self.config, "patch_refine_unique_segments", False))
            lookback_cond, past_maps = self._patch_refine_lookback_cond(past_norm)
            ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)

            def _sample_layout(layout: PatchLayout) -> torch.Tensor:
                with eval_bench_span("layout_aux"):
                    cond_l = expand_lookback_cond_for_layout(lookback_cond, layout)
                    aux_l, patch_coarse_bin_l, patch_time0_l = build_patch_aux_channels_layout(
                        naive,
                        edges,
                        layout,
                        patch_height=patch_h,
                        patch_width=patch_w,
                        canvas_height=canvas_h,
                        coarse_height=coarse_h,
                        horizon_width=W_fut,
                    )
                context_window_indices_l = layout.batch_index if ctx is not None else None
                n_l = layout.n_patches
                eval_bench_note("refine_n_patches", n_l)

                def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
                    canvas = self._inject_coordinate_channel(xt)
                    canvas = self._inject_time_channels(canvas)
                    return torch.cat([canvas, aux_l], dim=1)

                def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
                    horizon_emb = self._horizon_chunk_emb_for_rows(
                        t0, h_t, n_l, row_window_index=layout.batch_index,
                    )
                    out = self._predict_noise_chunked(
                        _build_canvas(xt),
                        t_batch,
                        cond_l,
                        ctx,
                        context_window_indices=context_window_indices_l,
                        variate_indices=layout.variate_index,
                        token_variate_ids=self._ctx_token_variate_ids,
                        patch_coarse_bin=patch_coarse_bin_l,
                        patch_time0=patch_time0_l,
                        horizon_chunk_emb=horizon_emb,
                    )
                    primary, zt = self._split_binary_heads(out)
                    x0_logits = self._x0_logits_from_prediction(primary, xt)
                    return x0_logits, zt

                sample_shape = (n_l, 1, patch_h, patch_w)
                if sampler in ("anchor", "deterministic_anchor"):
                    with eval_bench_span("anchor_decode"):
                        t_batch = torch.full(
                            (n_l,),
                            self.config.binary_num_steps - 1,
                            device=device,
                            dtype=torch.long,
                        )
                        neutral = self._binary_anchor_canvas_shape(sample_shape, device=device)
                        x0_logits, _ = _chunked_model_fn(neutral, t_batch)
                        return (torch.sigmoid(x0_logits) > 0.5).float()
                return self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=sample_shape,
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                    sampler=sampler,
                    reverse_step_indices=reverse_step_indices,
                    snapshot_timesteps=snapshot_timesteps,
                )

            if unique:
                col0s = primary_stride_col0s(int(edges.shape[-1]), patch_w, col_stride)
                eval_bench_note("n_stride_col0", len(col0s))
                with eval_bench_span("parallel_col0s"):
                    layouts = [
                        patch_layout_for_fixed_col0(
                            edges,
                            torch.full((B,), col0, device=device, dtype=torch.long),
                            canvas_height=canvas_h,
                            patch_height=patch_h,
                            patch_width=patch_w,
                        )
                        for col0 in col0s
                    ]
                    layout = PatchLayout.cat(layouts)
                    patch_cdf = _sample_layout(layout)
                with eval_bench_span("coverage_gap"):
                    gap_layout = coverage_gap_layout(
                        edges,
                        layout,
                        canvas_height=canvas_h,
                        patch_height=patch_h,
                        patch_width=patch_w,
                    )
                    if gap_layout is not None:
                        gap_pred = _sample_layout(gap_layout)
                        layout = PatchLayout.cat([layout, gap_layout])
                        patch_cdf = torch.cat([patch_cdf, gap_pred], dim=0)
            else:
                with eval_bench_span("parallel_patches"):
                    locations = select_patch_locations(
                        edges,
                        canvas_height=canvas_h,
                        patch_height=patch_h,
                        patch_width=patch_w,
                        col_stride=col_stride,
                    )
                    layout = PatchLayout.from_locations(locations, device=device)
                    patch_cdf = _sample_layout(layout)

            with eval_bench_span("blend_decode"):
                hir_cdf, patch_vote_counts = blend_patch_bins_layout(
                    patch_cdf,
                    layout,
                    edges,
                    canvas_height=canvas_h,
                    patch_height=patch_h,
                    patch_width=patch_w,
                )
                future_norm = self._decode_absolute_future_hir(hir_cdf)
                future_with_overlap = self._denormalize_future(
                    future_norm, past, stats, trim_overlap=False,
                )
                future = future_with_overlap[..., int(self.config.lookback_overlap):]
            with eval_bench_span("layout_to_cpu"):
                locations = layout.to_locations()
            out = {
                "prediction": future,
                "prediction_norm": future_norm,
                "prediction_global_norm": future,
                "prediction_with_overlap": future_with_overlap,
                "future_2d": hir_cdf,
                "future_2d_coarse": coarse,
                "future_2d_fine": hir_cdf,
                "past_2d_coarse": past_maps["coarse"],
                "past_2d_fine": past_maps["fine"],
                # Keep the pre-blend crops for diagnostics which must not average
                # the stride-overlapping patch predictions.
                "patch_cdf_unblended": patch_cdf,
                "patch_locations": locations,
                "patch_vote_counts": patch_vote_counts,
                "diffusion_stage": "patch_refine",
            }
            return out

    def _forward_binary_coarse(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        include_anchor: bool = True,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train the coarse staged denoiser (patch_refine has its own forward)."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        past, future, t0, horizon = self._slice_horizon_stitch_future(past, future)
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        C_occ = self._occupancy_channels()
        horizon_emb = self._horizon_chunk_emb_for_rows(t0, horizon, BV, n_variates=V)

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        target_2d = future_maps["coarse"]
        W_fut = target_2d.shape[3]
        H = target_2d.shape[2]
        target_flat = target_2d.reshape(BV, 1, H, W_fut)

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        xt_flat, zt_flat = self.binary_scheduler.add_noise(target_flat, t_flat)

        ctx = self._resolve_cross_variate_context(
            past, past_norm, cross_variate_context, context_token_variate_ids,
        )
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)

        base_cond_for_unet = cond_for_unet

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)

        # Staged visual conditioning is always GT during training. CFG dropout is
        # restricted to context tokens so the fine stage never sees predicted coarse.
        ctx_anchor = ctx_flat
        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            if ctx_flat is not None:
                ctx_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1),
                    torch.zeros_like(ctx_flat),
                    ctx_flat,
                )

        out_flat = self._predict_noise_chunked(
            canvas, t_flat, cond_for_unet, ctx_flat,
            variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            horizon_chunk_emb=horizon_emb,
        )
        primary_logits, zt_logits = self._split_binary_heads(out_flat)
        x0_logits = self._x0_logits_from_prediction(primary_logits, xt_flat)
        loss_breakdown = self.loss_function.regular(
            primary_logits, zt_logits, target_flat, zt_flat, t_flat,
        )
        loss_x0 = loss_breakdown.x0
        loss_zt = loss_breakdown.zt
        regular_loss = loss_breakdown.regular

        anchor_loss = torch.tensor(0.0, device=device)
        combined_loss = regular_loss
        if include_anchor and self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_flat.dtype,
            )
            neutral_future_flat = self._binary_anchor_canvas_like(target_flat)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
                horizon_chunk_emb=horizon_emb,
            )
            anchor_primary, _ = self._split_binary_heads(anchor_out_flat)
            anchor_x0_logits = self._x0_logits_from_prediction(anchor_primary, neutral_future_flat)
            anchor_loss = self.loss_function.bce(
                anchor_x0_logits,
                target_flat,
                weight_source=target_flat,
                apply_min_snr=False,
            )
            combined_loss = self.loss_function.combine_anchor(regular_loss, anchor_loss)

        x0_pred = torch.sigmoid(x0_logits).reshape(B, V, H, W_fut)
        result = {
            'loss': combined_loss,
            'noise_loss': regular_loss,
            'combined_mse_loss': combined_loss,
            'anchor_loss': anchor_loss,
            'loss_x0': loss_x0,
            'loss_zt': loss_zt,
            'emd_loss': torch.tensor(0.0, device=device),
            'guidance_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_pred,
            'x0_pred': x0_pred,
            'future_2d': target_2d,
            'future_2d_coarse': future_maps["coarse"],
            'future_2d_fine': future_maps["fine"],
            'past_2d_coarse': past_maps["coarse"],
            'past_2d_fine': past_maps["fine"],
            't': t,
            'diffusion_stage': self.stage_strategy.name,
        }
        result['x0_pred_coarse'] = x0_pred
        return result

    @torch.no_grad()
    def diagnostic_capture_staged(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        *,
        capture_cross_attn: bool = True,
    ) -> Dict[str, Any]:
        """Capture strategy-specific conditioning tensors for diagnostics."""
        return self.stage_strategy.capture_diagnostics(
            self, past, future, capture_cross_attn=capture_cross_attn,
        )

    @torch.no_grad()
    def _diagnostic_capture_coarse(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        *,
        capture_cross_attn: bool = True,
    ) -> Dict[str, Any]:
        """One coarse diagnostic forward: conditioning tensors and cross-attention."""
        past, future, t0, horizon = self._slice_horizon_stitch_future(past, future)
        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        BV = B * V
        past_norm, future_norm, norm_stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        target_2d = future_maps["coarse"]
        W_fut = target_2d.shape[3]
        H = target_2d.shape[2]
        target_flat = target_2d.reshape(BV, 1, H, W_fut)

        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)

        t = torch.zeros(B, device=device, dtype=torch.long)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        xt_flat, _ = self.binary_scheduler.add_noise(target_flat, t_flat)
        canvas = self._inject_coordinate_channel(xt_flat)
        canvas = self._inject_time_channels(canvas)

        base_cond = cond_for_unet
        self._predict_noise_chunked(
            canvas, t_flat, base_cond, ctx_flat,
            variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            return_cross_attn_weights=capture_cross_attn,
            horizon_chunk_emb=self._horizon_chunk_emb_for_rows(t0, horizon, BV, n_variates=V),
        )
        cross_attn_weights = getattr(self.noise_predictor, "_diag_cross_attn_weights", None)

        return {
            "past_norm": past_norm,
            "future_norm": future_norm,
            "norm_stats": norm_stats,
            "cond_for_unet": cond_for_unet,
            "past_maps": past_maps,
            "cross_attn_weights": cross_attn_weights,
            "future_maps": future_maps,
        }

    @torch.no_grad()
    def _diagnostic_capture_patch_refine(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        *,
        capture_cross_attn: bool = True,
    ) -> Dict[str, Any]:
        """Expose patch-refine's native stacked lookback condition for plotting."""
        del capture_cross_attn  # Cross-attention is evaluated per selected patch.
        past, future, _, _ = self._slice_horizon_stitch_future(past, future)
        past_norm, future_norm, norm_stats = self._normalize_sequence(past, future)
        cond_for_unet, past_maps = self._patch_refine_lookback_cond(past_norm)
        return {
            "past_norm": past_norm,
            "future_norm": future_norm,
            "norm_stats": norm_stats,
            "cond_for_unet": cond_for_unet,
            "past_maps": past_maps,
            "cross_attn_weights": None,
            "future_maps": self._encode_staged_maps(future_norm),
        }



    @torch.no_grad()
    def _generate_binary_coarse(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate coarse staged output (patch_refine has its own generate)."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"
        with eval_bench_span("coarse"):
            B = past.shape[0]
            V = self.config.num_variables
            H = self.config.image_height
            device = past.device
            BV = B * V
            C_occ = self._occupancy_channels()
            raw_hz_w = int(self.config.forecast_length)
            W_fut = self._repr_forecast_width(raw_hz_w)
            t0, h_t = self._window_horizon_ids(
                B, device, t0=kwargs.get("horizon_chunk_t0"),
            )
            horizon_emb = self._horizon_chunk_emb_for_rows(t0, h_t, BV, n_variates=V)
            eval_bench_note("coarse_B", B)
            eval_bench_note("coarse_V", V)
            eval_bench_note("coarse_W", W_fut)

            with eval_bench_span("normalize"):
                past_norm, _, stats = self._normalize_sequence(past)
            cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)

            ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
            ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
            variate_indices = None
            if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
                variate_indices = self._flat_variate_indices(BV, V, device)

            def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
                canvas = self._inject_coordinate_channel(xt)
                canvas = self._inject_time_channels(canvas)
                return canvas

            def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
                out = self._predict_noise_chunked(
                    _build_canvas(xt), t_batch, cond_for_unet, ctx_flat,
                    variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
                    horizon_chunk_emb=horizon_emb,
                )
                primary, zt = self._split_binary_heads(out)
                x0_logits = self._x0_logits_from_prediction(primary, xt)
                return x0_logits, zt

            intermediates = None
            sample_shape = (BV, C_occ, H, W_fut)
            if sampler in ("anchor", "deterministic_anchor"):
                with eval_bench_span("anchor_decode"):
                    t_batch = torch.full(
                        (BV,),
                        self.config.binary_num_steps - 1,
                        device=device,
                        dtype=torch.long,
                    )
                    neutral_future_flat = self._binary_anchor_canvas_shape(
                        sample_shape, device=device,
                    )
                    x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
                    future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
                    if yield_intermediates:
                        intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
            else:
                sample_kwargs = dict(
                    model_fn=_chunked_model_fn,
                    shape=sample_shape,
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                    sampler=sampler,
                    reverse_step_indices=reverse_step_indices,
                    snapshot_timesteps=snapshot_timesteps,
                )
                if yield_intermediates:
                    future_2d_flat, intermediates = self.binary_scheduler.sample(
                        yield_intermediates=True,
                        **sample_kwargs,
                    )
                else:
                    future_2d_flat = self.binary_scheduler.sample(**sample_kwargs)

            with eval_bench_span("decode"):
                generated_2d = future_2d_flat.reshape(B, V, H, W_fut)
                future_2d_coarse = generated_2d
                cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
                temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
                future_norm = self._decode_coarse_1d_from_map(
                    future_2d_coarse,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=temperature,
                )
                future_with_overlap = self._denormalize_future(
                    future_norm, past, stats, trim_overlap=False,
                )
                future = future_with_overlap[..., int(self.config.lookback_overlap):]

            result = {
                'prediction': future,
                'prediction_norm': future_norm,
                'prediction_global_norm': future,
                # Retain the K lookback-overlap predictions for diagnostic plots.
                # Metrics continue to consume the forecast-only tensors above.
                'prediction_with_overlap': future_with_overlap,
                'future_2d': generated_2d,
                'future_2d_coarse': future_2d_coarse,
                'past_2d_coarse': past_maps["coarse"],
                'past_2d_fine': past_maps["fine"],
                'diffusion_stage': self.stage_strategy.name,
            }
            if intermediates is not None:
                reshaped_intermediates = []
                for (t_idx, i_tensor) in intermediates:
                    reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
                result['intermediates'] = reshaped_intermediates
            return result

    def _forward_binary_factorized(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Binary training: XOR-noise hard CDF images and predict clean x0 + noise mask."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        future_2d = self.encode_to_2d_binary(future_norm)
        W_fut = future_2d.shape[3]

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        future_flat = future_2d.reshape(BV, 1, H, W_fut)
        xt_flat, zt_flat = self.binary_scheduler.add_noise(future_flat, t_flat)

        past_2d = self.encode_to_2d_binary(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)
        if self._past_cond_resize_to_horizon():
            cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
        else:
            cond_for_unet = past_flat
        cond_for_unet = self._apply_coarse_dropout(cond_for_unet)

        base_cond_for_unet = cond_for_unet

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        ctx_anchor = ctx_flat

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)

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


        out_flat = self._predict_noise_chunked(
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
        )

        x0_logits = out_flat[:, 0:1, :, :]
        zt_logits = out_flat[:, 1:2, :, :]
        loss_x0 = self._binary_plain_bce_loss(x0_logits, future_flat)
        loss_zt = self._binary_plain_bce_loss(zt_logits, zt_flat)
        regular_loss = loss_x0 + loss_zt

        anchor_loss = torch.tensor(0.0, device=device)
        combined_mse_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_flat.dtype,
            )
            neutral_future_flat = self._binary_anchor_canvas_like(future_flat)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            anchor_x0_logits = anchor_out_flat[:, 0:1, :, :]
            anchor_loss = self._binary_plain_bce_loss(anchor_x0_logits, future_flat)
            lam = self.config.deterministic_anchor_lambda
            combined_mse_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        loss = combined_mse_loss
        x0_pred = torch.sigmoid(x0_logits).reshape(B, V, H, W_fut)

        return {
            'loss': loss,
            'noise_loss': regular_loss,
            'combined_mse_loss': combined_mse_loss,
            'anchor_loss': anchor_loss,
            'loss_x0': loss_x0,
            'loss_zt': loss_zt,
            'emd_loss': torch.tensor(0.0, device=device),
            'guidance_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_pred,
            'x0_pred': x0_pred,
            't': t,
        }

    @torch.no_grad()
    def _generate_binary_factorized(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Binary reverse sampling from random bits to a clean hard CDF image."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        raw_hz_w = int(self.config.forecast_length)
        W_fut = self._repr_forecast_width(raw_hz_w)

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d_binary(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)
        if self._past_cond_resize_to_horizon():
            cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
        else:
            cond_for_unet = past_flat

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            canvas = _build_canvas(xt)
            out = self._predict_noise_chunked(
                canvas, t_batch, cond_for_unet, ctx_flat, variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            return out[:, 0:1], out[:, 1:2]

        intermediates = None
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = self._binary_anchor_canvas_shape(
                (BV, 1, H, W_fut), device=device,
            )
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            if yield_intermediates:
                future_2d_flat, intermediates = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=(BV, 1, H, W_fut),
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                    yield_intermediates=True,
                )
            else:
                future_2d_flat = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=(BV, 1, H, W_fut),
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                )
        future_2d = future_2d_flat.reshape(B, V, H, W_fut)
        future_norm = self.decode_from_2d(
            future_2d, from_diffusion=False, decoder_method=decoder_method, **kwargs
        )
        future_with_overlap = self._denormalize_future(
            future_norm, past, stats, trim_overlap=False,
        )
        future = future_with_overlap[..., int(self.config.lookback_overlap):]

        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'prediction_global_norm': future,
            # Retain the K lookback-overlap predictions for diagnostic plots.
            # Metrics continue to consume the forecast-only tensors above.
            'prediction_with_overlap': future_with_overlap,
            'future_2d': future_2d,
            'past_2d': past_2d,
        }
        if intermediates is not None:
            # Reshape intermediate lists from (BV, ...) to (B, V, ...)
            reshaped_intermediates = []
            for (t_idx, i_tensor) in intermediates:
                reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
            result['intermediates'] = reshaped_intermediates
            
        return result

    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        *,
        patch_col0: Optional[torch.Tensor] = None,
        loss_mode: str = "combined",
        include_anchor: bool = True,
        cross_variate_context: Optional[torch.Tensor] = None,
        context_token_variate_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training."""
        outputs = self.forward(
            past,
            future,
            patch_col0=patch_col0,
            loss_mode=loss_mode,
            include_anchor=include_anchor,
            cross_variate_context=cross_variate_context,
            context_token_variate_ids=context_token_variate_ids,
        )
        return outputs["loss"]
