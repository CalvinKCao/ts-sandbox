"""
2D U-Net for diffusion.

built for long skinny images:
- Height: fixed (value resolution, usually 64 or 128)
- Width: variable (sequence length)

stuff in here:
- res blocks with groupnorm
- timestep embeddings
- visual conditioning (concat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
import math
import logging
from typing import List, Optional, Tuple

from . import unet_profile as _unet_prof

logger = logging.getLogger(__name__)



class iTransformerTokenAdapter(nn.Module):
    """Projects frozen iTransformer encoder tokens to context_dim for U-Net cross-attention.

    Feeds iTransformer enc_out (before its linear projector) through a projection
    and a learned per-variate identity embedding, producing context tokens for the
    U-Net bottleneck cross-attention.

    A learned per-variate identity embedding is added so the diffusion model
    can distinguish variates within the shared-weight factorized forward passes.
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        max_variates: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, context_dim)
        self.variate_embed = nn.Embedding(max_variates, context_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, enc_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_tokens: (B, V, d_model) — iTransformer encoder output
        Returns:
            (B, V, context_dim)
        """
        B, V, _ = enc_tokens.shape
        x = self.proj(enc_tokens)                                   # (B, V, context_dim)
        ids = torch.arange(V, device=enc_tokens.device)
        x = x + self.variate_embed(ids)                             # broadcast over B
        return self.norm(self.drop(x))                              # (B, V, context_dim)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning on external context.
    
    Computes attention where:
    - Query (Q): From 2D image features (flattened spatially)
    - Key (K) & Value (V): From 1D context sequence
    
    This allows the U-Net to "look up" precise numerical values from the
    context encoder rather than guessing from the 2D image resolution.
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.0
    ):
        """
        Args:
            query_dim: Dimension of query features (U-Net channel dim)
            context_dim: Dimension of context features (from iTransformerTokenAdapter)
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Dropout rate
        """
        super().__init__()
        inner_dim = num_heads * head_dim
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Layer norms
        self.norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        
        # Projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Query features of shape (batch, seq_len_q, query_dim)
            context: Context features of shape (batch, seq_len_ctx, context_dim)
        
        Returns:
            Output features of shape (batch, seq_len_q, query_dim)
        """
        batch_size, seq_len_q, _ = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        context_norm = self.context_norm(context)
        
        # Compute Q, K, V
        q = self.to_q(x_norm)  # (batch, seq_len_q, inner_dim)
        k = self.to_k(context_norm)  # (batch, seq_len_ctx, inner_dim)
        v = self.to_v(context_norm)  # (batch, seq_len_ctx, inner_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        
        # Output projection + residual
        return x + self.to_out(out)


class SpatialTransformerBlock(nn.Module):
    """Spatial Transformer Block with self-attention and cross-attention.
    
    Similar to Stable Diffusion's attention blocks, this applies:
    1. Self-attention on flattened 2D features
    2. Cross-attention with 1D context (from iTransformerTokenAdapter)
    3. Feedforward network
    
    Used in the deeper levels of the U-Net for conditioning.
    """
    
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        num_groups: int = 8,
        dropout: float = 0.0
    ):
        """
        Args:
            channels: Number of channels in the 2D feature map
            context_dim: Dimension of context from iTransformerTokenAdapter
            num_heads: Number of attention heads
            head_dim: Dimension per head
            num_groups: Number of groups for GroupNorm
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input normalization (operates on 2D spatial data)
        self.norm = nn.GroupNorm(num_groups, channels)
        
        # Project channels to a consistent dimension for attention
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(channels)
        
        # Cross-attention with context
        self.cross_attn = CrossAttentionBlock(
            query_dim=channels,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
        # Project back to original channels
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 2D feature map of shape (batch, channels, height, width)
            context: 1D context of shape (batch, seq_len, context_dim), optional
        
        Returns:
            Output feature map of shape (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape
        residual = x
        
        # Normalize and project
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Flatten spatial dimensions: (batch, channels, height, width) -> (batch, height*width, channels)
        x = x.view(batch, channels, height * width).permute(0, 2, 1)
        
        # Self-attention
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Cross-attention with context (if provided)
        if context is not None:
            x = self.cross_attn(x, context)
        
        # Feedforward
        x = x + self.ff(x)
        
        # Reshape back: (batch, height*width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 2, 1).view(batch, channels, height, width)
        
        # Project out and add residual
        x = self.proj_out(x)
        
        return x + residual


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.
    
    From "Attention Is All You Need" and DDPM papers.
    
    Args:
        timesteps: Tensor of shape (batch,) containing timestep indices
        dim: Embedding dimension
        
    Returns:
        Embeddings of shape (batch, dim)
    """
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    
    return emb


class ResidualBlock(nn.Module):
    """Residual block with Group Normalization and time embedding.
    
    Architecture:
        x -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv -> + -> out
        |                                                              |
        +--------------------- (skip connection) ---------------------+
        
    Time embedding is added after first conv via a linear projection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        dropout: float = 0.1,
        kernel_size: Tuple[int, int] = (3, 3)
    ):
        super().__init__()
        
        # Calculate padding for 'same' output size: padding = (kernel_size - 1) // 2
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # First convolution block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second convolution block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        # Skip connection (identity or 1x1 conv if channels change)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            t_emb: Time embedding of shape (batch, time_emb_dim)
            
        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        h = x
        
        # First block
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding (broadcast to spatial dimensions)
        t_emb_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb_proj
        
        # Second block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block: ResBlocks + optional SpatialTransformerBlock + downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        context_dim: int = 128,
        num_groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3)
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim, num_groups, kernel_size=kernel_size)
            )

        self.use_attention = use_attention
        if use_attention:
            self.spatial_transformer = SpatialTransformerBlock(
                channels=out_channels,
                context_dim=context_dim,
                num_heads=4,
                num_groups=num_groups
            )

        downsample_padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=downsample_padding)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        if self.use_attention:
            x = self.spatial_transformer(x, encoder_hidden_states)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block: upsample + concat skip + ResBlocks + optional SpatialTransformerBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        context_dim: int = 128,
        num_groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3)
    ):
        super().__init__()

        upsample_kernel = (kernel_size[0] + 1, kernel_size[1] + 1)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=upsample_kernel, stride=2, padding=1)

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim, num_groups, kernel_size=kernel_size)
            )

        self.use_attention = use_attention
        if use_attention:
            self.spatial_transformer = SpatialTransformerBlock(
                channels=out_channels,
                context_dim=context_dim,
                num_heads=4,
                num_groups=num_groups
            )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        if self.use_attention:
            x = self.spatial_transformer(x, encoder_hidden_states)
        return x


class DilatedConvBlock(nn.Module):
    """Single dilated convolution block with time embedding injection.
    
    Used in the dilated middle block to capture long-range temporal dependencies.
    Dilation is applied only to the width (time) axis.
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        dilation: Tuple[int, int] = (1, 1),
        num_groups: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Padding to maintain dimensions: padding = dilation * (kernel_size - 1) // 2
        padding = (dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2)
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(h)
        h = self.conv(h)
        
        # Add time embedding
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj
        
        h = self.dropout(h)
        return x + h  # Residual connection


class DilatedMiddleBlock(nn.Module):
    """Middle block with dilated convolutions for expanded temporal receptive field.
    
    Uses exponentially increasing dilation factors on the WIDTH (time) axis only:
    - Layer 1: dilation=(1, 1)  - Standard convolution
    - Layer 2: dilation=(1, 2)  - Look at t, t±2
    - Layer 3: dilation=(1, 4)  - Look at t, t±4
    - Layer 4: dilation=(1, 8)  - Look at t, t±8
    
    Combined receptive field spans ~32 time steps, allowing the model to
    capture long-range wave patterns and asymmetric decays.
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        dilation_factors: List[int] = [1, 2, 4, 8],
        context_dim: int = 128
    ):
        super().__init__()
        self.res_in = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
        self.dilated_blocks = nn.ModuleList()
        for d in dilation_factors:
            self.dilated_blocks.append(
                DilatedConvBlock(
                    channels=channels,
                    time_emb_dim=time_emb_dim,
                    kernel_size=kernel_size,
                    dilation=(1, d),
                    num_groups=num_groups
                )
            )
        self.spatial_transformer = SpatialTransformerBlock(
            channels=channels, context_dim=context_dim, num_heads=4, num_groups=num_groups
        )
        self.res_out = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
        rf = sum([(kernel_size[1] - 1) * d for d in dilation_factors]) + 1
        logger.info(f"DilatedMiddleBlock: dilations={dilation_factors}, temporal receptive field={rf} steps")

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.res_in(x, t_emb)
        for dilated_block in self.dilated_blocks:
            x = dilated_block(x, t_emb)
        x = self.spatial_transformer(x, encoder_hidden_states)
        x = self.res_out(x, t_emb)
        return x


class MiddleBlock(nn.Module):
    """Middle block: ResBlock + SpatialTransformerBlock + ResBlock."""

    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        context_dim: int = 128
    ):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
        self.spatial_transformer = SpatialTransformerBlock(
            channels=channels, context_dim=context_dim, num_heads=4, num_groups=num_groups
        )
        self.res2 = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.spatial_transformer(x, encoder_hidden_states)
        x = self.res2(x, t_emb)
        return x



class ConditionalUNet2D(nn.Module):
    """Conditional 2D U-Net for diffusion-based time series forecasting.

    Predicts noise ε given noisy future image, diffusion timestep, past context image
    (concatenated directly as visual channels), and optional per-variate context tokens
    from an iTransformerTokenAdapter.

    Attention at levels listed in attention_levels uses SpatialTransformerBlock
    (self-attention + cross-attention to encoder_hidden_states when provided).
    The middle block always has a SpatialTransformerBlock.

    Note: When use_coordinate_channel is enabled, in_channels includes aux channels.
    The output channels = num_variables (predicted noise for each variable).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        num_res_blocks: int = 2,
        attention_levels: List[int] = [1, 2, 3],
        time_emb_dim: int = 256,
        num_groups: int = 8,
        image_height: int = 32,
        kernel_size: Tuple[int, int] = (3, 3),
        use_dilated_middle: bool = False,
        context_dim: int = 128,
        visual_cond_channels: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels (num_vars + aux_channels)
            out_channels: Number of output channels (num_variables for noise prediction)
            channels: Channel dimensions at each U-Net level
            num_res_blocks: Number of residual blocks per level
            attention_levels: Loop indices (0-indexed into channels[1:]) where a
                              SpatialTransformerBlock is added after the residual stack.
                              The middle block always has one regardless of this list.
            time_emb_dim: Dimension of time embedding
            num_groups: Number of groups for GroupNorm
            image_height: Height of the 2D image representation
            kernel_size: (height, width) kernel; height = value axis, width = time axis.
            use_dilated_middle: If True, use DilatedMiddleBlock (dilations 1,2,4,8 on time axis).
            context_dim: Dimension of context tokens from iTransformerTokenAdapter.
            visual_cond_channels: Past image channels concatenated to the noisy input.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.visual_cond_channels = visual_cond_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Calculate padding for 'same' output size
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Input = noisy_future (in_channels) + past_visual (visual_cond_channels)
        init_conv_in_channels = in_channels + visual_cond_channels
        
        # Initial convolution with configurable kernel size
        self.init_conv = nn.Conv2d(init_conv_in_channels, channels[0], kernel_size=kernel_size, padding=padding)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            self.down_blocks.append(
                DownBlock(
                    in_ch, out_ch, time_emb_dim, num_res_blocks,
                    use_attention=(i in attention_levels),
                    context_dim=context_dim,
                    num_groups=num_groups,
                    kernel_size=kernel_size
                )
            )
            in_ch = out_ch

        # Middle block (always has SpatialTransformerBlock)
        if use_dilated_middle:
            self.middle = DilatedMiddleBlock(
                channels[-1], time_emb_dim, num_groups, kernel_size=kernel_size,
                context_dim=context_dim
            )
        else:
            self.middle = MiddleBlock(
                channels[-1], time_emb_dim, num_groups, kernel_size=kernel_size,
                context_dim=context_dim
            )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            skip_ch = reversed_channels[i]
            self.up_blocks.append(
                UpBlock(
                    in_ch, out_ch, skip_ch, time_emb_dim, num_res_blocks,
                    use_attention=((len(channels) - 2 - i) in attention_levels),
                    context_dim=context_dim,
                    num_groups=num_groups,
                    kernel_size=kernel_size
                )
            )

        self.final_norm = nn.GroupNorm(num_groups, channels[0])
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=kernel_size, padding=padding)

        logger.info(f"ConditionalUNet2D initialized with channels={channels}, kernel_size={kernel_size}")
        logger.info(f"  Visual concat: {visual_cond_channels} past image channels directly concatenated")
        logger.info(f"  attention_levels={attention_levels}, context_dim={context_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy future image of shape (batch, in_channels, height, future_len)
            t: Diffusion timesteps of shape (batch,)
            cond: Past context image of shape (batch, visual_cond_channels, height, future_len),
                  already cropped/interpolated to target width by the caller.
            encoder_hidden_states: Optional per-variate context tokens from iTransformerTokenAdapter,
                                   shape (batch, seq_len, context_dim). Cross-attn skipped when None.
            
        Returns:
            Predicted noise of shape (batch, out_channels, height, future_len)
        """
        with _unet_prof.section("time_embed"):
            t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
            t_emb = self.time_mlp(t_emb)

        with _unet_prof.section("cat_input"):
            x = torch.cat([x, cond], dim=1)

        with _unet_prof.section("init_conv"):
            x = self.init_conv(x)

        skips = []
        for i, down_block in enumerate(self.down_blocks):
            with _unet_prof.section(f"down_{i}"):
                if self.use_gradient_checkpointing and self.training:
                    x, skip = grad_ckpt.checkpoint(
                        down_block, x, t_emb, encoder_hidden_states, use_reentrant=False
                    )
                else:
                    x, skip = down_block(x, t_emb, encoder_hidden_states)
            skips.append(skip)

        with _unet_prof.section("middle"):
            if self.use_gradient_checkpointing and self.training:
                x = grad_ckpt.checkpoint(
                    self.middle, x, t_emb, encoder_hidden_states, use_reentrant=False
                )
            else:
                x = self.middle(x, t_emb, encoder_hidden_states)

        for i, (up_block, skip) in enumerate(zip(self.up_blocks, reversed(skips))):
            with _unet_prof.section(f"up_{i}"):
                if self.use_gradient_checkpointing and self.training:
                    x = grad_ckpt.checkpoint(
                        up_block, x, skip, t_emb, encoder_hidden_states, use_reentrant=False
                    )
                else:
                    x = up_block(x, skip, t_emb, encoder_hidden_states)

        with _unet_prof.section("final_norm_act"):
            x = self.final_norm(x)
            x = F.silu(x)
        with _unet_prof.section("final_conv"):
            x = self.final_conv(x)
        return x

