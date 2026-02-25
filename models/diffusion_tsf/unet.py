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
import math
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """pos encodings for transformers.
    adds sin/cos waves so the model knows where it is.
    """
    
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """adds PE to x."""
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer1D(nn.Module):
    """transformer block for 1D.
    pre-norm setup:
    - multi-head attention
    - mlp with gelu
    """
    
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """standard transformer pass."""
        # attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # feed forward
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSeriesContextEncoder(nn.Module):
    """encoder for the 1D stuff.
    
    processes raw past data (values + time) for cross-attention.
    
    Input: (batch, seq_len, 2)
        - val 0: normalized values
        - val 1: time indices
    
    Output: (batch, seq_len, dim)
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        """
        Args:
            input_channels: Number of input channels (value + time index = 2)
            embedding_dim: Dimension of output embeddings
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Project input channels to embedding dimension
        self.input_proj = nn.Linear(input_channels, embedding_dim)
        
        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer1D(embedding_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        logger.info(f"TimeSeriesContextEncoder: input_ch={input_channels}, "
                    f"embed_dim={embedding_dim}, layers={num_layers}, heads={num_heads}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_channels)
               - x[:, :, 0] = normalized values
               - x[:, :, 1] = normalized time indices (0 to 1)
        
        Returns:
            Context embeddings of shape (batch, seq_len, embedding_dim)
        """
        # Project to embedding dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


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
            context_dim: Dimension of context features (from TimeSeriesContextEncoder)
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
    2. Cross-attention with 1D context (from TimeSeriesContextEncoder)
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
            context_dim: Dimension of context from TimeSeriesContextEncoder
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
    """Downsampling block: ResBlocks + optional attention + downsample."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_cross_attention: bool = False,
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
        
        # Simple self-attention (optional) - legacy mode without cross-attention
        self.use_attention = use_attention and not use_cross_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.GroupNorm(num_groups, out_channels)
        
        # Spatial Transformer with cross-attention (for hybrid conditioning)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.spatial_transformer = SpatialTransformerBlock(
                channels=out_channels,
                context_dim=context_dim,
                num_heads=4,
                num_groups=num_groups
            )
        
        # Downsample: strided conv with same kernel proportions
        # Padding calculated for output size = ceil(input_size / 2)
        downsample_padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=downsample_padding)
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            t_emb: Time embedding of shape (batch, time_emb_dim)
            encoder_hidden_states: Context from TimeSeriesContextEncoder,
                                   shape (batch, seq_len, context_dim)
        
        Returns:
            (downsampled output, skip connection output before downsampling)
        """
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        
        if self.use_attention:
            b, c, h, w = x.shape
            # Reshape for attention: (batch, h*w, channels)
            x_flat = x.view(b, c, h * w).permute(0, 2, 1)
            x_norm = self.attn_norm(x.view(b, c, -1)).view(b, c, h * w).permute(0, 2, 1)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
            x = x + attn_out.permute(0, 2, 1).view(b, c, h, w)
        
        if self.use_cross_attention:
            x = self.spatial_transformer(x, encoder_hidden_states)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block: upsample + concat skip + ResBlocks + optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_cross_attention: bool = False,
        context_dim: int = 128,
        num_groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3)
    ):
        super().__init__()
        
        # Upsample: use transposed conv with kernel = stride * 2 to avoid checkerboard artifacts
        # For stride=2, kernel=(4,4) or proportional rectangular version
        upsample_kernel = (kernel_size[0] + 1, kernel_size[1] + 1)  # Slightly larger for smooth upsampling
        upsample_padding = (upsample_kernel[0] // 4, upsample_kernel[1] // 4)  # Adjusted padding
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=upsample_kernel, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # First block takes concatenated features
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim, num_groups, kernel_size=kernel_size)
            )
        
        # Simple self-attention (optional) - legacy mode without cross-attention
        self.use_attention = use_attention and not use_cross_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.GroupNorm(num_groups, out_channels)
        
        # Spatial Transformer with cross-attention (for hybrid conditioning)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
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
        """
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            skip: Skip connection tensor of shape (batch, skip_channels, height, width)
            t_emb: Time embedding of shape (batch, time_emb_dim)
            encoder_hidden_states: Context from TimeSeriesContextEncoder,
                                   shape (batch, seq_len, context_dim)
        
        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        x = self.upsample(x)
        
        # Handle size mismatch from non-square images
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        
        if self.use_attention:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).permute(0, 2, 1)
            x_norm = self.attn_norm(x.view(b, c, -1)).view(b, c, h * w).permute(0, 2, 1)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
            x = x + attn_out.permute(0, 2, 1).view(b, c, h, w)
        
        if self.use_cross_attention:
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
        use_cross_attention: bool = False,
        context_dim: int = 128
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # Initial ResBlock to transform features
        self.res_in = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
        
        # Dilated convolution stack - each with increasing dilation on time axis
        self.dilated_blocks = nn.ModuleList()
        for d in dilation_factors:
            self.dilated_blocks.append(
                DilatedConvBlock(
                    channels=channels,
                    time_emb_dim=time_emb_dim,
                    kernel_size=kernel_size,
                    dilation=(1, d),  # Only dilate on width (time) axis
                    num_groups=num_groups
                )
            )
        
        # Self-attention for global context (if not using cross-attention)
        if not use_cross_attention:
            self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.GroupNorm(num_groups, channels)
        else:
            # Spatial Transformer with cross-attention
            self.spatial_transformer = SpatialTransformerBlock(
                channels=channels,
                context_dim=context_dim,
                num_heads=4,
                num_groups=num_groups
            )
        
        # Final ResBlock
        self.res_out = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
        
        # Log the receptive field calculation
        rf = sum([(kernel_size[1] - 1) * d for d in dilation_factors]) + 1
        logger.info(f"DilatedMiddleBlock: dilations={dilation_factors}, temporal receptive field={rf} steps")
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            t_emb: Time embedding of shape (batch, time_emb_dim)
            encoder_hidden_states: Context from TimeSeriesContextEncoder,
                                   shape (batch, seq_len, context_dim)
        
        Returns:
            Output tensor of shape (batch, channels, height, width)
        """
        # Initial transformation
        x = self.res_in(x, t_emb)
        
        # Apply dilated convolutions
        for dilated_block in self.dilated_blocks:
            x = dilated_block(x, t_emb)
        
        # Attention
        if self.use_cross_attention:
            x = self.spatial_transformer(x, encoder_hidden_states)
        else:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).permute(0, 2, 1)
            x_norm = self.attn_norm(x.view(b, c, -1)).view(b, c, h * w).permute(0, 2, 1)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
            x = x + attn_out.permute(0, 2, 1).view(b, c, h, w)
        
        # Final transformation
        x = self.res_out(x, t_emb)
        return x


class MiddleBlock(nn.Module):
    """Middle block: ResBlock + Attention + ResBlock.
    
    Legacy version without dilated convolutions.
    For expanded receptive field, use DilatedMiddleBlock instead.
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        use_cross_attention: bool = False,
        context_dim: int = 128
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        
        self.res1 = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
        
        if not use_cross_attention:
            self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.GroupNorm(num_groups, channels)
        else:
            # Spatial Transformer with cross-attention
            self.spatial_transformer = SpatialTransformerBlock(
                channels=channels,
                context_dim=context_dim,
                num_heads=4,
                num_groups=num_groups
            )
        
        self.res2 = ResidualBlock(channels, channels, time_emb_dim, num_groups, kernel_size=kernel_size)
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            t_emb: Time embedding of shape (batch, time_emb_dim)
            encoder_hidden_states: Context from TimeSeriesContextEncoder,
                                   shape (batch, seq_len, context_dim)
        
        Returns:
            Output tensor of shape (batch, channels, height, width)
        """
        x = self.res1(x, t_emb)
        
        if self.use_cross_attention:
            x = self.spatial_transformer(x, encoder_hidden_states)
        else:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).permute(0, 2, 1)
            x_norm = self.attn_norm(x.view(b, c, -1)).view(b, c, h * w).permute(0, 2, 1)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
            x = x + attn_out.permute(0, 2, 1).view(b, c, h, w)
        
        x = self.res2(x, t_emb)
        return x


class ConditioningEncoder(nn.Module):
    """CNN encoder for the past context image with global context.
    
    Processes the past 2D representation to produce conditioning features.
    Uses a combination of:
    1. Local features via CNN
    2. Global temporal context via pooling + broadcast
    
    Note: When use_coordinate_channel is enabled, in_channels=2 (data + vertical coords).
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64, height: int = 128, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__()
        self.out_channels = out_channels
        
        # Calculate padding for 'same' output size
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Local feature encoder - now handles variable input channels
        self.local_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv2d(64, out_channels // 2, kernel_size=kernel_size, padding=padding),
        )
        
        # Global context encoder: pool over time, then project
        # This captures the overall pattern of the past without losing it to interpolation
        # Uses height dimension of kernel only (width=1 for pooled temporal dimension)
        self.global_pool = nn.AdaptiveAvgPool2d((height, 1))  # Pool temporal dim to 1
        global_kernel = (kernel_size[0], 1)  # Only vertical kernel for global features
        global_padding = (kernel_size[0] // 2, 0)
        self.global_proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=global_kernel, padding=global_padding),
            nn.SiLU(),
            nn.Conv2d(32, out_channels // 2, kernel_size=global_kernel, padding=global_padding),
        )
    
    def forward(self, x: torch.Tensor, target_width: int) -> torch.Tensor:
        """
        Args:
            x: Past context image of shape (batch, 1, height, past_len)
            target_width: Width of the target (future) image to match
            
        Returns:
            Encoded features of shape (batch, out_channels, height, target_width)
        """
        batch, _, height, _ = x.shape
        
        # Global context: pool time dimension, then broadcast to target width
        global_feat = self.global_pool(x)  # (batch, 1, height, 1)
        global_feat = self.global_proj(global_feat)  # (batch, out_channels//2, height, 1)
        global_feat = global_feat.expand(-1, -1, -1, target_width)  # Broadcast to target width
        
        # Local features: use the END of the past context (most relevant for forecasting)
        # Take the last `target_width` timesteps or interpolate if past is shorter
        local_x = x[:, :, :, -target_width:] if x.shape[3] >= target_width else \
                  F.interpolate(x, size=(height, target_width), mode='bilinear', align_corners=False)
        local_feat = self.local_encoder(local_x)  # (batch, out_channels//2, height, target_width)
        
        # Concatenate global and local features
        combined = torch.cat([global_feat, local_feat], dim=1)  # (batch, out_channels, height, target_width)
        
        return combined


class ConditionalUNet2D(nn.Module):
    """Conditional 2D U-Net for diffusion-based time series forecasting.
    
    The model predicts noise ε given:
    - Noisy future image x_t
    - Diffusion timestep t
    - Past context image (conditioning)
    - Optional: 1D context embeddings from TimeSeriesContextEncoder (hybrid mode)
    
    Conditioning modes (controlled by conditioning_mode parameter):
    - "visual_concat": Directly concatenate past 2D image channels to input.
                       The past visual is passed through WITHOUT the ConditioningEncoder,
                       allowing the model to directly "see" the past trajectory pixels.
    - "vector_embedding": Use ConditioningEncoder to extract local/global features,
                          then concatenate encoded features to input. (Original behavior)
    
    When use_hybrid_condition=True, cross-attention layers are added at attention_levels
    to attend to 1D context embeddings from TimeSeriesContextEncoder.
    
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
        cond_channels: int = 64,
        num_groups: int = 8,
        image_height: int = 128,
        kernel_size: Tuple[int, int] = (3, 3),
        use_dilated_middle: bool = False,
        use_hybrid_condition: bool = False,
        context_dim: int = 128,
        conditioning_mode: str = "visual_concat",
        visual_cond_channels: int = 1,
        cond_in_channels: Optional[int] = None
    ):
        """
        Args:
            in_channels: Number of input channels (num_vars + aux_channels)
            out_channels: Number of output channels (num_variables for noise prediction)
            channels: Channel dimensions at each U-Net level
            num_res_blocks: Number of residual blocks per level
            attention_levels: Which levels to apply self-attention (0-indexed)
            time_emb_dim: Dimension of time embedding
            cond_channels: Number of channels from conditioning encoder (vector_embedding mode)
            num_groups: Number of groups for GroupNorm
            image_height: Height of the 2D image representation
            kernel_size: Tuple (height, width) for convolutional kernels.
                         Allows rectangular kernels (e.g., (3, 5) for 3 height, 5 width).
                         Height = value axis, Width = temporal axis.
            use_dilated_middle: If True, use DilatedMiddleBlock with exponentially
                               increasing dilations for expanded temporal receptive field.
            use_hybrid_condition: If True, enable cross-attention with 1D context embeddings
                                  from TimeSeriesContextEncoder at attention levels.
            context_dim: Dimension of 1D context embeddings (from TimeSeriesContextEncoder).
            conditioning_mode: "visual_concat" or "vector_embedding". Controls how past
                              context is fed to the model.
            visual_cond_channels: Number of visual conditioning channels (used in visual_concat mode).
                                  Typically equals num_variables (past image channels).
            cond_in_channels: Number of input channels for the ConditioningEncoder.
                              If None, defaults to in_channels.
                              Used when noisy input x and past cond have different channel counts
                              (e.g. when guidance channels are added to x but not cond).
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.use_hybrid_condition = use_hybrid_condition
        self.conditioning_mode = conditioning_mode
        self.visual_cond_channels = visual_cond_channels
        
        # Default cond_in_channels to in_channels if not specified
        if cond_in_channels is None:
            cond_in_channels = in_channels
        
        # Calculate padding for 'same' output size
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Conditioning encoder - only used in vector_embedding mode
        if conditioning_mode == "vector_embedding":
            self.cond_encoder = ConditioningEncoder(
                in_channels=cond_in_channels, out_channels=cond_channels, height=image_height, kernel_size=kernel_size
            )
            init_conv_in_channels = in_channels + cond_channels
        else:
            # visual_concat mode: no ConditioningEncoder, past visual is concatenated directly
            self.cond_encoder = None
            # Input = noisy_future (in_channels) + past_visual (visual_cond_channels)
            init_conv_in_channels = in_channels + visual_cond_channels
        
        # Initial convolution with configurable kernel size
        self.init_conv = nn.Conv2d(init_conv_in_channels, channels[0], kernel_size=kernel_size, padding=padding)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            use_attn = i in attention_levels
            # Use cross-attention at attention levels when hybrid conditioning is enabled
            use_cross_attn = use_hybrid_condition and use_attn
            self.down_blocks.append(
                DownBlock(
                    in_ch, out_ch, time_emb_dim, num_res_blocks,
                    use_attention=use_attn,
                    use_cross_attention=use_cross_attn,
                    context_dim=context_dim,
                    num_groups=num_groups,
                    kernel_size=kernel_size
                )
            )
            in_ch = out_ch
        
        # Middle block - optionally use dilated convolutions for expanded receptive field
        # Always use cross-attention in middle block if hybrid conditioning is enabled
        if use_dilated_middle:
            self.middle = DilatedMiddleBlock(
                channels[-1], time_emb_dim, num_groups, kernel_size=kernel_size,
                use_cross_attention=use_hybrid_condition,
                context_dim=context_dim
            )
        else:
            self.middle = MiddleBlock(
                channels[-1], time_emb_dim, num_groups, kernel_size=kernel_size,
                use_cross_attention=use_hybrid_condition,
                context_dim=context_dim
            )
        
        # Upsampling path
        # Skip connections come from down blocks and have the same channels as the down block output
        # For channels [c0, c1, c2, c3]: down blocks produce skips with [c1, c2, c3] channels
        # When reversed: up_block[i] receives skip with reversed_channels[i] channels
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            skip_ch = reversed_channels[i]  # Skip channels match the down block's output (= in_ch)
            use_attn = (len(channels) - 2 - i) in attention_levels
            # Use cross-attention at attention levels when hybrid conditioning is enabled
            use_cross_attn = use_hybrid_condition and use_attn
            self.up_blocks.append(
                UpBlock(
                    in_ch, out_ch, skip_ch, time_emb_dim, num_res_blocks,
                    use_attention=use_attn,
                    use_cross_attention=use_cross_attn,
                    context_dim=context_dim,
                    num_groups=num_groups,
                    kernel_size=kernel_size
                )
            )
        
        # Final convolution with configurable kernel size
        self.final_norm = nn.GroupNorm(num_groups, channels[0])
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=kernel_size, padding=padding)
        
        logger.info(f"ConditionalUNet2D initialized with channels={channels}, kernel_size={kernel_size}")
        logger.info(f"  Conditioning mode: {conditioning_mode}")
        if conditioning_mode == "visual_concat":
            logger.info(f"  Visual concat: {visual_cond_channels} past image channels directly concatenated")
        else:
            logger.info(f"  Vector embedding: ConditioningEncoder with {cond_channels} output channels")
        if use_hybrid_condition:
            logger.info(f"  Hybrid conditioning enabled: cross-attention with context_dim={context_dim}")
    
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
            cond: Past context conditioning:
                  - In "vector_embedding" mode: shape (batch, in_channels, height, past_len)
                  - In "visual_concat" mode: shape (batch, visual_cond_channels, height, future_len)
                    Already cropped/interpolated to target width by the caller.
            encoder_hidden_states: Optional 1D context embeddings from TimeSeriesContextEncoder,
                                   shape (batch, seq_len, context_dim). Required if use_hybrid_condition=True.
            
        Returns:
            Predicted noise of shape (batch, out_channels, height, future_len)
        """
        # Get time embeddings
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Prepare conditioning features based on mode
        if self.conditioning_mode == "vector_embedding":
            # Use ConditioningEncoder to extract local/global features
            target_width = x.shape[3]
            cond_features = self.cond_encoder(cond, target_width)
        else:
            # visual_concat mode: cond is already the past visual at target width
            # Just use it directly
            cond_features = cond
        
        # Concatenate along channel dimension
        x = torch.cat([x, cond_features], dim=1)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Downsampling with skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb, encoder_hidden_states)
            skips.append(skip)
        
        # Middle
        x = self.middle(x, t_emb, encoder_hidden_states)
        
        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, t_emb, encoder_hidden_states)
        
        # Final output
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        
        return x

