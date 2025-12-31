"""
Conditional 2D U-Net for Diffusion-based Time Series Forecasting.

This U-Net is designed for non-square, time-series-shaped images:
- Height: Fixed (e.g., 128 - the value resolution)
- Width: Variable (sequence length)

Key features:
- Residual blocks with Group Normalization
- Sinusoidal time embedding for diffusion timestep
- Conditioning via channel-wise concatenation of past context image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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
        dropout: float = 0.1
    ):
        super().__init__()
        
        # First convolution block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second convolution block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
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
        num_groups: int = 8
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim, num_groups)
            )
        
        # Simple self-attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.GroupNorm(num_groups, out_channels)
        
        # Downsample: 2x2 average pooling or strided conv
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
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
        num_groups: int = 8
    ):
        super().__init__()
        
        # Upsample
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # First block takes concatenated features
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim, num_groups)
            )
        
        # Simple self-attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.GroupNorm(num_groups, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
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
        
        return x


class MiddleBlock(nn.Module):
    """Middle block: ResBlock + Attention + ResBlock."""
    
    def __init__(self, channels: int, time_emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_emb_dim, num_groups)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.attn_norm = nn.GroupNorm(num_groups, channels)
        self.res2 = ResidualBlock(channels, channels, time_emb_dim, num_groups)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        
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
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64, height: int = 128):
        super().__init__()
        self.out_channels = out_channels
        
        # Local feature encoder
        self.local_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels // 2, kernel_size=3, padding=1),
        )
        
        # Global context encoder: pool over time, then project
        # This captures the overall pattern of the past without losing it to interpolation
        self.global_pool = nn.AdaptiveAvgPool2d((height, 1))  # Pool temporal dim to 1
        self.global_proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.SiLU(),
            nn.Conv2d(32, out_channels // 2, kernel_size=(3, 1), padding=(1, 0)),
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
    
    Conditioning is done via simple channel-wise concatenation:
    Input = [noisy_future, past_context_features] along channel dim
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
        image_height: int = 128
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (1 for noise prediction)
            channels: Channel dimensions at each U-Net level
            num_res_blocks: Number of residual blocks per level
            attention_levels: Which levels to apply self-attention (0-indexed)
            time_emb_dim: Dimension of time embedding
            cond_channels: Number of channels from conditioning encoder
            num_groups: Number of groups for GroupNorm
            image_height: Height of the 2D image representation
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Conditioning encoder with proper height
        self.cond_encoder = ConditioningEncoder(in_channels=1, out_channels=cond_channels, height=image_height)
        
        # Initial convolution
        # Input: noisy_future (in_channels) + cond_features (cond_channels)
        self.init_conv = nn.Conv2d(in_channels + cond_channels, channels[0], kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            use_attn = i in attention_levels
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attn, num_groups)
            )
            in_ch = out_ch
        
        # Middle block
        self.middle = MiddleBlock(channels[-1], time_emb_dim, num_groups)
        
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
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, skip_ch, time_emb_dim, num_res_blocks, use_attn, num_groups)
            )
        
        # Final convolution
        self.final_norm = nn.GroupNorm(num_groups, channels[0])
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        
        logger.info(f"ConditionalUNet2D initialized with channels={channels}")
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy future image of shape (batch, 1, height, future_len)
            t: Diffusion timesteps of shape (batch,)
            cond: Past context image of shape (batch, 1, height, past_len)
            
        Returns:
            Predicted noise of shape (batch, 1, height, future_len)
        """
        # Get time embeddings
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Encode conditioning (past context) - now properly handles dimension matching
        target_width = x.shape[3]
        cond_features = self.cond_encoder(cond, target_width)
        
        # Concatenate along channel dimension
        x = torch.cat([x, cond_features], dim=1)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Downsampling with skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb)
            skips.append(skip)
        
        # Middle
        x = self.middle(x, t_emb)
        
        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, t_emb)
        
        # Final output
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        
        return x

