"""
Shape-Augmented iTransformer: Fusing Time Series with 2D Shape Representations.

This module combines:
1. iTransformer: Inverted transformer that treats variates as tokens
2. CNN Branch: Extracts shape features from 2D "smudged" representations

The fusion allows the model to leverage both temporal patterns (iTransformer)
and visual shape patterns (CNN) for improved forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

# Setup path for iTransformer imports
script_dir = os.path.dirname(os.path.abspath(__file__))
itrans_dir = os.path.join(script_dir, '..', 'iTransformer')
if itrans_dir not in sys.path:
    sys.path.insert(0, itrans_dir)

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


@dataclass
class ShapeAugmentedConfig:
    """Configuration for ShapeAugmentediTransformer.
    
    Attributes:
        # Sequence parameters
        seq_len: int - Input sequence length (lookback)
        pred_len: int - Prediction horizon length
        num_variates: int - Number of input variates/channels
        
        # iTransformer parameters
        d_model: int - Model dimension
        n_heads: int - Number of attention heads
        e_layers: int - Number of encoder layers
        d_ff: int - Feed-forward dimension
        dropout: float - Dropout rate
        activation: str - Activation function ('relu' or 'gelu')
        use_norm: bool - Whether to use RevIN normalization
        
        # CNN Branch parameters
        cnn_channels: List[int] - Channel dimensions for each CNN layer
        cnn_kernel_sizes: List[Tuple[int, int]] - Kernel sizes for each layer
        cnn_use_batchnorm: bool - Whether to use batch normalization
        cnn_pooling: str - Pooling type ('max' or 'avg')
        
        # Image parameters (for smudged representation)
        image_height: int - Height of the 2D representation
        
        # Fusion parameters
        fusion_mode: str - How to fuse CNN and Transformer outputs
            - 'add': Add CNN embedding to each variate token
            - 'concat': Concatenate and project
            - 'cross_attention': CNN provides context for cross-attention
    """
    # Sequence parameters
    seq_len: int = 512
    pred_len: int = 96
    num_variates: int = 7  # ETTh1 has 7 variates
    
    # iTransformer parameters
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = 'gelu'
    use_norm: bool = True
    factor: int = 1  # Attention factor (for ProbAttention)
    output_attention: bool = False
    embed: str = 'fixed'
    freq: str = 'h'
    
    # CNN Branch parameters
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    cnn_kernel_sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(3, 3), (3, 3), (3, 3), (3, 3)]
    )
    cnn_use_batchnorm: bool = True
    cnn_pooling: str = 'max'  # 'max' or 'avg'
    
    # Image parameters
    image_height: int = 128  # Height of 2D smudged representation
    
    # Fusion parameters
    fusion_mode: str = 'add'  # 'add', 'concat', or 'cross_attention'
    
    # Class strategy for iTransformer
    class_strategy: str = 'projection'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.fusion_mode in ['add', 'concat', 'cross_attention'], \
            f"fusion_mode must be 'add', 'concat', or 'cross_attention', got {self.fusion_mode}"
        assert self.cnn_pooling in ['max', 'avg'], \
            f"cnn_pooling must be 'max' or 'avg', got {self.cnn_pooling}"
        assert len(self.cnn_channels) == len(self.cnn_kernel_sizes), \
            "cnn_channels and cnn_kernel_sizes must have same length"
        assert all(k[0] % 2 == 1 and k[1] % 2 == 1 for k in self.cnn_kernel_sizes), \
            "All kernel sizes must be odd for symmetric padding"


class CNNShapeEncoder(nn.Module):
    """Lightweight 2D CNN for extracting shape features from smudged images.
    
    Input: (Batch, Variates, Height, Width) - variates act as channels
    Output: (Batch, d_model) or (Batch, Variates, d_model) depending on output_per_variate
    
    Architecture:
    - Stack of Conv2d -> BatchNorm -> ReLU -> MaxPool layers
    - Global average pooling at the end
    - Final projection to d_model dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_sizes: List[Tuple[int, int]],
        d_model: int,
        use_batchnorm: bool = True,
        pooling: str = 'max',
        output_per_variate: bool = False
    ):
        """
        Args:
            in_channels: Number of input channels (= num_variates)
            channels: List of channel dimensions for each layer
            kernel_sizes: List of (H, W) kernel sizes for each layer
            d_model: Output embedding dimension
            use_batchnorm: Whether to use BatchNorm after each conv
            pooling: Pooling type ('max' or 'avg')
            output_per_variate: If True, output (Batch, num_variates, d_model)
                               If False, output (Batch, d_model)
        """
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.output_per_variate = output_per_variate
        
        layers = []
        prev_channels = in_channels
        
        for i, (out_ch, kernel) in enumerate(zip(channels, kernel_sizes)):
            # Padding for same spatial dims before pooling
            pad_h = kernel[0] // 2
            pad_w = kernel[1] // 2
            
            layers.append(
                nn.Conv2d(prev_channels, out_ch, kernel_size=kernel, padding=(pad_h, pad_w))
            )
            
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            
            layers.append(nn.ReLU(inplace=True))
            
            # Pool to reduce spatial dimensions
            if pooling == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            prev_channels = out_ch
        
        self.encoder = nn.Sequential(*layers)
        self.final_channels = channels[-1]
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(self.final_channels, d_model)
        
        # Optional per-variate output
        if output_per_variate:
            # Additional projection to expand to per-variate embeddings
            self.variate_expansion = nn.Linear(d_model, in_channels * d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Smudged image tensor of shape (Batch, Variates, Height, Width)
        
        Returns:
            If output_per_variate=False: (Batch, d_model)
            If output_per_variate=True: (Batch, Variates, d_model)
        """
        batch_size = x.shape[0]
        
        # Pass through CNN encoder
        # (B, Variates, H, W) -> (B, final_channels, H', W')
        features = self.encoder(x)
        
        # Global average pooling
        # (B, final_channels, H', W') -> (B, final_channels, 1, 1)
        pooled = self.global_pool(features)
        
        # Flatten and project
        # (B, final_channels) -> (B, d_model)
        flat = pooled.view(batch_size, -1)
        embedding = self.projection(flat)
        
        if self.output_per_variate:
            # Expand to per-variate embeddings
            # (B, d_model) -> (B, num_variates * d_model) -> (B, num_variates, d_model)
            expanded = self.variate_expansion(embedding)
            embedding = expanded.view(batch_size, self.in_channels, self.d_model)
        
        return embedding


class CrossAttentionFusion(nn.Module):
    """Cross-attention layer for fusing CNN context with transformer tokens.
    
    The transformer tokens attend to CNN-derived context.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor,  # (B, N, d_model) - transformer tokens
        context: torch.Tensor  # (B, M, d_model) - CNN context
    ) -> torch.Tensor:
        """Apply cross-attention from query (transformer) to context (CNN)."""
        # Cross-attention: Q from transformer, K/V from CNN
        attn_out, _ = self.cross_attn(query, context, context)
        
        # Residual connection and layer norm
        out = self.norm(query + self.dropout(attn_out))
        
        return out


class ShapeAugmentediTransformer(nn.Module):
    """
    Shape-Augmented iTransformer for time series forecasting.
    
    Combines:
    1. iTransformer branch: Processes raw time series (B, L, N)
    2. CNN branch: Processes 2D smudged representation (B, N, H, W)
    
    The two representations are fused before the final projection.
    
    Input shapes:
        x_ts: (Batch, Lookback, Variates) - raw time series
        x_img: (Batch, Variates, Height, Width) - smudged 2D image
    
    Output shape:
        (Batch, Pred_Len, Variates)
    """
    
    def __init__(self, config: ShapeAugmentedConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_variates = config.num_variates
        self.d_model = config.d_model
        self.use_norm = config.use_norm
        self.fusion_mode = config.fusion_mode
        self.output_attention = config.output_attention
        
        # ========================
        # iTransformer Branch
        # ========================
        
        # Inverted embedding: (B, L, N) -> (B, N, d_model)
        self.enc_embedding = DataEmbedding_inverted(
            c_in=config.seq_len,
            d_model=config.d_model,
            embed_type=config.embed,
            freq=config.freq,
            dropout=config.dropout
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention
                        ),
                        config.d_model,
                        config.n_heads
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model)
        )
        
        # ========================
        # CNN Shape Branch
        # ========================
        
        # For 'add' and 'concat' fusion, we need per-variate output
        output_per_variate = (config.fusion_mode in ['add', 'concat'])
        
        self.cnn_encoder = CNNShapeEncoder(
            in_channels=config.num_variates,
            channels=config.cnn_channels,
            kernel_sizes=config.cnn_kernel_sizes,
            d_model=config.d_model,
            use_batchnorm=config.cnn_use_batchnorm,
            pooling=config.cnn_pooling,
            output_per_variate=output_per_variate
        )
        
        # ========================
        # Fusion Components
        # ========================
        
        if config.fusion_mode == 'concat':
            # Concatenate transformer and CNN outputs, then project
            self.fusion_proj = nn.Linear(config.d_model * 2, config.d_model)
        elif config.fusion_mode == 'cross_attention':
            # Create context from CNN for cross-attention
            self.cnn_context_proj = nn.Linear(config.d_model, config.d_model)
            self.cross_attn_fusion = CrossAttentionFusion(
                config.d_model, config.n_heads, config.dropout
            )
        # 'add' mode doesn't need extra parameters
        
        # ========================
        # Output Projection
        # ========================
        
        # Project from d_model to pred_len
        self.projector = nn.Linear(config.d_model, config.pred_len, bias=True)
    
    def _apply_normalization(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply RevIN-style normalization.
        
        Args:
            x: (B, L, N) input tensor
            
        Returns:
            normalized x, means, stdev
        """
        means = x.mean(dim=1, keepdim=True).detach()
        x_normalized = x - means
        stdev = torch.sqrt(
            torch.var(x_normalized, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_normalized = x_normalized / stdev
        return x_normalized, means, stdev
    
    def _reverse_normalization(
        self,
        x: torch.Tensor,
        means: torch.Tensor,
        stdev: torch.Tensor
    ) -> torch.Tensor:
        """Reverse RevIN normalization.
        
        Args:
            x: (B, S, N) output tensor
            means: (B, 1, N) means from input
            stdev: (B, 1, N) stdev from input
            
        Returns:
            denormalized x
        """
        # Expand means and stdev to match output shape
        x = x * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        x = x + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return x
    
    def forward(
        self,
        x_ts: torch.Tensor,
        x_img: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with both time series and image inputs.
        
        Args:
            x_ts: Raw time series, shape (Batch, Lookback, Variates)
            x_img: Smudged 2D image, shape (Batch, Variates, Height, Width)
            x_mark_enc: Optional temporal marks for encoder
            x_dec: Optional decoder input (unused, for interface compatibility)
            x_mark_dec: Optional temporal marks for decoder (unused)
        
        Returns:
            Predictions of shape (Batch, Pred_Len, Variates)
        """
        batch_size = x_ts.shape[0]
        
        # Validate input shapes
        assert x_ts.shape == (batch_size, self.seq_len, self.num_variates), \
            f"Expected x_ts shape {(batch_size, self.seq_len, self.num_variates)}, got {x_ts.shape}"
        assert x_img.shape[0] == batch_size and x_img.shape[1] == self.num_variates, \
            f"Expected x_img shape (B, {self.num_variates}, H, W), got {x_img.shape}"
        
        # ========================
        # Normalization
        # ========================
        
        if self.use_norm:
            x_ts, means, stdev = self._apply_normalization(x_ts)
        
        # ========================
        # iTransformer Branch
        # ========================
        
        # Embed: (B, L, N) -> (B, N, d_model)
        # The inverted embedding treats each variate as a token
        enc_out = self.enc_embedding(x_ts, x_mark_enc)
        
        # Encoder: (B, N, d_model) -> (B, N, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # ========================
        # CNN Shape Branch
        # ========================
        
        # Extract shape features from 2D representation
        shape_embedding = self.cnn_encoder(x_img)
        # shape_embedding: (B, N, d_model) for add/concat, (B, d_model) for cross_attn
        
        # ========================
        # Fusion
        # ========================
        
        if self.fusion_mode == 'add':
            # Add shape embedding to each variate token
            # shape_embedding: (B, N, d_model), enc_out: (B, N, d_model)
            fused = enc_out + shape_embedding
            
        elif self.fusion_mode == 'concat':
            # Concatenate along feature dimension and project back
            # (B, N, d_model) + (B, N, d_model) -> (B, N, 2*d_model) -> (B, N, d_model)
            concat = torch.cat([enc_out, shape_embedding], dim=-1)
            fused = self.fusion_proj(concat)
            
        elif self.fusion_mode == 'cross_attention':
            # CNN embedding provides context for cross-attention
            # shape_embedding: (B, d_model) -> expand to (B, 1, d_model) for attention
            if shape_embedding.dim() == 2:
                context = shape_embedding.unsqueeze(1)  # (B, 1, d_model)
            else:
                context = shape_embedding  # Already (B, M, d_model)
            
            context = self.cnn_context_proj(context)
            fused = self.cross_attn_fusion(enc_out, context)
        
        # ========================
        # Output Projection
        # ========================
        
        # Project: (B, N, d_model) -> (B, N, pred_len)
        dec_out = self.projector(fused)
        
        # Permute: (B, N, pred_len) -> (B, pred_len, N)
        dec_out = dec_out.permute(0, 2, 1)
        
        # Filter to correct number of variates
        dec_out = dec_out[:, :, :self.num_variates]
        
        # ========================
        # De-normalization
        # ========================
        
        if self.use_norm:
            dec_out = self._reverse_normalization(dec_out, means, stdev)
        
        if self.output_attention:
            return dec_out, attns
        return dec_out
    
    def forecast_ts_only(
        self,
        x_ts: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forecast using only the time series branch (for comparison).
        
        This is the vanilla iTransformer forward pass.
        
        Args:
            x_ts: Raw time series, shape (Batch, Lookback, Variates)
            x_mark_enc: Optional temporal marks
        
        Returns:
            Predictions of shape (Batch, Pred_Len, Variates)
        """
        if self.use_norm:
            x_ts, means, stdev = self._apply_normalization(x_ts)
        
        # Embed and encode
        enc_out = self.enc_embedding(x_ts, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # Project
        dec_out = self.projector(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :self.num_variates]
        
        if self.use_norm:
            dec_out = self._reverse_normalization(dec_out, means, stdev)
        
        return dec_out


class VanillaiTransformer(nn.Module):
    """
    Vanilla iTransformer for comparison (no CNN branch).
    
    This is essentially the same as the original iTransformer
    but with a simplified interface.
    """
    
    def __init__(self, config: ShapeAugmentedConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_variates = config.num_variates
        self.d_model = config.d_model
        self.use_norm = config.use_norm
        self.output_attention = config.output_attention
        
        # Inverted embedding
        self.enc_embedding = DataEmbedding_inverted(
            c_in=config.seq_len,
            d_model=config.d_model,
            embed_type=config.embed,
            freq=config.freq,
            dropout=config.dropout
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention
                        ),
                        config.d_model,
                        config.n_heads
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model)
        )
        
        # Output projection
        self.projector = nn.Linear(config.d_model, config.pred_len, bias=True)
    
    def forward(
        self,
        x_ts: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_ts: Raw time series, shape (Batch, Lookback, Variates)
            x_mark_enc: Optional temporal marks
        
        Returns:
            Predictions of shape (Batch, Pred_Len, Variates)
        """
        if self.use_norm:
            means = x_ts.mean(dim=1, keepdim=True).detach()
            x_ts = x_ts - means
            stdev = torch.sqrt(
                torch.var(x_ts, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_ts = x_ts / stdev
        
        # Embed and encode
        enc_out = self.enc_embedding(x_ts, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # Project
        dec_out = self.projector(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :self.num_variates]
        
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity test
    print("Testing ShapeAugmentediTransformer...")
    
    # Create config
    config = ShapeAugmentedConfig(
        seq_len=96,
        pred_len=24,
        num_variates=7,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        cnn_channels=[16, 32, 64],
        cnn_kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        image_height=64,
        fusion_mode='add'
    )
    
    # Create model
    model = ShapeAugmentediTransformer(config)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    x_ts = torch.randn(batch_size, config.seq_len, config.num_variates)
    x_img = torch.randn(batch_size, config.num_variates, config.image_height, config.seq_len)
    
    with torch.no_grad():
        output = model(x_ts, x_img)
    
    print(f"Input x_ts shape: {x_ts.shape}")
    print(f"Input x_img shape: {x_img.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {config.pred_len}, {config.num_variates})")
    
    assert output.shape == (batch_size, config.pred_len, config.num_variates), "Shape mismatch!"
    print("✓ Forward pass test passed!")
    
    # Test vanilla comparison
    vanilla = VanillaiTransformer(config)
    print(f"\nVanilla iTransformer parameters: {count_parameters(vanilla):,}")
    
    with torch.no_grad():
        vanilla_out = vanilla(x_ts)
    
    print(f"Vanilla output shape: {vanilla_out.shape}")
    assert vanilla_out.shape == (batch_size, config.pred_len, config.num_variates), "Vanilla shape mismatch!"
    print("✓ Vanilla forward pass test passed!")
    
    # Test different fusion modes
    for fusion in ['add', 'concat', 'cross_attention']:
        config_test = ShapeAugmentedConfig(
            seq_len=96, pred_len=24, num_variates=7,
            d_model=64, n_heads=4, e_layers=2, d_ff=128,
            cnn_channels=[16, 32, 64],
            cnn_kernel_sizes=[(3, 3), (3, 3), (3, 3)],
            image_height=64,
            fusion_mode=fusion
        )
        model_test = ShapeAugmentediTransformer(config_test)
        with torch.no_grad():
            out = model_test(x_ts, x_img)
        assert out.shape == (batch_size, config_test.pred_len, config_test.num_variates)
        print(f"✓ Fusion mode '{fusion}' test passed!")
    
    print("\n✓ All tests passed!")

