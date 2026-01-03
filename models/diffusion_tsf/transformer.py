import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding (same as in U-Net)."""
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with optional dropout."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    """DiT-style transformer for 2D stripe images.

    - Splits (1, H, W) into non-overlapping patches (pH x pW)
    - Flattens to sequence, adds learned positional embeddings
    - Adds time and context tokens
    - Runs Transformer encoder, then projects back to patches and reshapes
    """

    def __init__(
        self,
        image_height: int = 128,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_height % patch_size == 0, "image_height must be divisible by patch_size"
        self.image_height = image_height
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_dim = patch_size * patch_size  # single-channel input

        # Patch projection
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Positional embeddings (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))  # large max, will slice

        # Time projection
        self.time_proj = nn.Linear(embed_dim, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection back to patch
        self.patch_out = nn.Linear(embed_dim, patch_dim)

    def _get_pos_embed(self, num_patches: int) -> torch.Tensor:
        # Slice learned positional embeddings to needed length
        if num_patches > self.pos_embed.shape[1]:
            # If more needed, interpolate
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_patches,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed[:, :num_patches, :]
        return pos

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy future image (B, 1, H, W)
            t: Timesteps (B,)
            cond: Past context image (B, 1, H, W_past)
        Returns:
            Noise prediction, same shape as x
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # 1. Process X (noisy future)
        # Pad width to multiple of patch size
        pad_w_x = (p - W % p) % p
        if pad_w_x > 0:
            x = F.pad(x, (0, pad_w_x, 0, 0), mode="reflect")
        _, _, _, Wp = x.shape

        # Extract patches for x: (B, num_x, p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)
        x_patches = x_patches.contiguous().view(B, C, -1, p, p)
        x_patches = x_patches.view(B, -1, p * p)
        num_x = x_patches.shape[1]

        # 2. Process Cond (clean past)
        # Pad width to multiple of patch size
        W_cond = cond.shape[3]
        pad_w_cond = (p - W_cond % p) % p
        if pad_w_cond > 0:
            cond = F.pad(cond, (0, pad_w_cond, 0, 0), mode="reflect")
        
        # Extract patches for cond: (B, num_cond, p*p)
        cond_patches = cond.unfold(2, p, p).unfold(3, p, p)
        cond_patches = cond_patches.contiguous().view(B, C, -1, p, p)
        cond_patches = cond_patches.view(B, -1, p * p)
        num_cond = cond_patches.shape[1]

        # 3. Embed both
        # We share patch_embed for both as they are in same domain
        x_tokens = self.patch_embed(x_patches)
        cond_tokens = self.patch_embed(cond_patches)

        # 4. Positional Embeddings
        # We treat cond + x as one long sequence
        total_len = num_cond + num_x
        pos = self._get_pos_embed(total_len).to(x.device)
        
        # Add positions
        cond_tokens = cond_tokens + pos[:, :num_cond, :]
        x_tokens = x_tokens + pos[:, num_cond:, :]

        # 5. Time Token
        t_emb = get_timestep_embedding(t, self.embed_dim).to(x.device)
        t_tok = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, D)

        # 6. Concatenate: [t, cond, x]
        tokens = torch.cat([t_tok, cond_tokens, x_tokens], dim=1)

        # 7. Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # 8. Extract x tokens (skip t and cond)
        # t is 1 token, cond is num_cond tokens
        x_out_tokens = tokens[:, 1 + num_cond:, :]

        # 9. Project back
        patch_out = self.patch_out(x_out_tokens)

        # Reshape to image
        patch_out = patch_out.view(B, H // p, Wp // p, p, p)
        patch_out = patch_out.permute(0, 1, 3, 2, 4).contiguous()
        patch_out = patch_out.view(B, 1, H, Wp)

        # Remove padding
        if pad_w_x > 0:
            patch_out = patch_out[:, :, :, :W]

        return patch_out

