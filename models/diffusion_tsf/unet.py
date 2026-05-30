"""Context adapter for FactorizedDiT cross-attention (historical module name)."""

import torch
import torch.nn as nn


class iTransformerTokenAdapter(nn.Module):
    """Projects frozen iTransformer encoder tokens to context_dim for DiT cross-attention.

    Feeds iTransformer enc_out (before its linear projector) through a projection
    and a learned per-variate identity embedding so the diffusion model can distinguish
    variates within shared-weight factorized forward passes.
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
        x = self.proj(enc_tokens)
        ids = torch.arange(V, device=enc_tokens.device)
        x = x + self.variate_embed(ids)
        return self.norm(self.drop(x))
