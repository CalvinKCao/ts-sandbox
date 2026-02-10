"""
Channel Clustering Module (CCM) Adapter.

Adapts the CCM approach from "From Similarity to Superiority: Channel Clustering
for Time Series Forecasting" (Chen et al., NeurIPS 2024) to work with our
pretrained 7-variate diffusion model.

Key idea: For datasets with >7 variates, cluster channels into 7 "super-channels",
pass through the pretrained model, then expand predictions back to original channels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def sinkhorn(out: torch.Tensor, epsilon: float = 0.05, iterations: int = 3) -> torch.Tensor:
    """Sinkhorn normalization for soft clustering assignments.
    
    Args:
        out: Raw logits (n_vars, n_clusters)
        epsilon: Temperature for softmax
        iterations: Number of Sinkhorn iterations (unused, we use simple softmax)
    
    Returns:
        Normalized probabilities (n_vars, n_clusters)
    """
    # Numerical stability: subtract max before exp (like log-sum-exp trick)
    out_stable = out / epsilon
    out_stable = out_stable - out_stable.max(dim=1, keepdim=True).values
    Q = torch.exp(out_stable)
    sum_Q = torch.sum(Q, dim=1, keepdim=True)
    Q = Q / (sum_Q + 1e-8)
    return Q


class ClusterAssigner(nn.Module):
    """Assigns channels to clusters based on learned embeddings.
    
    Computes soft cluster assignments using cosine similarity between
    channel embeddings and cluster prototypes.
    """
    
    def __init__(
        self, 
        n_vars: int, 
        n_clusters: int, 
        seq_len: int, 
        d_model: int = 128,
        epsilon: float = 0.05
    ):
        super().__init__()
        self.n_vars = n_vars
        self.n_clusters = n_clusters
        self.d_model = d_model
        self.epsilon = epsilon
        
        # Project time series to embedding space
        self.linear = nn.Linear(seq_len, d_model)
        
        # Learnable cluster embeddings (prototypes)
        self.cluster_emb = nn.Parameter(torch.empty(n_clusters, d_model))
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        
        self.l2norm = lambda x: F.normalize(x, dim=-1, p=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cluster assignments.
        
        Args:
            x: Input time series (batch, n_vars, seq_len)
            
        Returns:
            prob_avg: Cluster probabilities (n_vars, n_clusters)
            cluster_emb: Updated cluster embeddings (n_clusters, d_model)
        """
        B, C, L = x.shape
        
        # Embed each channel: (B, C, L) -> (B, C, d_model)
        x_emb = self.linear(x)
        
        # Flatten batch for similarity computation: (B*C, d_model)
        x_emb_flat = x_emb.reshape(-1, self.d_model)
        
        # Cosine similarity with cluster embeddings: (B*C, n_clusters)
        sim = torch.mm(self.l2norm(x_emb_flat), self.l2norm(self.cluster_emb).t())
        
        # Reshape to (B, C, n_clusters) and average over batch
        prob = sim.reshape(B, C, self.n_clusters)
        prob_avg = torch.mean(prob, dim=0)  # (C, n_clusters)
        
        # Apply sinkhorn normalization
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        
        return prob_avg, self.cluster_emb
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments for visualization.
        
        Args:
            x: Input time series (batch, n_vars, seq_len)
            
        Returns:
            assignments: Hard cluster assignment per channel (n_vars,)
        """
        prob_avg, _ = self.forward(x)
        return torch.argmax(prob_avg, dim=1)


def cluster_aggregate(x: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """Aggregate channels into clusters using soft assignments.
    
    Args:
        x: Input data (batch, n_vars, seq_len) 
        prob: Cluster probabilities (n_vars, n_clusters)
        
    Returns:
        Aggregated data (batch, n_clusters, seq_len)
    """
    # prob.T: (n_clusters, n_vars)
    # x: (B, n_vars, L)
    # Result: (B, n_clusters, L)
    
    # Normalize prob to sum to 1 per cluster (for weighted average)
    prob_norm = prob.t() / (prob.t().sum(dim=1, keepdim=True) + 1e-8)
    
    # Weighted sum: (B, n_clusters, L)
    return torch.einsum('kc,bcl->bkl', prob_norm, x)


def cluster_expand(x: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """Expand cluster predictions back to original channels.
    
    Args:
        x: Cluster predictions (batch, n_clusters, pred_len)
        prob: Cluster probabilities (n_vars, n_clusters)
        
    Returns:
        Expanded predictions (batch, n_vars, pred_len)
    """
    # Weighted combination of cluster predictions for each channel
    # prob: (n_vars, n_clusters)
    # x: (B, n_clusters, L)
    # Result: (B, n_vars, L)
    return torch.einsum('ck,bkl->bcl', prob, x)


def compute_cluster_loss(prob: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute clustering loss to encourage similar channels to cluster together.
    
    This implements Eq. 4 from the CCM paper: maximize intra-cluster similarity,
    minimize inter-cluster similarity.
    
    Args:
        prob: Cluster probabilities (n_vars, n_clusters)
        x: Input data (batch, n_vars, seq_len) for computing similarity
        
    Returns:
        Cluster loss scalar
    """
    # Compute pairwise channel similarity using RBF kernel
    # x: (B, C, L) -> mean over batch -> (C, L)
    x_mean = x.mean(dim=0)
    
    # Euclidean distance squared
    diff = x_mean.unsqueeze(0) - x_mean.unsqueeze(1)  # (C, C, L)
    dist_sq = (diff ** 2).sum(dim=-1)  # (C, C)
    
    # RBF similarity
    sigma = dist_sq.max().item() / 5 + 1e-8
    sim_matrix = torch.exp(-dist_sq / sigma)
    
    # Concrete Bernoulli for soft membership
    def concrete_bern(p, temp=0.07):
        noise = torch.empty_like(p).uniform_(1e-10, 1 - 1e-10)
        noise = torch.log(noise) - torch.log(1.0 - noise)
        p_logit = torch.log(p + 1e-10) - torch.log(1.0 - p + 1e-10)
        return torch.sigmoid((p_logit + noise) / temp)
    
    membership = concrete_bern(prob)  # (n_vars, n_clusters)
    
    # Cluster loss: -Tr(M^T S M) + Tr((I - MM^T) S)
    temp = torch.mm(membership.t(), sim_matrix)
    SAS = torch.mm(temp, membership)
    
    MMT = torch.mm(membership, membership.t())
    SS = (1 - MMT) * sim_matrix
    
    loss = -torch.trace(SAS) + torch.trace(SS) + membership.shape[0]
    
    # Entropy regularization to avoid degenerate solutions
    ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
    
    return loss + ent_loss


class CCMAdapter(nn.Module):
    """Wraps a pretrained 7-variate model to handle arbitrary input channels.
    
    Uses Channel Clustering Module to:
    1. Cluster input channels into 7 groups
    2. Aggregate each cluster into a single "super-channel"
    3. Pass through the pretrained 7-variate model
    4. Expand predictions back to original channel count
    """
    
    def __init__(
        self,
        n_original_vars: int,
        n_clusters: int = 7,
        seq_len: int = 512,
        d_model: int = 128,
        beta: float = 0.3,  # Weight for cluster loss
    ):
        super().__init__()
        
        if n_original_vars <= n_clusters:
            raise ValueError(
                f"CCMAdapter requires n_original_vars ({n_original_vars}) > "
                f"n_clusters ({n_clusters}). Use direct model for small datasets."
            )
        
        self.n_original_vars = n_original_vars
        self.n_clusters = n_clusters
        self.seq_len = seq_len
        self.beta = beta
        
        self.cluster_assigner = ClusterAssigner(
            n_vars=n_original_vars,
            n_clusters=n_clusters,
            seq_len=seq_len,
            d_model=d_model
        )
        
        # Cache for cluster probabilities (updated during forward)
        self.cluster_prob: Optional[torch.Tensor] = None
    
    def aggregate(self, x: torch.Tensor, compute_prob: bool = True) -> torch.Tensor:
        """Aggregate original channels into cluster channels.
        
        Args:
            x: Input (batch, n_original_vars, seq_len)
            compute_prob: If True, compute new cluster probabilities from x.
                         If False, use cached probabilities (for aggregating 
                         future tensor after computing prob from past).
            
        Returns:
            Aggregated (batch, n_clusters, seq_len)
        """
        if compute_prob:
            # Need to handle case where x has different length than expected
            # Pad or truncate to seq_len for cluster assignment
            actual_len = x.shape[-1]
            if actual_len != self.seq_len:
                # Use last seq_len timesteps, or pad with zeros if too short
                if actual_len >= self.seq_len:
                    x_for_cluster = x[:, :, -self.seq_len:]
                else:
                    pad = torch.zeros(x.shape[0], x.shape[1], self.seq_len - actual_len, 
                                     device=x.device, dtype=x.dtype)
                    x_for_cluster = torch.cat([pad, x], dim=-1)
            else:
                x_for_cluster = x
            
            self.cluster_prob, _ = self.cluster_assigner(x_for_cluster)
        
        if self.cluster_prob is None:
            raise RuntimeError("No cluster probabilities computed. Call with compute_prob=True first.")
        
        return cluster_aggregate(x, self.cluster_prob)
    
    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Expand cluster predictions back to original channels.
        
        Args:
            x: Cluster predictions (batch, n_clusters, pred_len)
            
        Returns:
            Expanded (batch, n_original_vars, pred_len)
        """
        if self.cluster_prob is None:
            raise RuntimeError("Must call aggregate() before expand()")
        return cluster_expand(x, self.cluster_prob)
    
    def get_cluster_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted cluster loss for training.
        
        Args:
            x: Input data used for similarity computation
            
        Returns:
            Weighted cluster loss (scalar)
        """
        if self.cluster_prob is None:
            return torch.tensor(0.0, device=x.device)
        return self.beta * compute_cluster_loss(self.cluster_prob, x)
    
    def get_cluster_info(self, x: torch.Tensor) -> dict:
        """Get cluster assignments and statistics for visualization.
        
        Args:
            x: Input data (batch, n_original_vars, seq_len)
            
        Returns:
            Dictionary with cluster info
        """
        with torch.no_grad():
            prob, cluster_emb = self.cluster_assigner(x)
            assignments = torch.argmax(prob, dim=1)
            
            # Count channels per cluster
            counts = torch.zeros(self.n_clusters, dtype=torch.long)
            for k in range(self.n_clusters):
                counts[k] = (assignments == k).sum()
            
            return {
                'probabilities': prob.cpu().numpy(),
                'assignments': assignments.cpu().numpy(),
                'cluster_counts': counts.cpu().numpy(),
                'cluster_embeddings': cluster_emb.cpu().numpy(),
            }


# ============================================================================
# Visualization utilities
# ============================================================================

def visualize_clusters(
    ccm_adapter: CCMAdapter,
    x: torch.Tensor,
    channel_names: Optional[list] = None,
    save_path: Optional[str] = None
) -> None:
    """Visualize cluster assignments and channel distributions.
    
    Args:
        ccm_adapter: Trained CCMAdapter
        x: Sample input data (batch, n_vars, seq_len)
        channel_names: Optional names for channels
        save_path: Path to save figure (shows if None)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    info = ccm_adapter.get_cluster_info(x)
    
    n_vars = info['probabilities'].shape[0]
    n_clusters = info['probabilities'].shape[1]
    
    if channel_names is None:
        channel_names = [f'Ch{i}' for i in range(n_vars)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Cluster assignment heatmap (probabilities)
    ax = axes[0, 0]
    im = ax.imshow(info['probabilities'], aspect='auto', cmap='Blues')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Channel')
    ax.set_title('Cluster Assignment Probabilities')
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    if n_vars <= 30:
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(channel_names, fontsize=6)
    plt.colorbar(im, ax=ax)
    
    # 2. Channels per cluster bar chart
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.arange(n_clusters))
    ax.bar(range(n_clusters), info['cluster_counts'], color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Channels')
    ax.set_title('Channels per Cluster')
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    
    # 3. Sample time series colored by cluster
    ax = axes[1, 0]
    x_sample = x[0].cpu().numpy()  # First batch sample
    assignments = info['assignments']
    
    # Plot a few channels from each cluster
    max_per_cluster = min(3, n_vars // n_clusters)
    for k in range(n_clusters):
        cluster_channels = np.where(assignments == k)[0]
        for i, ch_idx in enumerate(cluster_channels[:max_per_cluster]):
            alpha = 0.8 if i == 0 else 0.3
            label = f'C{k}' if i == 0 else None
            ax.plot(x_sample[ch_idx], color=colors[k], alpha=alpha, label=label, linewidth=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Sample Channels Colored by Cluster')
    ax.legend(loc='upper right', fontsize=8)
    
    # 4. Cluster embedding similarity matrix
    ax = axes[1, 1]
    emb = info['cluster_embeddings']
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb_norm @ emb_norm.T
    im = ax.imshow(sim, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cluster')
    ax.set_title('Cluster Embedding Similarity')
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax.set_yticklabels([f'C{i}' for i in range(n_clusters)])
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

