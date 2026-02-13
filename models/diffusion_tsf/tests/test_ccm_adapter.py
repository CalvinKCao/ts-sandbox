"""
Tests for CCM (Channel Clustering Module) adapter.

Run with: pytest models/diffusion_tsf/tests/test_ccm_adapter.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models.diffusion_tsf.ccm_adapter import (
    ClusterAssigner,
    CCMAdapter,
    cluster_aggregate,
    cluster_expand,
    compute_cluster_loss,
    sinkhorn,
)


class TestSinkhorn:
    """Tests for sinkhorn normalization."""
    
    def test_output_sums_to_one(self):
        """Each row should sum to 1 after sinkhorn."""
        x = torch.randn(10, 7)
        result = sinkhorn(x, epsilon=0.1)
        
        row_sums = result.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(10), atol=1e-5)
    
    def test_output_non_negative(self):
        """All values should be non-negative."""
        x = torch.randn(20, 5) * 10  # Large range
        result = sinkhorn(x)
        
        assert (result >= 0).all()
    
    def test_preserves_relative_order(self):
        """Higher input values should generally give higher probabilities."""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = sinkhorn(x)
        
        # Cluster 2 should have highest prob
        assert result[0, 2] > result[0, 1] > result[0, 0]


class TestClusterAssigner:
    """Tests for ClusterAssigner module."""
    
    @pytest.fixture
    def assigner(self):
        return ClusterAssigner(n_vars=21, n_clusters=7, seq_len=64, d_model=32)
    
    def test_output_shapes(self, assigner):
        """Check output tensor shapes."""
        x = torch.randn(4, 21, 64)  # batch=4, vars=21, seq=64
        
        prob, emb = assigner(x)
        
        assert prob.shape == (21, 7)  # (n_vars, n_clusters)
        assert emb.shape == (7, 32)   # (n_clusters, d_model)
    
    def test_probabilities_valid(self, assigner):
        """Probabilities should be valid (sum to 1, non-negative)."""
        x = torch.randn(2, 21, 64)
        prob, _ = assigner(x)
        
        # Non-negative
        assert (prob >= 0).all()
        
        # Each row sums to ~1
        row_sums = prob.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(21), atol=1e-4)
    
    def test_hard_assignments(self, assigner):
        """Hard cluster assignments should be valid indices."""
        x = torch.randn(2, 21, 64)
        assignments = assigner.get_cluster_assignments(x)
        
        assert assignments.shape == (21,)
        assert (assignments >= 0).all()
        assert (assignments < 7).all()


class TestClusterAggregateExpand:
    """Tests for cluster aggregation and expansion."""
    
    def test_aggregate_shape(self):
        """Aggregation should reduce channels to clusters."""
        x = torch.randn(4, 21, 100)  # batch=4, vars=21, seq=100
        prob = torch.softmax(torch.randn(21, 7), dim=1)  # (vars, clusters)
        
        result = cluster_aggregate(x, prob)
        
        assert result.shape == (4, 7, 100)  # (batch, clusters, seq)
    
    def test_expand_shape(self):
        """Expansion should restore original channel count."""
        x = torch.randn(4, 7, 50)  # (batch, clusters, pred_len)
        prob = torch.softmax(torch.randn(21, 7), dim=1)
        
        result = cluster_expand(x, prob)
        
        assert result.shape == (4, 21, 50)
    
    def test_aggregate_expand_consistency(self):
        """Aggregating then expanding should preserve some structure."""
        # If we have perfect cluster assignments (one-hot), 
        # aggregate->expand should approximately recover original
        
        n_vars, n_clusters = 14, 7
        
        # Create one-hot-ish probabilities (2 vars per cluster)
        prob = torch.zeros(n_vars, n_clusters)
        for i in range(n_vars):
            prob[i, i % n_clusters] = 1.0
        
        x = torch.randn(2, n_vars, 64)
        
        agg = cluster_aggregate(x, prob)
        exp = cluster_expand(agg, prob)
        
        # Should have same shape
        assert exp.shape == x.shape
        
        # Channels in same cluster should have same value after round-trip
        ch0, ch7 = x[0, 0], x[0, 7]  # Both map to cluster 0
        exp0, exp7 = exp[0, 0], exp[0, 7]
        
        # After aggregate (average) + expand (copy), channels in same cluster are equal
        assert torch.allclose(exp0, exp7, atol=1e-5)


class TestCCMAdapter:
    """Tests for the full CCM adapter."""
    
    @pytest.fixture
    def adapter(self):
        return CCMAdapter(n_original_vars=21, n_clusters=7, seq_len=64)
    
    def test_aggregate_shape(self, adapter):
        """Aggregate should reduce to 7 channels."""
        x = torch.randn(2, 21, 64)
        result = adapter.aggregate(x)
        
        assert result.shape == (2, 7, 64)
    
    def test_expand_shape(self, adapter):
        """Expand should restore 21 channels."""
        x = torch.randn(2, 21, 64)
        _ = adapter.aggregate(x)  # Sets cluster_prob
        
        pred = torch.randn(2, 7, 32)  # Predictions from model
        result = adapter.expand(pred)
        
        assert result.shape == (2, 21, 32)
    
    def test_cluster_prob_cached(self, adapter):
        """Cluster probabilities should be cached after aggregate."""
        x = torch.randn(2, 21, 64)
        
        assert adapter.cluster_prob is None
        
        _ = adapter.aggregate(x)
        
        assert adapter.cluster_prob is not None
        assert adapter.cluster_prob.shape == (21, 7)
    
    def test_cluster_loss_computes(self, adapter):
        """Cluster loss should be a scalar."""
        x = torch.randn(2, 21, 64)
        _ = adapter.aggregate(x)
        
        loss = adapter.get_cluster_loss(x)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() != 0  # Non-trivial loss
    
    def test_cluster_info(self, adapter):
        """get_cluster_info should return valid dict."""
        x = torch.randn(2, 21, 64)
        
        info = adapter.get_cluster_info(x)
        
        assert 'probabilities' in info
        assert 'assignments' in info
        assert 'cluster_counts' in info
        assert 'cluster_embeddings' in info
        
        assert info['probabilities'].shape == (21, 7)
        assert info['assignments'].shape == (21,)
        assert info['cluster_counts'].shape == (7,)
        assert info['cluster_counts'].sum() == 21
    
    def test_rejects_small_datasets(self):
        """Should raise error for datasets with <= n_clusters variates."""
        with pytest.raises(ValueError, match="n_original_vars"):
            CCMAdapter(n_original_vars=7, n_clusters=7, seq_len=64)
        
        with pytest.raises(ValueError, match="n_original_vars"):
            CCMAdapter(n_original_vars=5, n_clusters=7, seq_len=64)


class TestClusterLoss:
    """Tests for cluster loss computation."""
    
    def test_loss_is_scalar(self):
        """Loss should be a scalar tensor."""
        x = torch.randn(4, 15, 64)
        prob = torch.softmax(torch.randn(15, 5), dim=1)
        
        loss = compute_cluster_loss(prob, x)
        
        assert loss.ndim == 0
    
    def test_loss_gradable(self):
        """Loss should allow gradient computation."""
        x = torch.randn(4, 15, 64)
        prob = torch.softmax(torch.randn(15, 5, requires_grad=True), dim=1)
        
        loss = compute_cluster_loss(prob, x)
        loss.backward()
        
        # No error means gradients flow


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_forward_pass(self):
        """Test complete forward pass through CCM adapter."""
        # Simulate a mini model that expects 7 channels
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # x: (B, 7, L) -> (B, 7, L//2) prediction
                return x[:, :, ::2]  # Simple downsampling
        
        model = DummyModel()
        adapter = CCMAdapter(n_original_vars=21, n_clusters=7, seq_len=64)
        
        # Input data with 21 channels
        x = torch.randn(2, 21, 64)
        
        # Aggregate: 21 -> 7
        x_clustered = adapter.aggregate(x)
        assert x_clustered.shape == (2, 7, 64)
        
        # Model forward (predicts 7-channel output)
        pred_clustered = model(x_clustered)
        assert pred_clustered.shape == (2, 7, 32)
        
        # Expand: 7 -> 21
        pred = adapter.expand(pred_clustered)
        assert pred.shape == (2, 21, 32)
    
    def test_training_step_with_loss(self):
        """Simulate a training step with CCM cluster loss."""
        adapter = CCMAdapter(n_original_vars=15, n_clusters=5, seq_len=32, beta=0.3)
        
        x = torch.randn(2, 15, 32, requires_grad=False)
        target = torch.randn(2, 15, 16)
        
        # Aggregate
        x_agg = adapter.aggregate(x)
        
        # Dummy prediction
        pred_agg = x_agg[:, :, ::2]  # Downsample
        
        # Expand
        pred = adapter.expand(pred_agg)
        
        # Compute losses
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        cluster_loss = adapter.get_cluster_loss(x)
        
        total_loss = mse_loss + cluster_loss
        
        # Should be differentiable
        total_loss.backward()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])





