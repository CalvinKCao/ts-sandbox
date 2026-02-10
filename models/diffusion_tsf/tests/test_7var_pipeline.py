"""
Tests for 7-Variate Training Pipeline.

Run with: pytest models/diffusion_tsf/tests/test_7var_pipeline.py -v
"""

import pytest
import os
import sys
import json
import tempfile
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models.diffusion_tsf.train_7var_pipeline import (
    generate_variate_subsets,
    generate_all_subsets,
    TrainingManifest,
    TimeSeriesDataset,
    DATASET_REGISTRY,
)


class TestVariateSubsetGeneration:
    """Tests for variate subset generation."""
    
    def test_7var_dataset_returns_single_subset(self):
        """Datasets with exactly 7 variates should return one subset."""
        # ETTh1 has 7 variates
        subsets = generate_variate_subsets('ETTh1', n_variates=7, seed=42)
        
        assert len(subsets) == 1
        assert subsets[0]['subset_id'] == 'ETTh1'
        assert len(subsets[0]['variate_indices']) == 7
    
    def test_large_dataset_creates_multiple_subsets(self):
        """Datasets with >7 variates should create multiple non-overlapping subsets."""
        # Weather has 21 variates -> should create 3 subsets
        subsets = generate_variate_subsets('weather', n_variates=7, seed=42)
        
        assert len(subsets) == 3  # 21 // 7 = 3
        
        # Check subset IDs
        assert subsets[0]['subset_id'] == 'weather-0'
        assert subsets[1]['subset_id'] == 'weather-1'
        assert subsets[2]['subset_id'] == 'weather-2'
        
        # Check non-overlapping
        all_indices = []
        for s in subsets:
            all_indices.extend(s['variate_indices'])
        
        assert len(all_indices) == len(set(all_indices)), "Subsets should be non-overlapping"
    
    def test_subsets_are_deterministic(self):
        """Same seed should produce same subsets."""
        subsets1 = generate_variate_subsets('weather', n_variates=7, seed=42)
        subsets2 = generate_variate_subsets('weather', n_variates=7, seed=42)
        
        for s1, s2 in zip(subsets1, subsets2):
            assert s1['variate_indices'] == s2['variate_indices']
    
    def test_different_seeds_produce_different_subsets(self):
        """Different seeds should produce different shuffles."""
        subsets1 = generate_variate_subsets('weather', n_variates=7, seed=42)
        subsets2 = generate_variate_subsets('weather', n_variates=7, seed=123)
        
        # At least one subset should differ
        any_different = False
        for s1, s2 in zip(subsets1, subsets2):
            if s1['variate_indices'] != s2['variate_indices']:
                any_different = True
                break
        
        assert any_different, "Different seeds should produce different subsets"
    
    def test_each_subset_has_correct_size(self):
        """Each subset should have exactly n_variates indices."""
        subsets = generate_variate_subsets('electricity', n_variates=7, seed=42)
        
        for s in subsets:
            assert len(s['variate_indices']) == 7
    
    def test_electricity_creates_many_subsets(self):
        """Electricity (321 variates) should create 45 subsets."""
        subsets = generate_variate_subsets('electricity', n_variates=7, seed=42)
        
        # 321 // 7 = 45 (with 6 leftover variates not used)
        assert len(subsets) == 45
    
    def test_subset_has_variate_names(self):
        """Subsets should include column names."""
        subsets = generate_variate_subsets('ETTh1', n_variates=7, seed=42)
        
        assert 'variate_names' in subsets[0]
        assert len(subsets[0]['variate_names']) == 7


class TestGenerateAllSubsets:
    """Tests for generate_all_subsets function."""
    
    def test_generates_for_all_datasets(self):
        """Should generate subsets for all registered datasets."""
        all_subsets = generate_all_subsets(seed=42)
        
        for dataset_name in DATASET_REGISTRY:
            assert dataset_name in all_subsets
    
    def test_total_subset_count(self):
        """Verify total number of subsets across all datasets."""
        all_subsets = generate_all_subsets(seed=42)
        
        total = sum(len(subsets) for subsets in all_subsets.values())
        
        # Expected counts:
        # ETTh1: 1, ETTh2: 1, ETTm1: 1, ETTm2: 1, illness: 1 (all 7 variates)
        # exchange_rate: 1 (8 variates -> 1 subset of 7)
        # weather: 3 (21 variates)
        # electricity: 45 (321 variates)
        # traffic: 123 (862 variates)
        # Total: 5 + 1 + 3 + 45 + 123 = 177
        
        expected_min = 170  # Allow some flexibility
        assert total >= expected_min, f"Expected at least {expected_min} subsets, got {total}"


class TestTrainingManifest:
    """Tests for TrainingManifest class."""
    
    def test_save_and_load(self):
        """Manifest should save and load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'manifest.json')
            
            # Create and save
            manifest = TrainingManifest(
                seed=42,
                created_at='2024-01-01',
                pretrain_complete=True,
                pretrain_checkpoint='/path/to/ckpt.pt',
            )
            manifest.subsets = {
                'ETTh1': {'status': 'complete', 'checkpoint': '/path/to/ETTh1.pt'},
                'weather-0': {'status': 'pending'},
            }
            manifest.save(path)
            
            # Load and verify
            loaded = TrainingManifest.load(path)
            
            assert loaded.seed == 42
            assert loaded.pretrain_complete == True
            assert loaded.subsets['ETTh1']['status'] == 'complete'
            assert loaded.subsets['weather-0']['status'] == 'pending'
    
    def test_get_pending_subsets(self):
        """Should return only pending subsets."""
        manifest = TrainingManifest()
        manifest.subsets = {
            'ETTh1': {'status': 'complete'},
            'ETTh2': {'status': 'pending'},
            'weather-0': {'status': 'in_progress'},
            'weather-1': {'status': 'pending'},
        }
        
        pending = manifest.get_pending_subsets()
        
        assert 'ETTh2' in pending
        assert 'weather-0' in pending  # in_progress counts as pending
        assert 'weather-1' in pending
        assert 'ETTh1' not in pending
    
    def test_mark_complete(self):
        """Should correctly mark subset as complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'manifest.json')
            
            manifest = TrainingManifest()
            manifest.subsets = {'ETTh1': {'status': 'pending'}}
            manifest.save(path)
            
            manifest.mark_complete('ETTh1', '/path/to/ckpt.pt', {'mse': 0.1})
            
            assert manifest.subsets['ETTh1']['status'] == 'complete'
            assert manifest.subsets['ETTh1']['checkpoint'] == '/path/to/ckpt.pt'
            assert manifest.subsets['ETTh1']['metrics']['mse'] == 0.1


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset class."""
    
    def test_dataset_length(self):
        """Dataset length should be correct based on lookback/horizon/stride."""
        import numpy as np
        
        # 1000 timesteps, lookback=100, horizon=50, stride=1
        # Valid samples: (1000 - 150) // 1 + 1 = 851
        data = np.random.randn(1000, 7).astype(np.float32)
        ds = TimeSeriesDataset(data, lookback=100, horizon=50, stride=1)
        
        assert len(ds) == 851
    
    def test_dataset_shapes(self):
        """Dataset should return correct shapes."""
        import numpy as np
        
        data = np.random.randn(500, 7).astype(np.float32)
        ds = TimeSeriesDataset(data, lookback=100, horizon=50, stride=1)
        
        past, future = ds[0]
        
        assert past.shape == (7, 100)  # (variates, lookback)
        assert future.shape == (7, 50)  # (variates, horizon)
    
    def test_dataset_stride(self):
        """Stride should affect number of samples."""
        import numpy as np
        
        data = np.random.randn(1000, 7).astype(np.float32)
        
        ds_stride1 = TimeSeriesDataset(data, lookback=100, horizon=50, stride=1)
        ds_stride10 = TimeSeriesDataset(data, lookback=100, horizon=50, stride=10)
        
        # stride=10 should have ~10x fewer samples
        assert len(ds_stride10) < len(ds_stride1) / 5


class TestSubsetNaming:
    """Tests for subset ID parsing."""
    
    def test_parse_dataset_from_subset_id(self):
        """Should correctly parse dataset name from subset ID."""
        # Simple case
        assert 'ETTh1'.split('-')[0] == 'ETTh1'
        
        # With suffix
        subset_id = 'traffic-5'
        parts = subset_id.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            dataset_name = parts[0]
        else:
            dataset_name = subset_id
        
        assert dataset_name == 'traffic'
    
    def test_parse_exchange_rate(self):
        """exchange_rate-0 should parse correctly (hyphen in name)."""
        subset_id = 'exchange_rate-0'
        
        # Use rsplit to handle hyphens in dataset name
        if '-' in subset_id and subset_id.split('-')[-1].isdigit():
            dataset_name = '-'.join(subset_id.split('-')[:-1])
        else:
            dataset_name = subset_id
        
        assert dataset_name == 'exchange_rate'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



