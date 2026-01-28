
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.augmentation import generate_multivariate_synthetic_data
from models.diffusion_tsf import realts

class TestOrganicMultivariate(unittest.TestCase):
    
    def test_shape_and_values(self):
        """Sanity check for output shape and finite values."""
        samples = 2
        vars = 3
        length = 100
        data = generate_multivariate_synthetic_data(samples, vars, length)
        
        self.assertEqual(data.shape, (samples, vars, length))
        self.assertTrue(np.all(np.isfinite(data)))
        
    @patch('models.diffusion_tsf.realts.RWB')
    @patch('models.diffusion_tsf.realts.IFFTB')
    @patch('models.diffusion_tsf.realts.seasonal_periodicity')
    def test_uses_organic_generators(self, mock_seasonal, mock_ifftb, mock_rwb):
        """
        Test that the multivariate generator actually uses the organic
        generators from realts.py instead of the hardcoded step/pulse logic.
        """
        # Configure mocks to return valid data so the pipeline doesn't crash
        def side_effect(length):
            return np.random.randn(length)
            
        mock_rwb.side_effect = side_effect
        mock_ifftb.side_effect = side_effect
        mock_seasonal.side_effect = side_effect
        
        # We need to ensure that specific generators are picked.
        # Since the selection is random, we might need to patch the random choice 
        # or run enough samples to ensure coverage.
        # Alternatively, we just check that *at least one* of the organic mocks was called.
        
        generate_multivariate_synthetic_data(num_samples=5, num_vars=3, length=50)
        
        # Assert that at least one of the organic generators was called
        any_called = (mock_rwb.called or mock_ifftb.called or mock_seasonal.called)
        self.assertTrue(any_called, "Failed to use any organic generators (RWB, IFFTB, Seasonal)")

if __name__ == '__main__':
    unittest.main()
