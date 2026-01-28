import pytest
import torch
import torch.nn.functional as F
from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
from models.diffusion_tsf.metrics import compute_metrics, shape_preservation_score, monotonicity_loss

class TestConfig:
    def test_default_init(self):
        config = DiffusionTSFConfig()
        assert config.lookback_length == 512
        assert config.forecast_length == 96
        assert config.representation_mode == "pdf"
        assert config.backbone_in_channels > 0

    def test_validation(self):
        with pytest.raises(AssertionError):
            DiffusionTSFConfig(image_height=-10)
        with pytest.raises(AssertionError):
            DiffusionTSFConfig(representation_mode="invalid")
            
    def test_calculated_properties(self):
        config = DiffusionTSFConfig(
            num_variables=3,
            use_coordinate_channel=True,
            use_time_ramp=True,
            use_guidance_channel=True
        )
        # aux = coord(1) + ramp(1) = 2
        # backbone_in = num_vars(3) + aux(2) + guidance(3) = 8
        assert config.num_aux_channels == 2
        assert config.guidance_channels == 3
        assert config.backbone_in_channels == 8


class TestPreprocessing:
    @pytest.mark.parametrize("mode", ["pdf", "cdf"])
    def test_timeseries_to_2d_shapes(self, mode, sample_batch):
        batch_size, seq_len = sample_batch.shape
        height = 32
        ts2d = TimeSeriesTo2D(height=height, representation_mode=mode)
        
        # Univariate input: (batch, seq_len)
        out = ts2d(sample_batch)
        # Output: (batch, 1, height, seq_len)
        assert out.shape == (batch_size, 1, height, seq_len)
        
        if mode == "pdf":
            # Check one-hot property along height (sum should be 1.0)
            assert torch.allclose(out.sum(dim=2), torch.ones(batch_size, 1, seq_len))
        else:
            # CDF: values should be monotonic increasing along height (0 to 1)
            # Actually our implementation is occupancy: 1s at bottom, 0s at top.
            # So values are decreasing or equal as y increases? 
            # Implementation: filled = (height_range <= bin_indices).
            # height_range increases with y. bin_indices is constant for a col.
            # So for small y, range < bin -> 1. For large y, range > bin -> 0.
            # So it's 1s then 0s. Monotonically non-increasing.
            diff = out[:, :, 1:, :] - out[:, :, :-1, :]
            # diff should be <= 0
            assert torch.all(diff <= 0)

    def test_timeseries_to_2d_multivariate(self, sample_batch_multivariate):
        batch, vars, seq = sample_batch_multivariate.shape
        height = 32
        ts2d = TimeSeriesTo2D(height=height)
        out = ts2d(sample_batch_multivariate)
        assert out.shape == (batch, vars, height, seq)

    def test_inverse_mapping(self, sample_batch):
        # Round trip test (approximate due to discretization)
        height = 128
        ts2d = TimeSeriesTo2D(height=height, max_scale=3.5)
        
        # 1. Forward
        img = ts2d(sample_batch)
        
        # 2. Inverse
        # Use low temperature to sharpen the distribution for better reconstruction
        recon = ts2d.inverse(img, pdf_temperature=0.01)
        
        # Should be reasonably close
        # Note: Discretization error is at most bin_width/2
        # bin_width = 7.0 / 128 ~= 0.055
        mae = F.l1_loss(recon, sample_batch)
        assert mae < 0.1

    def test_vertical_gaussian_blur(self):
        blur = VerticalGaussianBlur(kernel_size=5, sigma=1.0)
        x = torch.zeros(1, 1, 10, 10)
        x[:, :, 5, 5] = 1.0  # Impulse
        
        out = blur(x)
        assert out.shape == x.shape
        
        # Should be blurred vertically but not horizontally
        col_5 = out[0, 0, :, 5]
        col_4 = out[0, 0, :, 4]
        
        assert col_5.sum() > 0.9  # Normalized kernel
        assert col_4.sum() == 0   # No horizontal bleed


class TestMetrics:
    def test_metrics_computation(self, sample_batch):
        pred = sample_batch
        target = sample_batch + 0.1
        
        metrics = compute_metrics(pred, target)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'gradient_mae' in metrics
        assert 'shape_score' in metrics
        
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0

    def test_monotonicity_loss(self):
        # Create a non-monotonic CDF map (violation)
        # Occupancy should be 1s then 0s (non-increasing).
        # monotonicity_loss checks: pred[:, :, 1:, :] - pred[:, :, :-1, :]
        # If value increases with y, that's a violation?
        # Code: diff = pred[:, :, 1:, :] - pred[:, :, :-1, :] (y+1 - y)
        # If 0 -> 1 (increase), diff > 0. This is a hole in the occupancy. Violation.
        # If 1 -> 0 (decrease), diff < 0. Normal transition.
        
        x = torch.zeros(1, 1, 4, 1)
        x[0, 0, 0, 0] = 0.0
        x[0, 0, 1, 0] = 1.0 # Jump up! Violation.
        x[0, 0, 2, 0] = 0.0
        x[0, 0, 3, 0] = 0.0
        
        loss = monotonicity_loss(x)
        assert loss > 0
        
        # Perfect occupancy
        y = torch.zeros(1, 1, 4, 1)
        y[0, 0, 0, 0] = 1.0
        y[0, 0, 1, 0] = 1.0
        y[0, 0, 2, 0] = 0.0
        y[0, 0, 3, 0] = 0.0
        
        loss_good = monotonicity_loss(y)
        assert loss_good == 0
