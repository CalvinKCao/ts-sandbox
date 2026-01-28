import pytest
import torch
import torch.nn as nn
from models.diffusion_tsf.dataset import apply_1d_augmentations, get_synthetic_dataloader
from models.diffusion_tsf.model import DiffusionTSF
from models.diffusion_tsf.config import DiffusionTSFConfig

class TestDataset:
    def test_augmentations_shape(self):
        seq = torch.randn(100)
        # Apply all augmentations
        out = apply_1d_augmentations(
            seq, 
            scale_prob=1.0, 
            warp_prob=1.0, 
            stretch_prob=1.0
        )
        assert out.shape == seq.shape

    def test_synthetic_dataloader(self):
        # Smoke test for dataloader creation
        dl = get_synthetic_dataloader(
            num_samples=10,
            lookback_length=32,
            forecast_length=16,
            batch_size=2
        )
        batch = next(iter(dl))
        past, future = batch
        assert past.shape == (2, 32)
        assert future.shape == (2, 16)


class TestDiffusionTSFIntegration:
    @pytest.fixture
    def univariate_config(self):
        return DiffusionTSFConfig(
            lookback_length=32,
            forecast_length=16,
            image_height=32,
            unet_channels=[8, 16],
            num_res_blocks=1,
            attention_levels=[],
            num_diffusion_steps=10,
            num_variables=1,
            conditioning_mode="visual_concat",
            use_coordinate_channel=True,
            use_time_ramp=True
        )

    @pytest.fixture
    def multivariate_config(self):
        return DiffusionTSFConfig(
            lookback_length=32,
            forecast_length=16,
            image_height=32,
            unet_channels=[8, 16],
            num_res_blocks=1,
            attention_levels=[],
            num_diffusion_steps=10,
            num_variables=3,
            conditioning_mode="vector_embedding",
            use_coordinate_channel=True,
            use_guidance_channel=True # Requires guidance model
        )

    def test_init(self, univariate_config):
        model = DiffusionTSF(univariate_config)
        assert isinstance(model.noise_predictor, nn.Module)
        assert model.config == univariate_config

    def test_forward_loss(self, univariate_config):
        model = DiffusionTSF(univariate_config)
        batch_size = 2
        past = torch.randn(batch_size, univariate_config.lookback_length)
        future = torch.randn(batch_size, univariate_config.forecast_length)
        
        # Training step
        model.train()
        outputs = model(past, future)
        
        assert 'loss' in outputs
        assert 'noise_loss' in outputs
        assert outputs['loss'].item() > 0

    def test_generate_univariate(self, univariate_config):
        model = DiffusionTSF(univariate_config)
        model.eval()
        batch_size = 2
        past = torch.randn(batch_size, univariate_config.lookback_length)
        
        # Inference
        with torch.no_grad():
            outputs = model.generate(
                past, 
                use_ddim=True, 
                num_ddim_steps=5
            )
            
        assert 'prediction' in outputs
        assert outputs['prediction'].shape == (batch_size, univariate_config.forecast_length)
        assert 'future_2d' in outputs

    def test_multivariate_guidance_flow(self, multivariate_config):
        # Initialize with guidance channel
        model = DiffusionTSF(multivariate_config) # Defaults to LinearRegression guidance
        
        batch_size = 2
        num_vars = 3
        past = torch.randn(batch_size, num_vars, multivariate_config.lookback_length)
        future = torch.randn(batch_size, num_vars, multivariate_config.forecast_length)
        
        # Forward
        outputs = model(past, future)
        assert 'loss' in outputs
        # Check if guidance_2d was generated and returned
        assert 'guidance_2d' in outputs
        assert outputs['guidance_2d'].shape == (
            batch_size, num_vars, multivariate_config.image_height, multivariate_config.forecast_length
        )

    def test_hybrid_conditioning_flow(self):
        config = DiffusionTSFConfig(
            lookback_length=32,
            forecast_length=16,
            image_height=32,
            unet_channels=[8, 16],
            use_hybrid_condition=True,
            context_encoder_layers=1,
            conditioning_mode="visual_concat"
        )
        model = DiffusionTSF(config)
        
        batch_size = 2
        past = torch.randn(batch_size, config.lookback_length)
        future = torch.randn(batch_size, config.forecast_length)
        
        # Should run without error and use context encoder
        outputs = model(past, future)
        assert 'loss' in outputs

    def test_value_channel_injection(self):
        config = DiffusionTSFConfig(
            lookback_length=32,
            forecast_length=16,
            image_height=32,
            unet_channels=[8],
            use_value_channel=True # Test this specific channel
        )
        model = DiffusionTSF(config)
        
        batch_size = 2
        past = torch.randn(batch_size, config.lookback_length)
        future = torch.randn(batch_size, config.forecast_length)
        
        # Should run without error
        outputs = model(past, future)
        assert 'loss' in outputs
