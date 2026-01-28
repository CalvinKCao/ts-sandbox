import pytest
import torch
import torch.nn as nn
from models.diffusion_tsf.unet import ConditionalUNet2D, TimeSeriesContextEncoder
from models.diffusion_tsf.transformer import DiffusionTransformer
from models.diffusion_tsf.diffusion import DiffusionScheduler
from models.diffusion_tsf.guidance import LastValueGuidance, LinearRegressionGuidance, iTransformerGuidance

class TestArchitecture:
    def test_unet_shapes(self):
        batch, in_ch, height, width = 2, 3, 32, 16
        out_ch = 1
        model = ConditionalUNet2D(
            in_channels=in_ch,
            out_channels=out_ch,
            channels=[8, 16],
            num_res_blocks=1,
            attention_levels=[1],
            image_height=height,
            conditioning_mode="visual_concat",
            visual_cond_channels=1
        )
        
        x = torch.randn(batch, in_ch, height, width)
        t = torch.randint(0, 10, (batch,))
        # visual_concat: cond is (batch, cond_ch, height, width)
        cond = torch.randn(batch, 1, height, width)
        
        out = model(x, t, cond)
        assert out.shape == (batch, out_ch, height, width)

    def test_unet_hybrid_shapes(self):
        batch, in_ch, height, width = 2, 3, 32, 16
        out_ch = 1
        model = ConditionalUNet2D(
            in_channels=in_ch,
            out_channels=out_ch,
            channels=[8, 16],
            num_res_blocks=1,
            use_hybrid_condition=True,
            context_dim=16,
            image_height=height
        )
        
        x = torch.randn(batch, in_ch, height, width)
        t = torch.randint(0, 10, (batch,))
        cond = torch.randn(batch, 1, height, width)
        # Context: (batch, seq, dim)
        context = torch.randn(batch, 10, 16)
        
        out = model(x, t, cond, encoder_hidden_states=context)
        assert out.shape == (batch, out_ch, height, width)

    def test_transformer_shapes(self):
        batch, in_ch, height, width = 2, 3, 32, 16
        # height must be div by patch size (16)
        model = DiffusionTransformer(
            image_height=height,
            patch_height=16,
            patch_width=16,
            embed_dim=16,
            depth=1,
            num_heads=4,
            in_channels=in_ch,
            out_channels=1
        )
        
        x = torch.randn(batch, in_ch, height, width)
        t = torch.randint(0, 10, (batch,))
        cond = torch.randn(batch, in_ch, height, width) # Transformer uses same channels for cond usually
        
        out = model(x, t, cond)
        assert out.shape == (batch, 1, height, width)

    def test_context_encoder(self):
        encoder = TimeSeriesContextEncoder(
            input_channels=2,
            embedding_dim=16,
            num_layers=1
        )
        x = torch.randn(2, 50, 2) # batch, seq, channels
        out = encoder(x)
        assert out.shape == (2, 50, 16)


class TestDiffusion:
    def test_schedule_creation(self, device):
        scheduler = DiffusionScheduler(num_steps=100, schedule="linear", device=device)
        assert len(scheduler.betas) == 100
        assert scheduler.alphas_cumprod[-1] < scheduler.alphas_cumprod[0]

    def test_add_noise(self, device):
        scheduler = DiffusionScheduler(num_steps=100, device=device)
        x0 = torch.zeros(4, 1, 32, 32, device=device)
        t = torch.full((4,), 50, device=device, dtype=torch.long)
        
        xt, noise = scheduler.add_noise(x0, t)
        
        assert xt.shape == x0.shape
        assert noise.shape == x0.shape
        # With x0=0, xt should be proportional to noise
        # xt = sqrt(1-alpha_bar) * noise
        coeff = scheduler.sqrt_one_minus_alphas_cumprod[50]
        assert torch.allclose(xt, coeff * noise)

    def test_step_functions(self, device):
        scheduler = DiffusionScheduler(num_steps=100, device=device)
        xt = torch.randn(2, 1, 16, 16, device=device)
        t = torch.tensor([50, 50], device=device)
        noise_pred = torch.randn_like(xt)
        
        # DDPM
        prev_ddpm = scheduler.ddpm_step(xt, t, noise_pred)
        assert prev_ddpm.shape == xt.shape
        
        # DDIM
        t_prev = torch.tensor([40, 40], device=device)
        prev_ddim = scheduler.ddim_step(xt, t, t_prev, noise_pred)
        assert prev_ddim.shape == xt.shape


class MockITransformer(nn.Module):
    def __init__(self, pred_len, seq_len, num_vars):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.num_vars = num_vars
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (batch, seq_len, num_vars)
        # Returns: (batch, pred_len, num_vars)
        batch = x_enc.shape[0]
        return torch.randn(batch, self.pred_len, self.num_vars)


class TestGuidance:
    def test_last_value(self):
        guidance = LastValueGuidance()
        # Univariate: (batch, past_len)
        past = torch.arange(10).float().unsqueeze(0) # 0..9
        forecast = guidance.get_forecast(past, forecast_length=5)
        assert forecast.shape == (1, 5)
        assert torch.all(forecast == 9.0)
        
        # Multivariate: (batch, vars, past_len)
        past_multi = torch.randn(2, 3, 10)
        forecast_multi = guidance.get_forecast(past_multi, forecast_length=5)
        assert forecast_multi.shape == (2, 3, 5)

    def test_linear_regression(self):
        guidance = LinearRegressionGuidance()
        # y = 2t + 1
        t = torch.arange(10).float()
        past = 2 * t + 1
        past = past.unsqueeze(0) # batch=1
        
        # Predict next 5
        forecast = guidance.get_forecast(past, forecast_length=5)
        
        t_future = torch.arange(10, 15).float()
        expected = 2 * t_future + 1
        
        assert torch.allclose(forecast, expected.unsqueeze(0), atol=1e-5)

    def test_itransformer_wrapper(self):
        mock_model = MockITransformer(pred_len=10, seq_len=20, num_vars=2)
        guidance = iTransformerGuidance(
            model=mock_model,
            seq_len=20,
            pred_len=10
        )
        
        # Input to wrapper: (batch, num_vars, seq_len)
        past = torch.randn(3, 2, 20)
        out = guidance.get_forecast(past, forecast_length=10)
        
        # Output: (batch, num_vars, pred_len)
        assert out.shape == (3, 2, 10)
