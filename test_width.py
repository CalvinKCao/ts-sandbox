import torch
import sys

from models.diffusion_tsf.experimental_diffusion_model import ExperimentalDiffusionTSFConfig

config = ExperimentalDiffusionTSFConfig(
    num_variables=7,
    lookback_length=512,
    forecast_length=96 + 8, # 8 overlap
    lookback_overlap=8,
    image_height=32,
    use_residual_diffusion=False,
    independent_norm=False,
    num_diffusion_steps=1000,
    model_type="unet"
)

width = config.forecast_length
print("width type:", type(width), width)

try:
    torch.linspace(-1.0, 1.0, width, device='cpu', dtype=torch.float32)
except Exception as e:
    print(repr(e))
