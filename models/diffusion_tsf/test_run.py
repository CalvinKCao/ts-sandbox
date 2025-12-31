"""
Test script for Diffusion-based Time Series Forecasting.

This script verifies that:
1. Forward pass works correctly
2. Backward pass computes gradients
3. Generation pipeline produces valid outputs
4. All components integrate properly
"""

import torch
import torch.nn as nn
import logging
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Handle imports for both module and script execution
import os
import sys

# Add parent directory to path for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now import our modules (using non-relative imports)
from config import DiffusionTSFConfig
from preprocessing import TimeSeriesTo2D, VerticalGaussianBlur, Standardizer
from unet import ConditionalUNet2D, get_timestep_embedding
from diffusion import DiffusionScheduler
from dataset import create_toy_batch, SyntheticTimeSeriesDataset
from metrics import compute_metrics, log_metrics
from model import DiffusionTSF


def test_preprocessing():
    """Test the preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Testing Preprocessing Pipeline")
    logger.info("=" * 60)
    
    batch_size = 4
    seq_len = 96
    height = 128
    max_scale = 3.5
    
    # Create test data
    x = torch.randn(batch_size, seq_len)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    
    # Test Standardizer
    standardizer = Standardizer()
    x_norm = standardizer.fit_transform(x)
    logger.info(f"After normalization - mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")
    
    # Test 2D encoding
    to_2d = TimeSeriesTo2D(height=height, max_scale=max_scale)
    image = to_2d(x_norm)
    logger.info(f"2D image shape: {image.shape}")
    logger.info(f"Image sum per column (should be 1): {image.sum(dim=2).mean():.4f}")
    
    # Verify one-hot property
    assert image.sum(dim=2).allclose(torch.ones(batch_size, 1, seq_len)), "One-hot property violated!"
    logger.info("[OK] One-hot property verified")
    
    # Test vertical Gaussian blur
    blur = VerticalGaussianBlur(kernel_size=31, sigma=6.0)
    blurred = blur(image)
    logger.info(f"Blurred image shape: {blurred.shape}")
    logger.info(f"Blurred image range: [{blurred.min():.4f}, {blurred.max():.4f}]")
    
    # Verify blur only affects height dimension
    # Each column should still sum to ~1 (slight difference due to edge effects)
    col_sums = blurred.sum(dim=2).mean()
    logger.info(f"Column sums after blur (should be ~1): {col_sums:.4f}")
    
    # Test inverse mapping
    x_reconstructed = to_2d.inverse(blurred)
    logger.info(f"Reconstructed shape: {x_reconstructed.shape}")
    
    reconstruction_error = (x_norm - x_reconstructed).abs().mean()
    logger.info(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # Test denormalization
    x_denorm = standardizer.inverse_transform(x_reconstructed)
    final_error = (x - x_denorm).abs().mean()
    logger.info(f"Final error (original vs reconstructed): {final_error:.4f}")
    
    logger.info("[OK] Preprocessing tests passed!")
    return True


def test_unet():
    """Test the U-Net architecture."""
    logger.info("=" * 60)
    logger.info("Testing U-Net Architecture")
    logger.info("=" * 60)
    
    batch_size = 2
    height = 128
    future_len = 96
    past_len = 512
    
    # Create model
    unet = ConditionalUNet2D(
        in_channels=1,
        out_channels=1,
        channels=[32, 64, 128],  # Smaller for testing
        num_res_blocks=1,
        attention_levels=[1, 2]
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"U-Net parameters: {num_params:,}")
    
    # Create test inputs
    x = torch.randn(batch_size, 1, height, future_len)  # Noisy future
    t = torch.randint(0, 1000, (batch_size,))  # Timesteps
    cond = torch.randn(batch_size, 1, height, past_len)  # Past context
    
    logger.info(f"Input x shape: {x.shape}")
    logger.info(f"Timesteps t: {t.tolist()}")
    logger.info(f"Condition shape: {cond.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = unet(x, t, cond)
    
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    logger.info("[OK] Output shape correct")
    
    # Test gradient flow
    unet.train()
    output = unet(x, t, cond)
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in unet.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    logger.info(f"Gradient norms - min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}, mean: {sum(grad_norms)/len(grad_norms):.6f}")
    logger.info("[OK] Gradients computed successfully")
    
    return True


def test_diffusion_scheduler():
    """Test the diffusion scheduler."""
    logger.info("=" * 60)
    logger.info("Testing Diffusion Scheduler")
    logger.info("=" * 60)
    
    scheduler = DiffusionScheduler(
        num_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule="linear"
    )
    
    logger.info(f"Betas range: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    logger.info(f"Alpha_bar range: [{scheduler.alphas_cumprod[-1]:.6f}, {scheduler.alphas_cumprod[0]:.6f}]")
    
    # Test forward process
    batch_size = 4
    x_0 = torch.randn(batch_size, 1, 128, 96)
    t = torch.randint(0, 1000, (batch_size,))
    
    x_t, noise = scheduler.add_noise(x_0, t)
    logger.info(f"Forward process: x_0 {x_0.shape} -> x_t {x_t.shape}")
    
    # Verify noise was added
    assert not torch.allclose(x_0, x_t), "Noise not added!"
    logger.info("[OK] Forward process works")
    
    # Test x_0 prediction from noise
    x_0_pred = scheduler.predict_x0_from_noise(x_t, t, noise)
    prediction_error = (x_0 - x_0_pred).abs().mean()
    logger.info(f"x_0 prediction error (should be ~0): {prediction_error:.6f}")
    assert prediction_error < 1e-5, "x_0 prediction error too large!"
    logger.info("[OK] x_0 prediction works")
    
    return True


def test_full_model():
    """Test the complete DiffusionTSF model."""
    logger.info("=" * 60)
    logger.info("Testing Full DiffusionTSF Model")
    logger.info("=" * 60)
    
    # Use smaller config for testing
    config = DiffusionTSFConfig(
        lookback_length=128,  # Smaller for faster testing
        forecast_length=32,
        image_height=64,  # Smaller
        unet_channels=[32, 64, 128],  # Smaller
        num_res_blocks=1,
        attention_levels=[1, 2],
        num_diffusion_steps=100,  # Fewer steps for testing
        ddim_steps=10  # Very few DDIM steps for testing
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create model
    model = DiffusionTSF(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create test data
    past, future = create_toy_batch(
        batch_size=2,
        lookback_length=config.lookback_length,
        forecast_length=config.forecast_length,
        device=device
    )
    
    logger.info(f"Past shape: {past.shape}")
    logger.info(f"Future shape: {future.shape}")
    
    # Test forward pass (training mode)
    logger.info("-" * 40)
    logger.info("Testing FORWARD PASS...")
    
    model.train()
    outputs = model(past, future)
    
    logger.info(f"Loss: {outputs['loss'].item():.6f}")
    logger.info(f"Noise shape: {outputs['noise'].shape}")
    logger.info(f"Noise pred shape: {outputs['noise_pred'].shape}")
    logger.info(f"Past 2D shape: {outputs['past_2d'].shape}")
    logger.info(f"Future 2D shape: {outputs['future_2d'].shape}")
    
    # Test backward pass
    logger.info("-" * 40)
    logger.info("Testing BACKWARD PASS...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    outputs['loss'].backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    logger.info(f"Number of parameters with gradients: {len(grad_norms)}")
    logger.info(f"Gradient norms - min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}")
    
    # Take an optimization step
    optimizer.step()
    logger.info("[OK] Backward pass and optimization step completed")
    
    # Test generation (inference mode)
    logger.info("-" * 40)
    logger.info("Testing GENERATION (DDIM)...")
    
    model.eval()
    start_time = time.time()
    
    gen_outputs = model.generate(
        past,
        use_ddim=True,
        num_ddim_steps=config.ddim_steps,
        eta=0.0,
        verbose=True
    )
    
    gen_time = time.time() - start_time
    logger.info(f"Generation time: {gen_time:.2f}s")
    logger.info(f"Prediction shape: {gen_outputs['prediction'].shape}")
    logger.info(f"Prediction range: [{gen_outputs['prediction'].min():.2f}, {gen_outputs['prediction'].max():.2f}]")
    
    # Compute metrics
    logger.info("-" * 40)
    logger.info("Computing METRICS...")
    
    metrics = compute_metrics(gen_outputs['prediction'], future)
    logger.info(log_metrics(metrics))
    
    logger.info("[OK] Full model test passed!")
    return True


def test_training_loop():
    """Test a mini training loop."""
    logger.info("=" * 60)
    logger.info("Testing Mini Training Loop")
    logger.info("=" * 60)
    
    # Tiny config for fast testing
    config = DiffusionTSFConfig(
        lookback_length=64,
        forecast_length=16,
        image_height=32,
        unet_channels=[16, 32],
        num_res_blocks=1,
        attention_levels=[1],
        num_diffusion_steps=50,
        ddim_steps=5
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and optimizer
    model = DiffusionTSF(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create small dataset
    dataset = SyntheticTimeSeriesDataset(
        num_samples=8,
        lookback_length=config.lookback_length,
        forecast_length=config.forecast_length
    )
    
    # Training loop
    num_epochs = 3
    model.train()
    
    logger.info(f"Training for {num_epochs} epochs on {len(dataset)} samples...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for i in range(len(dataset)):
            past, future = dataset[i]
            past = past.unsqueeze(0).to(device)
            future = future.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            outputs = model(past, future)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataset)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")
    
    # Final generation test
    model.eval()
    past, future = dataset[0]
    past = past.unsqueeze(0).to(device)
    future = future.unsqueeze(0).to(device)
    
    gen_outputs = model.generate(past, use_ddim=True, num_ddim_steps=5)
    
    metrics = compute_metrics(gen_outputs['prediction'], future)
    logger.info(f"Final metrics: {log_metrics(metrics)}")
    
    logger.info("[OK] Training loop test passed!")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("DIFFUSION TSF - TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("U-Net", test_unet),
        ("Diffusion Scheduler", test_diffusion_scheduler),
        ("Full Model", test_full_model),
        ("Training Loop", test_training_loop),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            logger.error(f"Test '{name}' FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for name, success, error in results:
        status = "PASSED" if success else f"FAILED: {error}"
        logger.info(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("SOME TESTS FAILED")
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

