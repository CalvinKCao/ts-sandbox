#!/usr/bin/env python
"""Test script to verify both conditioning modes work correctly."""

import torch
import sys

# Handle imports for both module and script execution
try:
    from .config import DiffusionTSFConfig
    from .model import DiffusionTSF
except ImportError:
    from config import DiffusionTSFConfig
    from model import DiffusionTSF


def test_visual_concat_mode():
    """Test the new visual_concat conditioning mode."""
    print("=" * 60)
    print("Test 1: Visual Concat Mode (new default)")
    print("=" * 60)
    
    config = DiffusionTSFConfig(
        lookback_length=64,
        forecast_length=24,
        image_height=64,
        unet_channels=[32, 64],
        num_res_blocks=1,
        attention_levels=[],
        conditioning_mode="visual_concat",
        use_hybrid_condition=False,
        num_diffusion_steps=100,
    )
    
    model = DiffusionTSF(config)
    print(f"Model created with conditioning_mode={config.conditioning_mode}")
    print(f"Visual cond channels: {config.visual_cond_channels}")
    
    # Test forward pass
    batch_size = 2
    past = torch.randn(batch_size, config.lookback_length)
    future = torch.randn(batch_size, config.forecast_length)
    
    outputs = model(past, future)
    print(f"Forward pass successful! Loss: {outputs['loss'].item():.4f}")
    
    # Test generate
    with torch.no_grad():
        gen_outputs = model.generate(past, use_ddim=True, num_ddim_steps=5, cfg_scale=1.0)
    print(f"Generation successful! Prediction shape: {gen_outputs['prediction'].shape}")
    
    return True


def test_vector_embedding_mode():
    """Test the old vector_embedding conditioning mode."""
    print()
    print("=" * 60)
    print("Test 2: Vector Embedding Mode (backward compatible)")
    print("=" * 60)
    
    config = DiffusionTSFConfig(
        lookback_length=64,
        forecast_length=24,
        image_height=64,
        unet_channels=[32, 64],
        num_res_blocks=1,
        attention_levels=[],
        conditioning_mode="vector_embedding",
        use_hybrid_condition=False,
        num_diffusion_steps=100,
    )
    
    model = DiffusionTSF(config)
    print(f"Model created with conditioning_mode={config.conditioning_mode}")
    
    # Test forward pass
    batch_size = 2
    past = torch.randn(batch_size, config.lookback_length)
    future = torch.randn(batch_size, config.forecast_length)
    
    outputs = model(past, future)
    print(f"Forward pass successful! Loss: {outputs['loss'].item():.4f}")
    
    # Test generate
    with torch.no_grad():
        gen_outputs = model.generate(past, use_ddim=True, num_ddim_steps=5, cfg_scale=1.0)
    print(f"Generation successful! Prediction shape: {gen_outputs['prediction'].shape}")
    
    return True


def test_visual_concat_with_hybrid():
    """Test visual_concat with hybrid 1D conditioning enabled."""
    print()
    print("=" * 60)
    print("Test 3: Visual Concat + Hybrid 1D Conditioning")
    print("=" * 60)
    
    config = DiffusionTSFConfig(
        lookback_length=64,
        forecast_length=24,
        image_height=64,
        unet_channels=[32, 64],
        num_res_blocks=1,
        attention_levels=[0],  # Need attention for cross-attention
        conditioning_mode="visual_concat",
        use_hybrid_condition=True,
        context_embedding_dim=32,
        context_encoder_layers=1,
        num_diffusion_steps=100,
    )
    
    model = DiffusionTSF(config)
    print(f"Model created with conditioning_mode={config.conditioning_mode}, hybrid={config.use_hybrid_condition}")
    
    # Test forward pass
    batch_size = 2
    past = torch.randn(batch_size, config.lookback_length)
    future = torch.randn(batch_size, config.forecast_length)
    
    outputs = model(past, future)
    print(f"Forward pass successful! Loss: {outputs['loss'].item():.4f}")
    
    # Test generate with CFG
    with torch.no_grad():
        gen_outputs = model.generate(past, use_ddim=True, num_ddim_steps=5, cfg_scale=2.0)
    print(f"Generation with CFG successful! Prediction shape: {gen_outputs['prediction'].shape}")
    
    return True


def test_multivariate():
    """Test visual_concat with multivariate data."""
    print()
    print("=" * 60)
    print("Test 4: Visual Concat - Multivariate")
    print("=" * 60)
    
    config = DiffusionTSFConfig(
        lookback_length=64,
        forecast_length=24,
        image_height=64,
        num_variables=3,
        unet_channels=[32, 64],
        num_res_blocks=1,
        attention_levels=[],
        conditioning_mode="visual_concat",
        use_hybrid_condition=False,
        num_diffusion_steps=100,
    )
    
    model = DiffusionTSF(config)
    print(f"Model created with num_variables={config.num_variables}, visual_cond_channels={config.visual_cond_channels}")
    
    # Test forward pass with multivariate input
    batch_size = 2
    past = torch.randn(batch_size, config.num_variables, config.lookback_length)
    future = torch.randn(batch_size, config.num_variables, config.forecast_length)
    
    outputs = model(past, future)
    print(f"Forward pass successful! Loss: {outputs['loss'].item():.4f}")
    
    # Test generate
    with torch.no_grad():
        gen_outputs = model.generate(past, use_ddim=True, num_ddim_steps=5, cfg_scale=1.0)
    print(f"Generation successful! Prediction shape: {gen_outputs['prediction'].shape}")
    
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_visual_concat_mode()
        success &= test_vector_embedding_mode()
        success &= test_visual_concat_with_hybrid()
        success &= test_multivariate()
        
        print()
        print("=" * 60)
        if success:
            print("ALL TESTS PASSED!")
        else:
            print("SOME TESTS FAILED!")
        print("=" * 60)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

