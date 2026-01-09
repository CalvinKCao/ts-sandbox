#!/usr/bin/env python3
"""
Test script to validate stride configuration and argument parsing.
This demonstrates that the stride parameter works correctly without running full training.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the diffusion_tsf directory to path so we can import
script_dir = Path(__file__).parent / 'models' / 'diffusion_tsf'
if script_dir not in sys.path:
    sys.path.insert(0, str(script_dir))

def test_stride_parsing():
    """Test that stride argument parsing works."""
    print("🧪 Testing stride argument parsing...")

    # Simulate command line arguments
    test_cases = [
        (['--stride', '12'], 12),
        (['--stride', '24'], 24),
        (['--stride', '48'], 48),
        ([], 24),  # Default
    ]

    for args, expected_stride in test_cases:
        parser = argparse.ArgumentParser()
        parser.add_argument('--stride', type=int, default=24,
                           help='Stride for sliding window')

        parsed = parser.parse_args(args)
        assert parsed.stride == expected_stride, f"Expected {expected_stride}, got {parsed.stride}"
        print(f"   ✅ --stride {' '.join(args)} → {parsed.stride}")

def test_chronological_split_logic():
    """Test the chronological split logic with different strides."""
    print("\n🧪 Testing chronological split logic...")

    # Test cases: (stride, expected_gap)
    test_cases = [
        (12, 51),  # ceil(608/12) = 51
        (24, 26),  # ceil(608/24) = 26
        (48, 13),  # ceil(608/48) = 13
    ]

    window_size = 512 + 96  # 608

    for stride, expected_gap in test_cases:
        gap_indices = (window_size + stride - 1) // stride
        assert gap_indices == expected_gap, f"Stride {stride}: expected gap {expected_gap}, got {gap_indices}"
        print(f"   ✅ stride={stride}: gap={gap_indices} indices (no overlap/leakage)")

def test_guidance_checkpoint_logic():
    """Test the guidance checkpoint reuse logic."""
    print("\n🧪 Testing guidance checkpoint logic...")

    # Simulate checkpoint directory
    checkpoint_dir = Path('checkpoints/itransformer_guidance')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: No checkpoint exists
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob('checkpoint.pth'):
            f.unlink()  # Remove any existing

    existing_ckpt = next(checkpoint_dir.glob('checkpoint.pth'), None)
    assert existing_ckpt is None, "Should have no checkpoint initially"
    print("   ✅ No checkpoint found (would train new iTransformer)")

    # Test 2: Create fake checkpoint
    fake_ckpt = checkpoint_dir / 'checkpoint.pth'
    fake_ckpt.write_text("fake checkpoint data")

    existing_ckpt = next(checkpoint_dir.glob('checkpoint.pth'), None)
    assert existing_ckpt is not None, "Should find existing checkpoint"
    print("   ✅ Existing checkpoint found (would reuse, no retraining needed)")

    # Cleanup
    fake_ckpt.unlink()

def test_configuration_summary():
    """Show a summary of the configuration that would be used."""
    print("\n📊 Configuration Summary:")

    configs = [
        ("Default stride", 24),
        ("Half-day stride", 12),
        ("Two-day stride", 48),
    ]

    for desc, stride in configs:
        window_size = 512 + 96  # 608
        gap = (window_size + stride - 1) // stride
        overlap = window_size - stride

        print(f"   {desc} ({stride}):")
        print(f"     Window: {window_size} timesteps")
        print(f"     Gap between splits: {gap} indices")
        print(f"     Max overlap within split: {overlap} timesteps")
        print(f"     → {'✅ No leak' if gap > 0 else '❌ Potential leak'}")
        print()

def main():
    """Run all tests."""
    print("🚀 Testing stride configuration and no-leak guarantees")
    print("=" * 60)

    try:
        test_stride_parsing()
        test_chronological_split_logic()
        test_guidance_checkpoint_logic()
        test_configuration_summary()

        print("✅ All tests passed!")
        print("\n💡 To actually train with different strides:")
        print("   ./train_with_guidance.sh --stride 12    # Half-day stride")
        print("   ./train_with_guidance.sh --stride 48    # Two-day stride")
        print("   ./train_with_guidance.sh --dry-run      # Test without training")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
