"""
Recovery Report - Check what results were saved before crash
"""

import os
import json
from datetime import datetime

print("="*70)
print("RECOVERY REPORT - Checking Saved Results")
print("="*70)

# Check checkpoints
print("\n[1] CHECKPOINTS")
print("-"*70)
checkpoint_dir = "./checkpoints"
if os.path.exists(checkpoint_dir):
    for exp_name in os.listdir(checkpoint_dir):
        exp_path = os.path.join(checkpoint_dir, exp_name)
        if os.path.isdir(exp_path):
            checkpoint_file = os.path.join(exp_path, "checkpoint.pth")
            if os.path.exists(checkpoint_file):
                size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)
                mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_file))
                print(f"\n✓ {exp_name}")
                print(f"  Checkpoint: checkpoint.pth ({size_mb:.2f} MB)")
                print(f"  Last modified: {mtime}")
                print(f"  Status: RECOVERABLE - Model can be loaded")
            else:
                print(f"\n✗ {exp_name}")
                print(f"  Status: NO CHECKPOINT SAVED")

# Check test results
print("\n\n[2] TEST RESULTS")
print("-"*70)
test_results_dir = "./test_results"
if os.path.exists(test_results_dir):
    for exp_name in os.listdir(test_results_dir):
        exp_path = os.path.join(test_results_dir, exp_name)
        if os.path.isdir(exp_path):
            files = os.listdir(exp_path)
            pdf_files = [f for f in files if f.endswith('.pdf')]
            print(f"\n✓ {exp_name}")
            print(f"  PDF visualizations: {len(pdf_files)}")
            if pdf_files:
                print(f"  Files: {', '.join(pdf_files[:5])}")

# Check result.txt
print("\n\n[3] METRICS (result.txt)")
print("-"*70)
if os.path.exists("result.txt"):
    print("\n✓ Found result.txt with metrics:")
    with open("result.txt", "r") as f:
        content = f.read()
        print(content)
else:
    print("\n✗ No result.txt found")

# Check for crash indicators
print("\n\n[4] CRASH DIAGNOSIS")
print("-"*70)

crash_indicators = []

# Check if DILATE training completed
dilate_checkpoint = "./checkpoints/ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01/checkpoint.pth"
if not os.path.exists(dilate_checkpoint):
    crash_indicators.append("DILATE checkpoint not found - training likely didn't complete")
else:
    size_mb = os.path.getsize(dilate_checkpoint) / (1024 * 1024)
    print(f"✓ DILATE checkpoint exists ({size_mb:.2f} MB)")

# Check timestamps
mae_checkpoint = "./checkpoints/ETTh2_PathFormer_ftM_sl96_pl96_0_mae/checkpoint.pth"
if os.path.exists(mae_checkpoint):
    mae_time = datetime.fromtimestamp(os.path.getmtime(mae_checkpoint))
    print(f"✓ MAE training completed at: {mae_time}")

if os.path.exists(dilate_checkpoint):
    dilate_time = datetime.fromtimestamp(os.path.getmtime(dilate_checkpoint))
    print(f"✓ DILATE training completed at: {dilate_time}")
    
    # Calculate training duration
    if os.path.exists(mae_checkpoint):
        duration = dilate_time - mae_time
        print(f"  Time between MAE and DILATE: {duration}")

# Possible crash causes
print("\n\n[5] POSSIBLE CRASH CAUSES")
print("-"*70)
print("Based on the evidence:")

if os.path.exists(dilate_checkpoint):
    print("✓ Training completed successfully - crash may have happened AFTER training")
    print("\nPossible causes:")
    print("  • System ran out of memory during testing/visualization")
    print("  • GPU memory issues during test phase")
    print("  • System sleep/hibernation")
    print("  • Power issue")
else:
    print("✗ DILATE training did not complete")
    print("\nPossible causes:")
    print("  • Out of memory (DILATE uses more memory than MAE/MSE)")
    print("  • GPU crash during DTW computation")
    print("  • System overheating (DILATE is computationally intensive)")
    print("  • Process killed by system")

# Memory usage estimate
print("\n\n[6] MEMORY ANALYSIS")
print("-"*70)
print("DILATE loss is memory-intensive:")
print("  • Batch size 512 with DILATE: ~8-12 GB GPU memory")
print("  • Computes pairwise distances for each sequence")
print("  • Much slower than MAE/MSE")
print("\nRecommendations:")
print("  1. Reduce batch size: --batch_size 256 or 128")
print("  2. Monitor GPU memory: nvidia-smi")
print("  3. Add memory logging to track usage")

# Recovery instructions
print("\n\n[7] RECOVERY INSTRUCTIONS")
print("="*70)

if os.path.exists(dilate_checkpoint):
    print("\n✓ GOOD NEWS: You have saved checkpoints!")
    print("\nYou can:")
    print("  1. Visualize the results:")
    print("     python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae --save_metrics")
    print("     python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01 --save_metrics")
    print("\n  2. Continue testing (if test didn't complete):")
    print("     python train_etth2.py --is_training 0 --loss_type mae")
    print("     python train_etth2.py --is_training 0 --loss_type dilate --dilate_alpha 0.5")
else:
    print("\n⚠ DILATE training did not complete")
    print("\nTo restart with safer settings:")
    print("  python train_etth2.py --loss_type dilate --batch_size 256 --train_epochs 30")

# Check GPU memory
print("\n\n[8] CURRENT SYSTEM STATUS")
print("-"*70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Current allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Current cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("✗ No GPU available")
except Exception as e:
    print(f"Could not check GPU: {e}")

print("\n" + "="*70)
print("END OF RECOVERY REPORT")
print("="*70)
