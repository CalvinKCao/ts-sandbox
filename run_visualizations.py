import os
import sys
import argparse
import torch

# Add paths
# Use relative path to find model files regardless of where the script is run
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "models", "diffusion_tsf")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from visualize import visualize_samples
from train_electricity import DATASET_REGISTRY

# Parse arguments
parser = argparse.ArgumentParser(description="Generate DiffusionTSF visualizations")
parser.add_argument('--samples', type=int, default=5, help="Number of samples per dataset (default: 5)")
parser.add_argument('--dataset', type=str, default=None, help="Only visualize specific dataset (e.g., ETTh2)")
parser.add_argument('--ddim-steps', type=int, default=50, help="DDIM sampling steps (default: 50)")
parser.add_argument('--pretrain-only', action='store_true', help="Use universal pretrained model (no fine-tuning) for comparison")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducible sample selection")
args = parser.parse_args()

NUM_SAMPLES = args.samples
DDIM_STEPS = args.ddim_steps
PRETRAIN_ONLY = args.pretrain_only

# Set seed for reproducible sample selection
torch.manual_seed(args.seed)
import random
random.seed(args.seed)

datasets = {
    "ETTh2": "MUFL",
    "ETTm1": "HUFL",
    "illness": "AGE 0-4",
    "exchange_rate": "OT",
    "traffic": "394",
    "weather": "raining (s)"
}

# Filter to specific dataset if requested
if args.dataset:
    if args.dataset in datasets:
        datasets = {args.dataset: datasets[args.dataset]}
    else:
        print(f"❌ Unknown dataset: {args.dataset}")
        print(f"Available: {list(datasets.keys())}")
        sys.exit(1)

checkpoint_base = os.path.join(script_dir, "checkpoints")

if PRETRAIN_ONLY:
    output_base = os.path.join(script_dir, "final_visualizations_pretrain")
    pretrain_model_path = os.path.join(checkpoint_base, "universal_synthetic_pretrain", "best_model.pt")
    print(f"🔬 PRETRAIN-ONLY MODE: Using universal pretrained model (no fine-tuning)")
    print(f"   Model: {pretrain_model_path}")
    if not os.path.exists(pretrain_model_path):
        print(f"❌ Pretrained model not found at {pretrain_model_path}")
        sys.exit(1)
else:
    output_base = os.path.join(script_dir, "final_visualizations")

print(f"Settings: samples={NUM_SAMPLES}, ddim_steps={DDIM_STEPS}, pretrain_only={PRETRAIN_ONLY}")

def sanitize_name(name):
    # This must match exactly how the shell script sanitized the names
    import re
    # Replace anything not alphanumeric, hyphen or underscore with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Replace multiple underscores with one
    sanitized = re.sub(r'__+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

for ds_name, col in datasets.items():
    sanitized_col = sanitize_name(col)
    ds_var = f"{ds_name}_{sanitized_col}"
    
    if PRETRAIN_ONLY:
        # Use the universal pretrained model for all datasets
        model_path = pretrain_model_path
        guidance_path = None  # Pretrained model has no guidance
        output_suffix = f"{ds_var}_pretrain"
    else:
        # Use fine-tuned model
        model_path = os.path.join(checkpoint_base, ds_var, "best_model.pt")
        if not os.path.exists(model_path):
            print(f"Skipping {ds_var}, model not found at {model_path}")
            continue
        
        guidance_path = os.path.join(checkpoint_base, ds_var, "guidance", "checkpoint.pth")
        if not os.path.exists(guidance_path):
            guidance_path = None
            print(f"No guidance found for {ds_var}")
        output_suffix = ds_var
    
    data_rel_path = DATASET_REGISTRY[ds_name][0]
    data_path = os.path.join(current_dir, "datasets", data_rel_path)
    
    mode_label = "PRETRAINED" if PRETRAIN_ONLY else "fine-tuned"
    print(f"\n>>> Visualizing {ds_var} ({NUM_SAMPLES} samples, {mode_label})...")
    try:
        visualize_samples(
            model_path=model_path,
            data_path=data_path,
            num_samples=NUM_SAMPLES,
            output_dir=os.path.join(output_base, output_suffix),
            guidance_checkpoint=guidance_path,
            decoder_method="mean"
        )
        print(f"✅ Saved to {os.path.join(output_base, output_suffix)}")
    except Exception as e:
        print(f"❌ Error visualizing {ds_var}: {e}")
        import traceback
        traceback.print_exc()

print("\nDone with all visualizations.")


