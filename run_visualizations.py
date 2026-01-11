import os
import sys
import torch

# Add paths
script_dir = "/home/cao/ts-sandbox/models/diffusion_tsf"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from visualize import visualize_samples
from train_electricity import DATASET_REGISTRY

datasets = {
    "ETTh2": "MUFL",
    "ETTm1": "HUFL",
    "illness": "AGE 0-4",
    "exchange_rate": "OT",
    "traffic": "394",
    "weather": "raining (s)"
}

checkpoint_base = "/home/cao/ts-sandbox/models/diffusion_tsf/checkpoints"
output_base = "/home/cao/ts-sandbox/models/diffusion_tsf/final_visualizations"

def sanitize_name(name):
    # This must match exactly how the shell script sanitized the names
    # shell script used: echo "$name" | sed 's/[^a-zA-Z0-9_-]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//'
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
    model_path = os.path.join(checkpoint_base, ds_var, "best_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Skipping {ds_var}, model not found at {model_path}")
        continue
        
    guidance_path = os.path.join(checkpoint_base, ds_var, "guidance", "checkpoint.pth")
    if not os.path.exists(guidance_path):
        guidance_path = None
        print(f"No guidance found for {ds_var}")
    
    data_rel_path = DATASET_REGISTRY[ds_name][0]
    data_path = os.path.join("/home/cao/ts-sandbox/datasets", data_rel_path)
    
    print(f"\n>>> Visualizing {ds_var}...")
    try:
        visualize_samples(
            model_path=model_path,
            data_path=data_path,
            num_samples=5,
            output_dir=os.path.join(output_base, ds_var),
            guidance_checkpoint=guidance_path,
            decoder_method="mean"
        )
    except Exception as e:
        print(f"Error visualizing {ds_var}: {e}")

print("\nDone with all visualizations.")


