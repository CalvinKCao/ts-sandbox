"""
Visualization for 7-Variate pipeline models.

Wraps the existing visualize.py to work with 7var pipeline checkpoints.
Generates forecast plots for all completed models.

Usage:
    # On cluster: generate viz for all completed models
    python -m models.diffusion_tsf.visualize_7var

    # Specific subset
    python -m models.diffusion_tsf.visualize_7var --subset ETTh1

    # Then sync to local:
    rsync -avz user@narval:~/projects/def-*/diffusion-tsf/results/viz/ ./synced_viz/
"""

import argparse
import json
import os
import sys

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from models.diffusion_tsf.train_7var_pipeline import (
    CHECKPOINT_DIR, RESULTS_DIR, DATASET_REGISTRY, DATASETS_DIR,
    TrainingManifest, MANIFEST_PATH,
    create_itransformer, create_diffusion_model,
    LOOKBACK_LENGTH, FORECAST_LENGTH, N_VARIATES,
)
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.visualize import visualize_samples


def visualize_subset(
    subset_id: str,
    checkpoint_dir: str = CHECKPOINT_DIR,
    output_dir: str = None,
    num_samples: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Generate visualizations for a single trained subset."""
    
    subset_dir = os.path.join(checkpoint_dir, subset_id)
    best_ckpt = os.path.join(subset_dir, 'best.pt')
    metadata_path = os.path.join(subset_dir, 'metadata.json')
    
    if not os.path.exists(best_ckpt):
        print(f"No checkpoint for {subset_id}, skipping")
        return
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    dataset_name = metadata['dataset_name']
    
    # Resolve data path
    if dataset_name not in DATASET_REGISTRY:
        print(f"Unknown dataset {dataset_name}, skipping")
        return
    
    data_path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])
    
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, 'viz', subset_id)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find iTransformer checkpoint
    itrans_ckpt = os.path.join(checkpoint_dir, 'pretrained_itransformer.pt')
    guidance_ckpt = itrans_ckpt if os.path.exists(itrans_ckpt) else None
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {subset_id}")
    print(f"Dataset: {dataset_name}")
    print(f"Variates: {metadata.get('variate_indices', [])}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    try:
        visualize_samples(
            model_path=best_ckpt,
            data_path=data_path,
            num_samples=num_samples,
            device=device,
            output_dir=output_dir,
            guidance_checkpoint=guidance_ckpt,
        )
        print(f"[OK] Saved visualizations to {output_dir}")
    except Exception as e:
        print(f"[ERROR] {subset_id}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Visualize 7-Variate pipeline models')
    parser.add_argument('--subset', type=str, default=None, help='Specific subset to visualize')
    parser.add_argument('--num-samples', type=int, default=5, help='Samples per model')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (default: results_7var/viz/)')
    
    args = parser.parse_args()
    
    manifest_path = os.path.join(args.checkpoint_dir, 'training_manifest.json')
    
    if args.subset:
        visualize_subset(
            args.subset, 
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
        )
    else:
        # Visualize all completed models
        if not os.path.exists(manifest_path):
            print("No training manifest found. Train models first.")
            return
        
        manifest = TrainingManifest.load(manifest_path)
        completed = [k for k, v in manifest.subsets.items() if v.get('status') == 'complete']
        
        if not completed:
            print("No completed models found.")
            return
        
        print(f"Visualizing {len(completed)} completed models...")
        
        for subset_id in completed:
            visualize_subset(
                subset_id,
                checkpoint_dir=args.checkpoint_dir,
                num_samples=args.num_samples,
            )
    
    print(f"\nDone! Sync to local with:")
    print(f"  rsync -avz user@narval:~/projects/def-*/diffusion-tsf/results/viz/ ./synced_viz/")


if __name__ == '__main__':
    main()

