"""
Test Script for ETTm2 GAN
Evaluate trained model on test set and generate visualizations
"""

import torch
import numpy as np
import argparse
import os
from ettm2_dataloader import get_ettm2_dataloader
from GANModels import Generator
from visualization import TimeSeriesVisualizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
import json


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def generate_samples(generator, device, num_samples, latent_dim, batch_size=64):
    """Generate synthetic samples using the trained generator"""
    generator.eval()
    generated_samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            z = torch.randn(current_batch, latent_dim).to(device)
            gen_data = generator(z)
            generated_samples.append(gen_data.cpu().numpy())
    
    generated_samples = np.concatenate(generated_samples, axis=0)
    print(f"Generated {num_samples} samples with shape {generated_samples.shape}")
    return generated_samples


def calculate_metrics(real_data, generated_data):
    """Calculate evaluation metrics between real and generated data"""
    metrics = {}
    
    # Reshape data for metric calculation
    real_flat = real_data.reshape(real_data.shape[0], -1)
    gen_flat = generated_data.reshape(generated_data.shape[0], -1)
    
    # Use same number of samples
    n_samples = min(len(real_flat), len(gen_flat))
    real_flat = real_flat[:n_samples]
    gen_flat = gen_flat[:n_samples]
    
    # Statistical metrics
    metrics['mean_mse'] = mean_squared_error(real_flat.mean(axis=0), gen_flat.mean(axis=0))
    metrics['mean_mae'] = mean_absolute_error(real_flat.mean(axis=0), gen_flat.mean(axis=0))
    
    # Feature-wise statistics
    real_mean = real_data.mean(axis=(0, 2, 3))
    gen_mean = generated_data.mean(axis=(0, 2, 3))
    real_std = real_data.std(axis=(0, 2, 3))
    gen_std = generated_data.std(axis=(0, 2, 3))
    
    metrics['mean_difference'] = np.abs(real_mean - gen_mean).mean()
    metrics['std_difference'] = np.abs(real_std - gen_std).mean()
    
    # Wasserstein distance (sample a subset for speed)
    sample_size = min(1000, len(real_flat))
    indices = np.random.choice(len(real_flat), sample_size, replace=False)
    metrics['wasserstein_distance'] = wasserstein_distance(
        real_flat[indices].flatten(), 
        gen_flat[indices].flatten()
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test ETTm2 GAN on test set')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='Sequence length')
    parser.add_argument('--patch_size', type=int, default=12,
                       help='Patch size')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension')
    parser.add_argument('--embed_dim', type=int, default=10,
                       help='Embedding dimension')
    parser.add_argument('--gen_depth', type=int, default=3,
                       help='Generator depth')
    parser.add_argument('--num_heads', type=int, default=5,
                       help='Number of attention heads')
    
    # Data parameters
    parser.add_argument('--data_path', type=str,
                       default='../../datasets/ETT-small/ETTm2.csv',
                       help='Path to ETTm2 dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for loading test data')
    parser.add_argument('--num_test_samples', type=int, default=500,
                       help='Number of test samples to evaluate')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--generate_viz', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--num_viz_samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("ETTm2 GAN Test Evaluation")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Initialize generator
    print("\nInitializing generator...")
    generator = Generator(
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        channels=7,
        num_classes=1,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        depth=args.gen_depth,
        num_heads=args.num_heads,
        forward_drop_rate=0.0,
        attn_drop_rate=0.0
    )
    
    # Load generator weights
    if 'avg_gen_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['avg_gen_state_dict'])
    elif 'gen_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['gen_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.to(device)
    generator.eval()
    print("Generator loaded successfully!")
    
    # Load test data
    print("\nLoading test data...")
    test_loader, test_dataset = get_ettm2_dataloader(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        data_mode='Test',
        num_workers=0,
        shuffle=False,
        normalize=True,
        features='M'
    )
    
    # Get real test samples
    real_samples = test_dataset.get_data_for_visualization(args.num_test_samples)
    print(f"Loaded {len(real_samples)} real test samples")
    
    # Generate synthetic samples
    print("\nGenerating synthetic samples...")
    generated_samples = generate_samples(
        generator=generator,
        device=device,
        num_samples=args.num_test_samples,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size
    )
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(real_samples, generated_samples)
    
    print("\n" + "=" * 80)
    print("Evaluation Metrics:")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key:30s}: {value:.6f}")
    print("=" * 80)
    
    # Save metrics to file (convert numpy types to python native types)
    metrics_json = {k: float(v) for k, v in metrics.items()}
    metrics_file = os.path.join(args.save_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f"\nMetrics saved to {metrics_file}")
    
    # Generate visualizations if requested
    if args.generate_viz:
        print("\nGenerating visualizations...")
        feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        visualizer = TimeSeriesVisualizer(
            save_dir=args.save_dir,
            feature_names=feature_names
        )
        
        visualizer.generate_full_report(
            real_data=real_samples,
            generated_data=generated_samples,
            num_samples=args.num_viz_samples,
            random_seed=42,
            report_name='test_evaluation'
        )
        
        print(f"Visualizations saved to {args.save_dir}/")
    
    print("\n" + "=" * 80)
    print("Test evaluation completed!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
