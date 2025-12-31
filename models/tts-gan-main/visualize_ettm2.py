"""
Visualization Script for ETTm2 GAN Results
Generates visualizations from trained GAN model and real data
"""

import torch
import numpy as np
import argparse
import os
from ettm2_dataloader import get_ettm2_dataloader
from GANModels import Generator
from visualization import TimeSeriesVisualizer


def load_generator(checkpoint_path, seq_len=96, patch_size=12, channels=7, 
                   latent_dim=100, embed_dim=10, depth=3, num_heads=5):
    """
    Load trained generator from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        seq_len: Sequence length
        patch_size: Patch size
        channels: Number of channels
        latent_dim: Latent dimension
        embed_dim: Embedding dimension
        depth: Transformer depth
        num_heads: Number of attention heads
    
    Returns:
        Loaded generator model
    """
    # Initialize generator
    gen_net = Generator(
        seq_len=seq_len,
        patch_size=patch_size,
        channels=channels,
        num_classes=1,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        forward_drop_rate=0.0,  # No dropout during inference
        attn_drop_rate=0.0
    )
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict (handle different checkpoint formats)
    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    elif 'gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
    else:
        gen_net.load_state_dict(checkpoint)
    
    gen_net.to(device)
    gen_net.eval()
    
    print(f"Generator loaded from {checkpoint_path}")
    print(f"Device: {device}")
    
    return gen_net, device


def generate_samples(generator, device, num_samples=100, latent_dim=100):
    """
    Generate synthetic samples using the trained generator
    
    Args:
        generator: Trained generator model
        device: torch device
        num_samples: Number of samples to generate
        latent_dim: Latent dimension
    
    Returns:
        Generated samples as numpy array
    """
    generator.eval()
    
    generated_samples = []
    batch_size = min(64, num_samples)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            
            # Sample noise
            z = torch.randn(current_batch, latent_dim).to(device)
            
            # Generate
            gen_data = generator(z)
            
            # Move to CPU
            gen_data = gen_data.cpu().numpy()
            generated_samples.append(gen_data)
    
    generated_samples = np.concatenate(generated_samples, axis=0)
    
    print(f"Generated {num_samples} samples with shape {generated_samples.shape}")
    
    return generated_samples


def main():
    parser = argparse.ArgumentParser(description='Visualize ETTm2 GAN results')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
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
    parser.add_argument('--data_mode', type=str, default='Test',
                       choices=['Train', 'Val', 'Test'],
                       help='Which data split to use')
    
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--specific_indices', type=int, nargs='+', default=None,
                       help='Specific indices to visualize')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--report_name', type=str, default='ettm2_gan',
                       help='Base name for report files')
    
    # Visualization options
    parser.add_argument('--full_report', action='store_true',
                       help='Generate full visualization report')
    parser.add_argument('--ts_comparison', action='store_true',
                       help='Generate time series comparison')
    parser.add_argument('--distributions', action='store_true',
                       help='Generate distribution plots')
    parser.add_argument('--pca', action='store_true',
                       help='Generate PCA plot')
    parser.add_argument('--tsne', action='store_true',
                       help='Generate t-SNE plot')
    parser.add_argument('--stats', action='store_true',
                       help='Generate statistical summary')
    
    args = parser.parse_args()
    
    # If no specific visualization is requested, generate full report
    if not any([args.full_report, args.ts_comparison, args.distributions, 
                args.pca, args.tsne, args.stats]):
        args.full_report = True
    
    print("=" * 80)
    print("ETTm2 GAN Visualization")
    print("=" * 80)
    
    # Load generator
    print("\n1. Loading generator...")
    generator, device = load_generator(
        checkpoint_path=args.checkpoint,
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        channels=7,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        depth=args.gen_depth,
        num_heads=args.num_heads
    )
    
    # Load real data
    print("\n2. Loading real data...")
    data_loader, dataset = get_ettm2_dataloader(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.num_samples,
        data_mode=args.data_mode,
        num_workers=0,
        shuffle=True,
        normalize=True,
        features='M'
    )
    
    # Get real samples
    real_samples = dataset.get_data_for_visualization(args.num_samples)
    print(f"Real samples shape: {real_samples.shape}")
    
    # Generate synthetic samples
    print("\n3. Generating synthetic samples...")
    generated_samples = generate_samples(
        generator=generator,
        device=device,
        num_samples=args.num_samples,
        latent_dim=args.latent_dim
    )
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # Initialize visualizer
    print("\n4. Creating visualizations...")
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    visualizer = TimeSeriesVisualizer(
        save_dir=args.save_dir,
        feature_names=feature_names
    )
    
    # Generate visualizations
    if args.full_report:
        visualizer.generate_full_report(
            real_data=real_samples,
            generated_data=generated_samples,
            num_samples=args.num_vis_samples,
            random_seed=args.random_seed,
            report_name=args.report_name
        )
    else:
        if args.ts_comparison:
            if args.specific_indices:
                visualizer.plot_specific_horizons(
                    real_data=real_samples,
                    generated_data=generated_samples,
                    horizon_indices=args.specific_indices,
                    save_name=f'{args.report_name}_specific'
                )
            else:
                visualizer.plot_random_samples(
                    real_data=real_samples,
                    generated_data=generated_samples,
                    num_samples=args.num_vis_samples,
                    random_seed=args.random_seed,
                    save_name=f'{args.report_name}_comparison'
                )
        
        if args.distributions:
            visualizer.plot_feature_distributions(
                real_data=real_samples,
                generated_data=generated_samples,
                save_name=f'{args.report_name}_distributions'
            )
        
        if args.pca:
            visualizer.plot_pca_embedding(
                real_data=real_samples,
                generated_data=generated_samples,
                save_name=f'{args.report_name}_pca'
            )
        
        if args.tsne:
            visualizer.plot_tsne_embedding(
                real_data=real_samples,
                generated_data=generated_samples,
                random_seed=args.random_seed,
                save_name=f'{args.report_name}_tsne'
            )
        
        if args.stats:
            visualizer.plot_statistical_summary(
                real_data=real_samples,
                generated_data=generated_samples,
                save_name=f'{args.report_name}_stats'
            )
    
    print("\n" + "=" * 80)
    print(f"Visualization complete! Results saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
