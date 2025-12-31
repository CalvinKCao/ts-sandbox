"""
Generate synthetic dataset with horizontal lines and sudden jumps
Then train and test TTS-GAN on it
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_jump_dataset(num_samples=70000, seq_len=96, num_channels=7, 
                          jump_probability=0.02, jump_magnitude_range=(2, 5),
                          noise_level=0.05, save_path='../../datasets/synthetic_jumps.csv'):
    """
    Generate synthetic time series with mostly horizontal lines and sudden jumps
    
    Args:
        num_samples: Total number of timesteps
        seq_len: Length of sequences (for reference)
        num_channels: Number of features/channels
        jump_probability: Probability of a jump occurring at any timestep
        jump_magnitude_range: Range of jump magnitudes (min, max)
        noise_level: Amount of Gaussian noise to add
        save_path: Where to save the CSV file
    """
    
    print("Generating synthetic jump dataset...")
    print(f"  Samples: {num_samples}")
    print(f"  Channels: {num_channels}")
    print(f"  Jump probability: {jump_probability}")
    print(f"  Jump magnitude range: {jump_magnitude_range}")
    print(f"  Noise level: {noise_level}")
    
    # Initialize data
    data = np.zeros((num_samples, num_channels))
    
    # Generate data for each channel
    for ch in range(num_channels):
        current_level = np.random.randn() * 2  # Random starting level
        
        for i in range(num_samples):
            # Check if a jump occurs
            if np.random.rand() < jump_probability:
                # Random jump (up or down)
                jump_magnitude = np.random.uniform(*jump_magnitude_range)
                jump_direction = np.random.choice([-1, 1])
                current_level += jump_direction * jump_magnitude
            
            # Add small random noise
            noise = np.random.randn() * noise_level
            data[i, ch] = current_level + noise
    
    # Create DataFrame
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'][:num_channels]
    
    # Create timestamps
    timestamps = pd.date_range(start='2020-01-01', periods=num_samples, freq='15min')
    
    df = pd.DataFrame(data, columns=feature_names)
    df.insert(0, 'date', timestamps)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"\nDataset saved to: {save_path}")
    print(f"Shape: {df.shape}")
    
    return save_path


def visualize_synthetic_data(csv_path, num_samples_to_plot=500):
    """Visualize the generated synthetic data"""
    
    print(f"\nVisualizing synthetic data from {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    data = df.drop(columns=['date']).values[:num_samples_to_plot]
    
    num_channels = data.shape[1]
    feature_names = df.columns[1:]
    
    # Create plots
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels))
    
    if num_channels == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(data[:, i], linewidth=1)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{name} - Horizontal lines with jumps', fontsize=11)
    
    axes[-1].set_xlabel('Time Step', fontsize=10)
    
    plt.tight_layout()
    save_path = 'synthetic_jumps_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()
    
    return save_path


def train_gan_on_synthetic(data_path, epochs=100, batch_size=64):
    """Train GAN on synthetic dataset"""
    
    print("\n" + "="*80)
    print("Training GAN on Synthetic Jump Dataset")
    print("="*80)
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable, 'train_ettm2.py',
        '--batch_size', str(batch_size),
        '--gen_batch_size', str(batch_size),
        '--dis_batch_size', str(batch_size),
        '--max_epoch', str(epochs),
        '--n_critic', '1',
        '--accumulated_times', '1',
        '--g_accumulated_times', '1',
        '--optimizer', 'adamw',
        '--g_lr', '0.0001',
        '--d_lr', '0.0003',
        '--beta1', '0.0',
        '--beta2', '0.9',
        '--wd', '0.001',
        '--lr_decay',
        '--loss', 'lsgan',
        '--phi', '1',
        '--latent_dim', '100',
        '--dropout', '0.5',
        '--patch_size', '12',
        '--data_path', data_path,
        '--num_workers', '4',
        '--exp_name', 'synthetic_jumps_gan',
        '--val_freq', '20',
        '--print_freq', '50',
        '--early_stop',
        '--early_stop_patience', '15',
        '--init_type', 'xavier_uniform',
        '--seed', '12345',
        '--gpu', '0',
        '--rank', '0',
        '--world-size', '1',
        '--dist-url', 'tcp://localhost:4321',
        '--dist-backend', 'nccl',
        '--grow_steps', '0', '0'
    ]
    
    print("\nRunning training command...")
    result = subprocess.run(cmd)
    
    return result.returncode


def test_gan_on_synthetic(checkpoint_path):
    """Test GAN on synthetic dataset"""
    
    print("\n" + "="*80)
    print("Testing GAN on Synthetic Jump Dataset")
    print("="*80)
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable, 'test_ettm2.py',
        '--checkpoint', checkpoint_path,
        '--data_path', '../../datasets/synthetic_jumps.csv',
        '--generate_viz',
        '--num_test_samples', '500',
        '--num_viz_samples', '10',
        '--save_dir', 'synthetic_test_results'
    ]
    
    print("\nRunning test command...")
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    print("\n" + "#"*80)
    print("# Synthetic Jump Dataset - GAN Training Pipeline")
    print("#"*80)
    
    # Step 1: Generate synthetic dataset
    print("\n[Step 1/4] Generating synthetic dataset...")
    data_path = generate_jump_dataset(
        num_samples=70000,
        seq_len=96,
        num_channels=7,
        jump_probability=0.02,  # 2% chance of jump at each timestep
        jump_magnitude_range=(2, 5),  # Jumps between 2 and 5 units
        noise_level=0.05,  # Small noise
        save_path='../../datasets/synthetic_jumps.csv'
    )
    
    # Step 2: Visualize the data
    print("\n[Step 2/4] Visualizing synthetic dataset...")
    visualize_synthetic_data(data_path, num_samples_to_plot=1000)
    
    # Step 3: Train GAN
    print("\n[Step 3/4] Training GAN on synthetic dataset...")
    print("This will take some time...")
    train_result = train_gan_on_synthetic(data_path, epochs=100, batch_size=64)
    
    if train_result != 0:
        print("\nTraining failed or was interrupted.")
        return
    
    # Step 4: Test and visualize
    print("\n[Step 4/4] Testing and visualizing results...")
    
    # Find the best checkpoint
    checkpoint_dir = 'logs/synthetic_jumps_gan'
    checkpoint_files = []
    
    if os.path.exists(checkpoint_dir):
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pth'):
                    checkpoint_files.append(os.path.join(root, file))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    # Use the best checkpoint if available, otherwise the last one
    best_checkpoint = None
    for ckpt in checkpoint_files:
        if 'best' in ckpt:
            best_checkpoint = ckpt
            break
    
    if not best_checkpoint:
        # Use the latest checkpoint
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
        best_checkpoint = checkpoint_files[-1]
    
    print(f"\nUsing checkpoint: {best_checkpoint}")
    test_gan_on_synthetic(best_checkpoint)
    
    print("\n" + "#"*80)
    print("# Pipeline Complete!")
    print("#"*80)
    print("\nResults:")
    print("  1. Synthetic data: ../../datasets/synthetic_jumps.csv")
    print("  2. Data visualization: synthetic_jumps_visualization.png")
    print("  3. Training logs: logs/synthetic_jumps_gan/")
    print("  4. Test results: synthetic_test_results/")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
