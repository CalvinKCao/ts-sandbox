"""
Example Usage Script for ETTm2 GAN
Demonstrates how to use the data loader, training, and visualization modules
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from ettm2_dataloader import get_ettm2_dataloader, ETTm2Dataset
from visualization import TimeSeriesVisualizer


def demo_dataloader():
    """Demonstrate the ETTm2 data loader"""
    print("\n" + "="*80)
    print("DEMO 1: ETTm2 Data Loader")
    print("="*80)
    
    # Create data loaders for all splits
    train_loader, train_dataset = get_ettm2_dataloader(
        seq_len=96,
        batch_size=32,
        data_mode='Train',
        features='M'
    )
    
    val_loader, val_dataset = get_ettm2_dataloader(
        seq_len=96,
        batch_size=32,
        data_mode='Val',
        features='M'
    )
    
    test_loader, test_dataset = get_ettm2_dataloader(
        seq_len=96,
        batch_size=32,
        data_mode='Test',
        features='M'
    )
    
    print(f"\nTrain dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")
    print(f"Test dataset: {len(test_dataset)} sequences")
    
    # Get one batch
    for batch_data, batch_labels in train_loader:
        print(f"\nBatch shape: {batch_data.shape}")
        print(f"  - Batch size: {batch_data.shape[0]}")
        print(f"  - Channels: {batch_data.shape[1]}")
        print(f"  - Height: {batch_data.shape[2]}")
        print(f"  - Sequence length: {batch_data.shape[3]}")
        print(f"\nData statistics:")
        print(f"  - Min: {batch_data.min():.4f}")
        print(f"  - Max: {batch_data.max():.4f}")
        print(f"  - Mean: {batch_data.mean():.4f}")
        print(f"  - Std: {batch_data.std():.4f}")
        break
    
    # Visualize a few samples
    print("\nVisualizing raw data samples...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    for i in range(min(9, len(batch_data))):
        # Get one sample and one feature
        sample = batch_data[i, 0, 0, :].numpy()  # First feature
        axes[i].plot(sample)
        axes[i].set_title(f'Sample {i} - {feature_names[0]}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_raw_data.png', dpi=100)
    print("Saved raw data visualization to demo_raw_data.png")
    plt.close()


def demo_visualization():
    """Demonstrate the visualization module with synthetic data"""
    print("\n" + "="*80)
    print("DEMO 2: Visualization Module")
    print("="*80)
    
    # Create synthetic data to simulate real and generated data
    print("\nGenerating synthetic data for demonstration...")
    batch_size, channels, height, seq_len = 100, 7, 1, 96
    
    # Real data: sine waves with noise
    t = np.linspace(0, 4*np.pi, seq_len)
    real_data = np.zeros((batch_size, channels, height, seq_len))
    
    for i in range(batch_size):
        for j in range(channels):
            phase = np.random.rand() * 2 * np.pi
            freq = 0.5 + np.random.rand() * 1.5
            amp = 0.5 + np.random.rand() * 0.5
            noise = np.random.randn(seq_len) * 0.1
            real_data[i, j, 0, :] = amp * np.sin(freq * t + phase) + noise
    
    # Generated data: similar but with slight differences
    generated_data = np.zeros_like(real_data)
    for i in range(batch_size):
        for j in range(channels):
            phase = np.random.rand() * 2 * np.pi
            freq = 0.5 + np.random.rand() * 1.5
            amp = 0.5 + np.random.rand() * 0.5
            noise = np.random.randn(seq_len) * 0.15  # Slightly more noise
            generated_data[i, j, 0, :] = amp * np.sin(freq * t + phase) + noise
    
    # Initialize visualizer
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    visualizer = TimeSeriesVisualizer(
        save_dir='demo_visualizations',
        feature_names=feature_names
    )
    
    print("\nCreating visualizations...")
    
    # 1. Random samples comparison
    print("  1. Random samples comparison...")
    visualizer.plot_random_samples(
        real_data=real_data,
        generated_data=generated_data,
        num_samples=5,
        random_seed=42,
        save_name='demo_random_samples'
    )
    
    # 2. Specific horizons
    print("  2. Specific horizons...")
    visualizer.plot_specific_horizons(
        real_data=real_data,
        generated_data=generated_data,
        horizon_indices=[0, 10, 20],
        save_name='demo_specific_horizons'
    )
    
    # 3. Feature distributions
    print("  3. Feature distributions...")
    visualizer.plot_feature_distributions(
        real_data=real_data,
        generated_data=generated_data,
        save_name='demo_distributions'
    )
    
    # 4. PCA embedding
    print("  4. PCA embedding...")
    visualizer.plot_pca_embedding(
        real_data=real_data,
        generated_data=generated_data,
        save_name='demo_pca'
    )
    
    # 5. Statistical summary
    print("  5. Statistical summary...")
    visualizer.plot_statistical_summary(
        real_data=real_data,
        generated_data=generated_data,
        save_name='demo_stats'
    )
    
    print("\nAll visualizations saved to demo_visualizations/")


def demo_full_report():
    """Demonstrate generating a full report"""
    print("\n" + "="*80)
    print("DEMO 3: Full Report Generation")
    print("="*80)
    
    # Load real ETTm2 data
    print("\nLoading real ETTm2 data...")
    test_loader, test_dataset = get_ettm2_dataloader(
        data_path='../../datasets/ETT-small/ETTm2.csv',
        seq_len=96,
        batch_size=100,
        data_mode='Test',
        features='M',
        num_workers=0
    )
    
    # Get real samples
    real_samples = test_dataset.get_data_for_visualization(100)
    print(f"Real samples shape: {real_samples.shape}")
    
    # Generate fake samples (normally from trained GAN)
    # Here we use slightly modified real data as a placeholder
    print("\nGenerating synthetic samples (placeholder)...")
    generated_samples = real_samples + np.random.randn(*real_samples.shape) * 0.1
    
    # Initialize visualizer
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    visualizer = TimeSeriesVisualizer(
        save_dir='demo_full_report',
        feature_names=feature_names
    )
    
    # Generate full report
    print("\nGenerating comprehensive report...")
    visualizer.generate_full_report(
        real_data=real_samples,
        generated_data=generated_samples,
        num_samples=5,
        random_seed=42,
        report_name='ettm2_demo'
    )
    
    print("\nFull report saved to demo_full_report/")


def demo_training_flow():
    """Demonstrate the training workflow (without actual training)"""
    print("\n" + "="*80)
    print("DEMO 4: Training Workflow Overview")
    print("="*80)
    
    print("\nTraining workflow:")
    print("\n1. Data Loading:")
    print("   train_loader, train_dataset = get_ettm2_dataloader(...)")
    
    print("\n2. Model Initialization:")
    print("   from GANModels import Generator, Discriminator")
    print("   gen_net = Generator(seq_len=96, channels=7, ...)")
    print("   dis_net = Discriminator(in_channels=7, ...)")
    
    print("\n3. Training Loop:")
    print("   for epoch in range(max_epochs):")
    print("       train(gen_net, dis_net, train_loader, ...)")
    print("       save_checkpoint(...)")
    
    print("\n4. Visualization:")
    print("   generator = load_generator(checkpoint_path)")
    print("   generated_samples = generate_samples(generator, ...)")
    print("   visualizer.generate_full_report(real_data, generated_samples)")
    
    print("\nTo actually train:")
    print("   python ETTm2_Train.py")
    print("\nTo visualize results:")
    print("   python visualize_ettm2.py --checkpoint logs/.../checkpoint.pth --full_report")


def main():
    """Run all demos"""
    print("\n" + "#"*80)
    print("# ETTm2 GAN - Example Usage Demonstrations")
    print("#"*80)
    
    try:
        # Demo 1: Data loader
        demo_dataloader()
        
        # Demo 2: Visualization with synthetic data
        demo_visualization()
        
        # Demo 3: Full report with real data
        demo_full_report()
        
        # Demo 4: Training workflow overview
        demo_training_flow()
        
        print("\n" + "#"*80)
        print("# All demos completed successfully!")
        print("#"*80)
        print("\nGenerated files:")
        print("  - demo_raw_data.png")
        print("  - demo_visualizations/")
        print("  - demo_full_report/")
        print("\nNext steps:")
        print("  1. Review the generated visualizations")
        print("  2. Run 'python ETTm2_Train.py' to start training")
        print("  3. Use 'python visualize_ettm2.py' to visualize trained model results")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Make sure the ETTm2 dataset is available at ../../datasets/ETT-small/ETTm2.csv")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
