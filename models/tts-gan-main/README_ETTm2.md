# TTS-GAN for ETTm2 Time Series Generation

This directory contains code for training and evaluating TTS-GAN (Transformer-based Time-Series GAN) on the ETTm2 dataset.

## Files Overview

### Core Components

1. **ettm2_dataloader.py** - Data loading and preprocessing for ETTm2 dataset
   - Loads ETTm2 time series data
   - Handles train/val/test splits
   - Supports normalization and sequence creation
   - Compatible with TTS-GAN format

2. **train_ettm2.py** - Training script adapted for ETTm2
   - Modified from original train_GAN.py
   - Configured for 7-channel ETTm2 data
   - Supports checkpoint saving/loading
   - Tensorboard logging

3. **visualization.py** - Reusable visualization module
   - Time series comparison plots
   - Random and specific horizon sampling
   - Feature distribution analysis
   - PCA and t-SNE embeddings
   - Statistical summary plots
   - Full report generation

4. **visualize_ettm2.py** - Standalone visualization script
   - Load trained models
   - Generate synthetic samples
   - Create visualizations from checkpoints
   - Flexible command-line interface

5. **ETTm2_Train.py** - Main training launcher
   - Pre-configured hyperparameters
   - Easy-to-modify settings
   - One-command training

## Quick Start

### 1. Test the Data Loader

```bash
python ettm2_dataloader.py
```

### 2. Train the Model

#### Option A: Using the launcher (Recommended)
```bash
python ETTm2_Train.py
```

#### Option B: Direct training with custom parameters
```bash
python train_ettm2.py --exp_name ettm2_gan --batch_size 64 --max_epoch 200 --seq_len 96 --patch_size 12 --latent_dim 100 --g_lr 0.0001 --d_lr 0.0003 --optimizer adamw --loss lsgan --data_path ../../datasets/ETT-small/ETTm2.csv --num_workers 4 --save_freq 10
```

### 3. Visualize Results

#### Generate Full Report
```bash
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --full_report --num_samples 100 --num_vis_samples 5 --random_seed 42
```

#### Visualize Specific Horizons
```bash
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --ts_comparison --specific_indices 0 10 20 30 40
```

#### Individual Visualizations
```bash
# Just time series comparison
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --ts_comparison

# Just distributions
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --distributions

# Just PCA
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --pca

# Just t-SNE
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --tsne

# Just statistics
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --stats
```

## Dataset Information

**ETTm2 Dataset:**
- Source: `../../datasets/ETT-small/ETTm2.csv`
- Features: 7 channels (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- Frequency: 15-minute intervals
- Total samples: 69,680

**Default Splits:**
- Training: 70% (48,776 samples)
- Validation: 10% (6,968 samples)  
- Testing: 20% (13,936 samples)

## Model Architecture

**Generator:**
- Pure transformer encoder architecture
- Input: Random noise vector (latent_dim=100)
- Output: Time series (7 channels × 96 timesteps)
- Positional encoding for temporal information
- Multi-head self-attention (5 heads)
- 3 transformer layers

**Discriminator:**
- Pure transformer encoder architecture
- Patch-based embedding (patch_size=12)
- Classification head for real/fake discrimination
- Multi-head self-attention (5 heads)
- 3 transformer layers

## Key Hyperparameters

```python
seq_len = 96              # Sequence length
patch_size = 12           # Patch size for transformer
latent_dim = 100          # Latent dimension
embed_dim = 10            # Embedding dimension (generator)
dis_embed_dim = 50        # Embedding dimension (discriminator)
batch_size = 64           # Batch size
max_epoch = 200           # Maximum epochs
g_lr = 0.0001            # Generator learning rate
d_lr = 0.0003            # Discriminator learning rate
optimizer = 'adamw'       # AdamW optimizer
loss = 'lsgan'           # Least squares GAN loss
```

## Visualization Features

The visualization module provides:

### 1. Time Series Comparison
- Plot real vs generated sequences side-by-side
- Support for random sampling with seed
- Support for specific test horizons
- Multi-feature visualization

### 2. Feature Distributions
- Histogram comparison for each feature
- Density plots for real vs generated data

### 3. PCA Embedding
- 2D or 3D PCA projection
- Visualize data manifold
- Compare real and generated distributions

### 4. t-SNE Embedding
- 2D t-SNE projection
- Non-linear dimensionality reduction
- Cluster analysis

### 5. Statistical Summary
- Mean, standard deviation, min, max
- Feature-wise comparison
- Bar charts for easy interpretation

### 6. Full Report
- All visualizations in one go
- Comprehensive analysis
- Saved with consistent naming

## Using the Visualization Module Programmatically

```python
from visualization import TimeSeriesVisualizer
import numpy as np

# Initialize visualizer
feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
visualizer = TimeSeriesVisualizer(
    save_dir='my_visualizations',
    feature_names=feature_names
)

# Load your data
real_data = ...  # Shape: (N, 7, 1, 96)
generated_data = ...  # Shape: (N, 7, 1, 96)

# Generate full report
visualizer.generate_full_report(
    real_data=real_data,
    generated_data=generated_data,
    num_samples=5,
    random_seed=42,
    report_name='my_experiment'
)

# Or individual plots
visualizer.plot_random_samples(
    real_data, generated_data, 
    num_samples=10, random_seed=42
)

visualizer.plot_specific_horizons(
    real_data, generated_data,
    horizon_indices=[0, 10, 20, 30]
)

visualizer.plot_feature_distributions(real_data, generated_data)
visualizer.plot_pca_embedding(real_data, generated_data)
visualizer.plot_tsne_embedding(real_data, generated_data, random_seed=42)
visualizer.plot_statistical_summary(real_data, generated_data)
```

## Training Tips

1. **Start with default hyperparameters** - They are tuned for ETTm2
2. **Monitor training** - Use Tensorboard: `tensorboard --logdir=logs`
3. **Save frequently** - Checkpoints every 10 epochs by default
4. **Use GPU** - Training is much faster on GPU
5. **Adjust learning rates** - If training is unstable, reduce learning rates

## Troubleshooting

**Out of Memory:**
- Reduce batch_size
- Reduce seq_len
- Reduce model depth

**Training Unstable:**
- Reduce learning rates
- Try different loss functions ('hinge', 'standard', 'lsgan', 'wgangp')
- Increase gradient accumulation (accumulated_times)

**Poor Quality:**
- Train longer (more epochs)
- Adjust discriminator/generator balance (n_critic)
- Try different patch sizes

## Output Structure

```
logs/
  ettm2_gan/
    Model/
      checkpoint_epoch_10.pth
      checkpoint_epoch_20.pth
      ...
    Log/
      events.out.tfevents...
      
visualizations/
  ettm2_gan_random_samples.png
  ettm2_gan_distributions.png
  ettm2_gan_pca.png
  ettm2_gan_tsne.png
  ettm2_gan_stats.png
```

## Advanced Usage

### Resume Training
```bash
python train_ettm2.py --load_path logs/ettm2_gan/Model/checkpoint_epoch_50.pth --exp_name ettm2_gan_continued
```

### Different Sequence Length
```bash
python train_ettm2.py --seq_len 192 --patch_size 24 --exp_name ettm2_gan_long
```

### CPU Training
```bash
python train_ettm2.py --gpu None --exp_name ettm2_gan_cpu
```

## Citation

If you use this code, please cite the original TTS-GAN paper:

```
@inproceedings{desai2021ttsgan,
  title={TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network},
  author={Desai, Ameya and others},
  booktitle={AIME 2022},
  year={2022}
}
```

## License

See LICENSE file in the main directory.

## Contact

For questions or issues, please open an issue on GitHub.
