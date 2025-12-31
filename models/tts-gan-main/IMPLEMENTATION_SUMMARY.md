# ETTm2 TTS-GAN Implementation Summary

## Overview
I've created a complete implementation for training and visualizing TTS-GAN on the ETTm2 time series dataset. All files are in the `models/tts-gan-main` folder.

## Files Created

### 1. **ettm2_dataloader.py** (231 lines)
- Custom PyTorch Dataset for ETTm2
- Loads 7-channel time series data (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- Handles train/val/test splits (70%/10%/20%)
- Normalizes data using StandardScaler
- Creates sequences with configurable length and stride
- Outputs data in TTS-GAN compatible format: (batch, channels, 1, seq_len)
- Includes inverse transform for denormalization
- Standalone testing capability

### 2. **train_ettm2.py** (333 lines)
- Complete training script adapted for ETTm2
- Initializes Generator and Discriminator with ETTm2-specific parameters
- Supports GPU and distributed training
- Checkpoint saving and loading
- TensorBoard logging
- Learning rate scheduling
- EMA (Exponential Moving Average) for generator weights
- Configurable loss functions (hinge, standard, lsgan, wgangp)

### 3. **visualization.py** (566 lines)
- Comprehensive, reusable visualization toolkit
- **TimeSeriesVisualizer** class with methods:
  - `plot_time_series_comparison()` - Side-by-side real vs generated plots
  - `plot_random_samples()` - Random sampling with seed control
  - `plot_specific_horizons()` - Visualize specific test horizons
  - `plot_feature_distributions()` - Histogram comparisons
  - `plot_pca_embedding()` - 2D/3D PCA projection
  - `plot_tsne_embedding()` - t-SNE embedding with seed control
  - `plot_statistical_summary()` - Mean, std, min, max comparisons
  - `generate_full_report()` - Creates all visualizations at once
- Handles multiple data formats automatically
- Customizable feature names
- High-quality output (150 DPI)

### 4. **visualize_ettm2.py** (260 lines)
- Standalone visualization script
- Loads trained generator from checkpoint
- Generates synthetic samples
- Creates visualizations using visualization.py
- Flexible command-line interface:
  - `--full_report` - Generate all visualizations
  - `--ts_comparison` - Time series comparison only
  - `--distributions` - Distribution plots only
  - `--pca` - PCA embedding only
  - `--tsne` - t-SNE embedding only
  - `--stats` - Statistical summary only
  - `--specific_indices` - Visualize specific samples
  - `--random_seed` - Control random sampling

### 5. **ETTm2_Train.py** (111 lines)
- Main training launcher with pre-configured hyperparameters
- Easy-to-modify configuration dictionary
- One-command training
- Default settings optimized for ETTm2

### 6. **README_ETTm2.md** (340 lines)
- Comprehensive documentation
- Quick start guide
- Dataset information
- Model architecture details
- Hyperparameter explanations
- Visualization features overview
- Programmatic usage examples
- Training tips and troubleshooting
- Advanced usage scenarios

### 7. **demo_usage.py** (305 lines)
- Interactive demonstration script
- Shows how to use data loader
- Demonstrates visualization module
- Generates example outputs
- Includes training workflow overview
- Creates sample visualizations with synthetic data

## Key Features

### Data Loading
- ✅ 7-channel multivariate time series support
- ✅ Configurable sequence length (default: 96 timesteps)
- ✅ Train/val/test splits with configurable ratios
- ✅ Data normalization with inverse transform
- ✅ Overlapping sequences with stride control
- ✅ Compatible with TTS-GAN architecture

### Training
- ✅ Transformer-based Generator and Discriminator
- ✅ Multiple loss functions (LSGAN, Hinge, Standard, WGANGP)
- ✅ AdamW optimizer with weight decay
- ✅ Learning rate decay
- ✅ Exponential moving average (EMA)
- ✅ Gradient accumulation
- ✅ Checkpoint saving/loading
- ✅ TensorBoard logging
- ✅ GPU and distributed training support

### Visualization (Highly Compartmentalized & Reusable)
- ✅ **Random sampling with seed control** - Reproducible visualizations
- ✅ **Specific horizon visualization** - Plot exact test indices
- ✅ Time series comparison plots - Real vs generated side-by-side
- ✅ Feature distribution analysis - Histogram comparisons
- ✅ PCA embeddings - 2D/3D projections
- ✅ t-SNE embeddings - Non-linear manifold visualization
- ✅ Statistical summaries - Mean, std, min, max comparisons
- ✅ Full report generation - All visualizations at once
- ✅ Flexible API - Use programmatically or via CLI
- ✅ Automatic data format handling - Works with various shapes
- ✅ Customizable labels - Feature names for interpretability

## Usage Examples

### Quick Start (3 Commands)
```bash
# 1. Test the data loader
python ettm2_dataloader.py

# 2. Train the model
python ETTm2_Train.py

# 3. Visualize results
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --full_report
```

### Advanced Visualization Examples
```bash
# Random samples with specific seed
python visualize_ettm2.py --checkpoint model.pth --ts_comparison --random_seed 42 --num_vis_samples 10

# Specific test horizons
python visualize_ettm2.py --checkpoint model.pth --ts_comparison --specific_indices 0 10 20 30 40 50

# Individual visualizations
python visualize_ettm2.py --checkpoint model.pth --pca
python visualize_ettm2.py --checkpoint model.pth --tsne --random_seed 123
python visualize_ettm2.py --checkpoint model.pth --distributions
```

### Programmatic Usage
```python
from visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer(
    save_dir='my_results',
    feature_names=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
)

# Random samples
visualizer.plot_random_samples(real_data, gen_data, num_samples=10, random_seed=42)

# Specific horizons
visualizer.plot_specific_horizons(real_data, gen_data, horizon_indices=[0, 50, 100])

# Full report
visualizer.generate_full_report(real_data, gen_data, random_seed=42)
```

## Model Configuration

**Default Hyperparameters:**
- Sequence length: 96 timesteps
- Patch size: 12
- Latent dimension: 100
- Generator depth: 3 layers
- Discriminator depth: 3 layers
- Batch size: 64
- Learning rates: G=0.0001, D=0.0003
- Loss: Least Squares GAN (LSGAN)
- Optimizer: AdamW with weight decay

## Architecture Details

**Generator:**
- Input: Random noise (100-dim)
- Architecture: Transformer encoder
- Output: 7 channels × 96 timesteps
- Positional encoding
- 5 attention heads
- 3 transformer layers

**Discriminator:**
- Input: Time series (7 × 96)
- Architecture: Patch-based transformer
- Patch size: 12 (8 patches)
- 5 attention heads
- 3 transformer layers
- Binary classification output

## Visualization Capabilities

The visualization module is **highly compartmentalized and reusable**:

1. **Modular Design**: Each visualization type is a separate method
2. **Flexible Input**: Accepts various data formats automatically
3. **Seed Control**: Random operations support seed for reproducibility
4. **Specific Sampling**: Can visualize exact test horizons
5. **CLI and API**: Use from command line or import as library
6. **Publication Ready**: High-quality outputs (150 DPI)
7. **Feature Labels**: Customizable names for interpretability
8. **Comprehensive Reports**: Generate all plots with one command

## Next Steps

1. **Run the demo**: `python demo_usage.py`
2. **Start training**: `python ETTm2_Train.py`
3. **Monitor progress**: `tensorboard --logdir=logs`
4. **Visualize results**: `python visualize_ettm2.py --checkpoint <path> --full_report`
5. **Experiment**: Modify hyperparameters in `ETTm2_Train.py`

## File Locations

All files are in: `c:\Users\kevin\dev\ts-sandbox\models\tts-gan-main\`

- `ettm2_dataloader.py` - Data loading
- `train_ettm2.py` - Training script  
- `visualization.py` - Visualization toolkit
- `visualize_ettm2.py` - CLI visualization tool
- `ETTm2_Train.py` - Training launcher
- `demo_usage.py` - Usage examples
- `README_ETTm2.md` - Full documentation

## Requirements

The code uses existing TTS-GAN dependencies:
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorBoard

All dependencies should already be installed for the TTS-GAN project.
