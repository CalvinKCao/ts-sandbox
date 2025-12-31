# Quick Reference Guide - ETTm2 TTS-GAN

## Files Created (All in models/tts-gan-main/)

1. **ettm2_dataloader.py** - Data loading for ETTm2 dataset
2. **train_ettm2.py** - Training script for ETTm2
3. **ETTm2_Train.py** - Simple launcher with pre-configured settings
4. **visualization.py** - Reusable visualization toolkit (compartmentalized)
5. **visualize_ettm2.py** - CLI tool for visualizing trained models
6. **demo_usage.py** - Example usage demonstrations
7. **README_ETTm2.md** - Complete documentation
8. **IMPLEMENTATION_SUMMARY.md** - This implementation summary

## Quick Start (3 Steps)

```bash
cd models/tts-gan-main

# Step 1: Test data loader
python ettm2_dataloader.py

# Step 2: Train model (this will take a while!)
python ETTm2_Train.py

# Step 3: Visualize results after training
python visualize_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --full_report
```

## Common Commands

### Training
```bash
# Basic training with defaults
python ETTm2_Train.py

# Custom training
python train_ettm2.py --exp_name my_experiment --batch_size 32 --max_epoch 100

# Resume from checkpoint
python train_ettm2.py --load_path logs/ettm2_gan/Model/checkpoint_epoch_50.pth
```

### Visualization

```bash
# Full report (recommended)
python visualize_ettm2.py --checkpoint PATH_TO_CHECKPOINT --full_report

# Random samples with seed
python visualize_ettm2.py --checkpoint PATH --ts_comparison --random_seed 42 --num_vis_samples 10

# Specific test horizons
python visualize_ettm2.py --checkpoint PATH --ts_comparison --specific_indices 0 10 20 30

# Individual plots
python visualize_ettm2.py --checkpoint PATH --pca
python visualize_ettm2.py --checkpoint PATH --tsne
python visualize_ettm2.py --checkpoint PATH --distributions
python visualize_ettm2.py --checkpoint PATH --stats
```

### Demo
```bash
# Run all demonstrations
python demo_usage.py
```

### Monitor Training
```bash
# Start TensorBoard
tensorboard --logdir=logs

# Then open browser to http://localhost:6006
```

## Key Features of Visualization Module

✅ **Random sampling with seed** - Reproducible random samples
✅ **Specific horizons** - Visualize exact test indices  
✅ **Full reports** - All visualizations at once
✅ **Modular design** - Use individual methods as needed
✅ **CLI and API** - Command line or Python import
✅ **Multiple formats** - Auto-handles data shapes
✅ **Publication ready** - High-quality outputs

## Visualization Methods

```python
from visualization import TimeSeriesVisualizer

viz = TimeSeriesVisualizer(save_dir='results', feature_names=[...])

# Random samples
viz.plot_random_samples(real, gen, num_samples=10, random_seed=42)

# Specific samples
viz.plot_specific_horizons(real, gen, horizon_indices=[0, 10, 20])

# Distributions
viz.plot_feature_distributions(real, gen)

# Embeddings
viz.plot_pca_embedding(real, gen)
viz.plot_tsne_embedding(real, gen, random_seed=42)

# Statistics
viz.plot_statistical_summary(real, gen)

# Everything
viz.generate_full_report(real, gen, random_seed=42)
```

## File Purposes

| File | Purpose |
|------|---------|
| ettm2_dataloader.py | Load and preprocess ETTm2 data |
| train_ettm2.py | Core training script |
| ETTm2_Train.py | Easy launcher with defaults |
| visualization.py | Reusable visualization toolkit |
| visualize_ettm2.py | CLI for post-training visualization |
| demo_usage.py | Examples and demonstrations |
| README_ETTm2.md | Full documentation |

## Default Configuration

- **Sequence length**: 96 timesteps
- **Channels**: 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- **Batch size**: 64
- **Epochs**: 200
- **Generator LR**: 0.0001
- **Discriminator LR**: 0.0003
- **Loss**: LSGAN
- **Optimizer**: AdamW

## Troubleshooting

**Data not found?**
- Check path: `../../datasets/ETT-small/ETTm2.csv`

**Out of memory?**
- Reduce batch_size in ETTm2_Train.py
- Reduce seq_len

**Training unstable?**
- Reduce learning rates (g_lr, d_lr)
- Try different loss function

**Need help?**
- See README_ETTm2.md for detailed documentation
- Run demo_usage.py for examples

## Output Locations

```
logs/
  ettm2_gan/
    Model/checkpoint_epoch_*.pth
    Log/events.out.tfevents.*

visualizations/
  *.png

demo_visualizations/
  *.png

demo_full_report/
  *.png
```

## Testing on Test Set

```bash
# Basic test evaluation
python test_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_best.pth

# Test with visualizations
python test_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_best.pth --generate_viz --num_test_samples 500

# Test specific checkpoint
python test_ettm2.py --checkpoint logs/ettm2_gan/Model/checkpoint_epoch_100.pth --generate_viz
```

## Checkpoints Saved

- **`checkpoint_epoch_N.pth`** - Regular checkpoints every N epochs
- **`checkpoint_best.pth`** - Best model (lowest loss) when early stopping enabled
- **`checkpoint_interrupted.pth`** - Auto-saved if you Ctrl+C during training

## Next Steps

1. ✅ Files are ready
2. 🔄 Test data loader: `python ettm2_dataloader.py`
3. 🔄 Run demo: `python demo_usage.py`
4. 🔄 Start training: `python ETTm2_Train.py`
5. 🔄 Monitor: `tensorboard --logdir=logs`
6. 🔄 Test model: `python test_ettm2.py --checkpoint PATH --generate_viz`
7. 🔄 Visualize: `python visualize_ettm2.py --checkpoint PATH --full_report`

Enjoy training! 🚀
