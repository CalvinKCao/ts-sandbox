# Quick Reference Guide

## Installation

First, make sure you have the required packages:
```bash
pip install torch numpy pandas matplotlib scikit-learn seaborn numba
```

## File Locations

```
ts-sandbox/
├── shared_utils/                    # Reusable visualization utilities
│   ├── visualization.py            # Main visualizer class
│   ├── visualization_metrics.py    # PCA/t-SNE functions
│   └── README.md                   # Documentation
│
├── models/pathformer-main/
│   ├── train_etth2.py             # Training script (NEW)
│   ├── visualize_etth2.py         # Visualization script (NEW)
│   ├── dilate_loss_wrapper.py     # DILATE loss (NEW)
│   ├── test_implementation.py     # Test script (NEW)
│   ├── README_ETTH2.md           # Full documentation (NEW)
│   ├── quick_start.bat            # Quick start script (NEW)
│   └── run_experiments.bat        # Run all experiments (NEW)
│
├── datasets/ETT-small/
│   └── ETTh2.csv                  # Dataset (should exist)
│
└── losses/DILATE-master/          # DILATE loss code (should exist)
```

## Quick Commands

### 1. Test Everything Works
```bash
cd models/pathformer-main
python test_implementation.py
```

### 2. Train with Default Loss (MAE)
```bash
python train_etth2.py
```

### 3. Train with DILATE Loss
```bash
python train_etth2.py --loss_type dilate
```

### 4. Visualize Results
```bash
python visualize_etth2.py --save_metrics
```

### 5. Quick Start (train + visualize)
```bash
quick_start.bat
```

### 6. Run All Experiments
```bash
run_experiments.bat
```

## Common Arguments

### Training (`train_etth2.py`)
```bash
--loss_type mae|mse|dilate    # Loss function (default: mae)
--dilate_alpha 0.5             # DILATE alpha parameter (0-1)
--dilate_gamma 0.01            # DILATE gamma parameter
--train_epochs 30              # Number of epochs
--batch_size 512               # Batch size
--learning_rate 0.0005         # Learning rate
--seq_len 96                   # Input length
--pred_len 96                  # Prediction length
```

### Visualization (`visualize_etth2.py`)
```bash
--setting <name>               # Checkpoint name to visualize
--save_metrics                 # Save metrics to JSON
--num_samples 5                # Number of samples to plot
```

## Understanding Loss Functions

### MAE (Mean Absolute Error)
- **Best for**: General forecasting
- **Speed**: Fast ⚡⚡⚡
- **Use case**: Default choice, good all-around

### MSE (Mean Squared Error)
- **Best for**: When large errors matter more
- **Speed**: Fast ⚡⚡⚡
- **Use case**: Penalize outliers more heavily

### DILATE
- **Best for**: When shape/pattern similarity matters
- **Speed**: Slower ⚡ (due to DTW computation)
- **Use case**: Complex patterns, temporal alignment important
- **Parameters**:
  - `alpha`: 0=temporal focus, 1=shape focus, 0.5=balanced
  - `gamma`: Lower=precise, higher=smooth

## Example Workflows

### Workflow 1: Quick Test
```bash
python train_etth2.py --train_epochs 5
python visualize_etth2.py
```

### Workflow 2: Compare Loss Functions
```bash
python train_etth2.py --loss_type mae
python train_etth2.py --loss_type dilate --dilate_alpha 0.5

python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae --save_metrics
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01 --save_metrics
```

### Workflow 3: Tune DILATE Parameters
```bash
# Shape-focused (good for pattern matching)
python train_etth2.py --loss_type dilate --dilate_alpha 0.8

# Temporal-focused (good for timing accuracy)
python train_etth2.py --loss_type dilate --dilate_alpha 0.2

# Balanced (recommended starting point)
python train_etth2.py --loss_type dilate --dilate_alpha 0.5
```

## Output Locations

- **Checkpoints**: `checkpoints/<setting>/checkpoint.pth`
- **Visualizations**: `visualizations/<setting>/*.png`
- **Metrics**: `visualizations/<setting>/metrics.json`
- **Test Results**: `test_results/<setting>/`
- **Training Logs**: Console output

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch
```

### "No module named 'numba'"
```bash
pip install numba
```

### "Dataset not found"
Make sure `datasets/ETT-small/ETTh2.csv` exists in the correct location.

### Out of memory
```bash
python train_etth2.py --batch_size 256  # or 128
```

### Can't find checkpoint
List available checkpoints:
```bash
dir checkpoints
```
Then use exact name:
```bash
python visualize_etth2.py --setting <exact_checkpoint_name>
```

## Tips

1. **Start with MAE**: It's fast and gives good baseline results
2. **Try DILATE for complex patterns**: If shape similarity matters
3. **Adjust batch size**: Based on your GPU memory
4. **Use --save_metrics**: To compare experiments quantitatively
5. **Check visualizations**: Look at all plots to understand model behavior

## Expected Training Times (GPU)

- **MAE/MSE**: ~30-60 seconds per epoch
- **DILATE**: ~2-5 minutes per epoch

## Expected Performance (96→96 forecast on ETTh2)

- **MAE**: ~0.25-0.30
- **MSE**: ~0.10-0.15
- **RMSE**: ~0.32-0.38

Performance may vary based on:
- Random seed
- Training epochs
- Early stopping
- Loss function choice

## Getting Help

1. Read `README_ETTH2.md` for detailed documentation
2. Check `shared_utils/README.md` for visualization documentation
3. Run `python test_implementation.py` to verify setup
4. Check console output for error messages

## What's New

This implementation adds:
- ✨ DILATE loss support for Pathformer
- ✨ Easy-to-use training script for ETTh2
- ✨ Comprehensive visualization script
- ✨ Shared visualization utilities
- ✨ Detailed documentation
- ✨ Helper scripts for Windows

All reusing existing visualization code from TTS-GAN!
