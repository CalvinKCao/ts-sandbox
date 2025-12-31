# Pathformer for ETTh2 with DILATE Loss

This folder contains scripts to train and visualize Pathformer model on the ETTh2 dataset with an option to use DILATE loss instead of the default MAE/MSE loss.

## Features

- **Multiple Loss Functions**: Train with MAE (default), MSE, or DILATE loss
- **DILATE Loss**: Shape-aware loss function that considers both shape similarity and temporal alignment
- **Comprehensive Visualizations**: Reusable visualization utilities from `shared_utils/`
- **Easy to Use**: Simple training and visualization scripts with sensible defaults

## Quick Start

### 1. Train with Default Loss (MAE)

```bash
cd models/pathformer-main
python train_etth2.py
```

### 2. Train with DILATE Loss

```bash
python train_etth2.py --loss_type dilate --dilate_alpha 0.5 --dilate_gamma 0.01
```

### 3. Train with MSE Loss

```bash
python train_etth2.py --loss_type mse
```

### 4. Visualize Results

After training, visualize the predictions:

```bash
python visualize_etth2.py
```

Or for a specific checkpoint:

```bash
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae
```

## Key Arguments

### Training Arguments (`train_etth2.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--loss_type` | `mae` | Loss function: `mae`, `mse`, or `dilate` |
| `--dilate_alpha` | `0.5` | DILATE shape vs temporal weight (0-1) |
| `--dilate_gamma` | `0.01` | DILATE smoothing parameter |
| `--seq_len` | `96` | Input sequence length |
| `--pred_len` | `96` | Prediction horizon |
| `--train_epochs` | `30` | Number of training epochs |
| `--batch_size` | `512` | Batch size |
| `--learning_rate` | `0.0005` | Learning rate |
| `--patience` | `10` | Early stopping patience |

### Visualization Arguments (`visualize_etth2.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--setting` | `None` | Specific checkpoint to load |
| `--num_samples` | `5` | Number of samples to visualize |
| `--save_metrics` | `False` | Save metrics to JSON file |

## File Structure

```
pathformer-main/
├── train_etth2.py              # Training script with DILATE support
├── visualize_etth2.py          # Visualization script
├── dilate_loss_wrapper.py      # DILATE loss implementation
├── run_experiments.bat         # Batch script to run multiple experiments
├── README_ETTH2.md            # This file
├── checkpoints/                # Saved model checkpoints
├── visualizations/             # Generated visualizations
└── test_results/               # Test metrics and results
```

## DILATE Loss Parameters

The DILATE loss combines shape similarity and temporal alignment:

**DILATE = α × Shape Loss + (1-α) × Temporal Loss**

- **alpha (0 to 1)**: Controls the balance between shape and temporal loss
  - `alpha=1.0`: Only shape loss (like soft-DTW)
  - `alpha=0.0`: Only temporal loss (penalizes time shifts)
  - `alpha=0.5`: Equal balance (recommended)

- **gamma**: Smoothing parameter for soft-DTW
  - Lower values: More precise alignment but harder to optimize
  - Higher values: Smoother but less precise
  - `gamma=0.01`: Recommended starting point

## Output Files

### Checkpoints
Trained models are saved in `checkpoints/<setting>/checkpoint.pth`

### Visualizations
For each experiment, the following visualizations are generated:
- `etth2_predictions_random_samples.png`: Random sample predictions
- `etth2_predictions_distributions.png`: Feature distribution comparison
- `etth2_predictions_pca.png`: PCA embedding
- `etth2_predictions_tsne.png`: t-SNE embedding
- `etth2_predictions_stats.png`: Statistical summary
- `first_3_samples.png`: First 3 test samples
- `last_3_samples.png`: Last 3 test samples
- `metrics.json`: Evaluation metrics (if `--save_metrics` flag is used)

### Test Results
Performance metrics are logged in `result.txt` and saved in `test_results/<setting>/`

## ETTh2 Dataset

The ETTh2 (Electricity Transformer Temperature - Hourly 2) dataset contains:
- **7 features**: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- **Frequency**: Hourly measurements
- **Task**: Multivariate time series forecasting

## Example: Running Multiple Experiments

Compare different loss functions:

```bash
# MAE loss
python train_etth2.py --loss_type mae --train_epochs 30

# MSE loss
python train_etth2.py --loss_type mse --train_epochs 30

# DILATE loss (shape-focused)
python train_etth2.py --loss_type dilate --dilate_alpha 0.8 --train_epochs 30

# DILATE loss (temporal-focused)
python train_etth2.py --loss_type dilate --dilate_alpha 0.2 --train_epochs 30

# DILATE loss (balanced)
python train_etth2.py --loss_type dilate --dilate_alpha 0.5 --train_epochs 30
```

Then visualize all results:

```bash
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae --save_metrics
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mse --save_metrics
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.8_g0.01 --save_metrics
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.2_g0.01 --save_metrics
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01 --save_metrics
```

## Shared Visualization Utils

The visualization scripts have been moved to `shared_utils/` for better portability:
- `shared_utils/visualization.py`: Main TimeSeriesVisualizer class
- `shared_utils/visualization_metrics.py`: PCA/t-SNE visualization functions

These can be reused across different models and datasets.

## Requirements

Make sure you have the required packages installed:
- torch
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn

For DILATE loss, you also need:
- numba

## Tips

1. **DILATE Loss Tuning**:
   - Start with `alpha=0.5` and `gamma=0.01`
   - If predictions have good shapes but poor timing, decrease alpha
   - If predictions have good timing but poor shapes, increase alpha
   - Adjust gamma if training is unstable

2. **GPU Usage**:
   - The code automatically uses GPU if available
   - For multi-GPU training, use `--use_multi_gpu --devices 0,1,2,3`

3. **Early Stopping**:
   - Default patience is 10 epochs
   - Increase if training loss is still decreasing
   - Decrease to save time if validation loss plateaus quickly

4. **Batch Size**:
   - Default 512 works well for most GPUs
   - Reduce if you run out of memory
   - Increase if you have more GPU memory for faster training

## Troubleshooting

**Issue**: DILATE loss not found
- Make sure the `losses/DILATE-master` folder is present
- Check that `numba` is installed: `pip install numba`

**Issue**: Out of memory
- Reduce `--batch_size` (try 256, 128, or 64)
- Reduce `--seq_len` or `--pred_len`

**Issue**: Training is very slow
- Increase `--batch_size` if you have GPU memory
- Reduce `--num_workers` if CPU is bottleneck
- Use `--use_amp` for automatic mixed precision (faster but may affect accuracy)

## Citation

If you use this code, please cite the original papers:

**Pathformer**:
```
@article{pathformer2023,
  title={PathFormer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting},
  author={...},
  journal={...},
  year={2023}
}
```

**DILATE**:
```
@inproceedings{dilate2019,
  title={Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
  author={Le Guen, Vincent and Thome, Nicolas},
  booktitle={NeurIPS},
  year={2019}
}
```
