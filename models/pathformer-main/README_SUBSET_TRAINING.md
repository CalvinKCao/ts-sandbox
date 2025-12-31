# Pathformer Subset Training with Hyperparameter Tuning

This script allows you to train the Pathformer model on a subset of the ETTm2 dataset with automatic hyperparameter tuning on an even smaller subset.

## Features

- **Subset Training**: Train on any percentage of the training data (e.g., 25%, 50%)
- **Quick Hyperparameter Tuning**: Automatically find the best hyperparameters using a small subset
- **Important Hyperparameters Tuned**:
  - `learning_rate`: [0.0005, 0.001, 0.002]
  - `d_model`: [16, 32]
  - `batch_size`: [32, 64]
  - `drop` (dropout): [0.05, 0.1, 0.2]

## Quick Start

### Using the batch file (easiest):
```bash
run_subset_training.bat
```

This will use default settings: 25% training data, 10% for tuning.

### Custom subset sizes:
```bash
run_subset_training.bat 0.5 0.15
```
This trains on 50% of data, tunes on 15%.

### Using Python directly:
```bash
python train_subset_with_tuning.py --train_subset 0.25 --tune_subset 0.10
```

## Command Line Arguments

### Key Arguments:
- `--train_subset`: Percentage of training data to use for final training (default: 0.25)
- `--tune_subset`: Percentage of training data to use for hyperparameter tuning (default: 0.10)
- `--skip_tuning`: Skip hyperparameter tuning and use specified params
- `--train_epochs`: Number of epochs for final training (default: 10)

### Data Arguments:
- `--data`: Dataset name (default: 'ETTm2')
- `--root_path`: Path to dataset (default: '../../datasets/ETT-small/')
- `--seq_len`: Input sequence length (default: 96)
- `--pred_len`: Prediction length (default: 96)

### Model Arguments:
- `--d_model`: Model dimension (default: 16, will be tuned)
- `--batch_size`: Batch size (default: 64, will be tuned)
- `--learning_rate`: Learning rate (default: 0.001, will be tuned)
- `--drop`: Dropout rate (default: 0.1, will be tuned)

## How It Works

1. **Hyperparameter Tuning Phase**:
   - Tests 12 different hyperparameter combinations (3 LRs × 2 d_models × 2 batch_sizes × 1 dropout variations = simplified grid)
   - Each configuration trains for 2 epochs on the tune_subset
   - Selects configuration with lowest validation loss
   - Results saved to `checkpoints_subset/tuning_results.json`

2. **Final Training Phase**:
   - Trains model with best hyperparameters on the full train_subset
   - Trains for the specified number of epochs (default: 10)
   - Uses early stopping (patience: 3 epochs)
   - Evaluates on test set
   - Saves model to `checkpoints_subset/`

## Output Files

- **Checkpoints**: Saved in `checkpoints_subset/`
- **Tuning Results**: `checkpoints_subset/tuning_results.json` contains all tested configurations
- **Test Results**: Saved in `test_results/` folder
- **Metrics**: Appended to `result.txt`

## Examples

### Example 1: Quick test with 10% data
```bash
python train_subset_with_tuning.py --train_subset 0.1 --tune_subset 0.05 --train_epochs 5
```

### Example 2: Skip tuning, use specific hyperparameters
```bash
python train_subset_with_tuning.py --train_subset 0.5 --skip_tuning --learning_rate 0.001 --d_model 32 --batch_size 64
```

### Example 3: Full training with tuning
```bash
python train_subset_with_tuning.py --train_subset 1.0 --tune_subset 0.2 --train_epochs 20
```

## Performance Tips

- Use `--num_workers 0` on Windows to avoid multiprocessing issues
- For quick experiments, use smaller subsets (10-25%)
- For tuning, 10-15% of data is usually sufficient
- GPU is highly recommended but not required

## Estimated Runtimes

For ETTm2 dataset:
- **Hyperparameter tuning** (10% data, 12 configs, 2 epochs each): ~15-30 minutes on GPU
- **Final training** (25% data, 10 epochs): ~10-20 minutes on GPU
- **Full pipeline** (tune + train): ~30-50 minutes on GPU

## Troubleshooting

### Issue: "RuntimeError: DataLoader worker... exited unexpectedly"
**Solution**: Add `--num_workers 0`

### Issue: CUDA out of memory
**Solution**: Reduce `--batch_size` (try 16 or 32)

### Issue: Training too slow
**Solution**: 
- Reduce `--train_subset` and `--tune_subset`
- Reduce `--train_epochs`
- Use smaller `--d_model`

## Notes

- The script preserves the same train/val/test splits as the original dataset
- Subset selection is random but seeded for reproducibility (seed=1024)
- All original Pathformer features are supported (RevIN, AMP, etc.)
