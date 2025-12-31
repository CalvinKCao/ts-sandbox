# Pathformer Subset Training - Quick Start Guide

## What Was Created

I've created a complete training system for Pathformer on ETTm2 data subsets with automatic hyperparameter tuning:

### Main Files Created:
1. **`train_subset_with_tuning.py`** - Main training script with hyperparameter search
2. **`run_subset_training.bat`** - Easy-to-use batch file to run training
3. **`examples_subset_training.bat`** - Interactive examples menu
4. **`README_SUBSET_TRAINING.md`** - Comprehensive documentation
5. **`CODE_ISSUES_FOUND.md`** - Documentation of bugs found in existing code

## Quick Start

### Easiest Way (Default Settings):
```bash
run_subset_training.bat
```
This runs with 25% training data and 10% tuning data.

### Custom Subset Sizes:
```bash
run_subset_training.bat 0.5 0.15
```
This uses 50% for training and 15% for tuning.

### Python Command:
```bash
python train_subset_with_tuning.py --train_subset 0.25 --tune_subset 0.10
```

## What It Does

### Phase 1: Hyperparameter Tuning
- Tests 12 different configurations on a small subset (default 10% of training data)
- Tunes 4 key hyperparameters:
  - **learning_rate**: [0.0005, 0.001, 0.002]
  - **d_model**: [16, 32]
  - **batch_size**: [32, 64]
  - **drop**: [0.05, 0.1, 0.2]
- Each config trains for only 2 epochs (very fast)
- Selects the best configuration based on validation loss
- Saves results to `checkpoints_subset/tuning_results.json`

### Phase 2: Final Training
- Trains with best hyperparameters on the full specified subset (default 25%)
- Trains for full epochs (default 10)
- Uses early stopping (patience: 3 epochs)
- Evaluates on test set
- Saves model checkpoints

## Key Features

✓ **Fast**: Quick hyperparameter search on small subsets  
✓ **Automatic**: Finds best hyperparameters without manual tuning  
✓ **Flexible**: Train on any percentage of data (1% to 100%)  
✓ **Documented**: Clear output showing progress and results  
✓ **Safe**: Preserves all original dataset functionality  

## Example Outputs

You'll see output like:
```
================================================================================
STARTING QUICK HYPERPARAMETER TUNING
================================================================================

Testing 12 hyperparameter configurations on 10.0% of training data

[1/12] Testing config: {'learning_rate': 0.0005, 'd_model': 16, 'batch_size': 32, 'drop': 0.05}
Using 3456/34560 training samples (10.0%)
...
Validation Loss: 0.543210
★ New best configuration! Val Loss: 0.543210

[2/12] Testing config: {'learning_rate': 0.001, 'd_model': 16, 'batch_size': 32, 'drop': 0.05}
...
```

## Estimated Times (GPU)

- **Hyperparameter Tuning**: ~15-30 minutes (10% data, 12 configs, 2 epochs each)
- **Final Training**: ~10-20 minutes (25% data, 10 epochs)
- **Total Pipeline**: ~30-50 minutes

## Important Notes

1. **Windows Users**: Use `--num_workers 0` if you get DataLoader errors
2. **Memory Issues**: Reduce `--batch_size` to 32 or 16 if you run out of memory
3. **Quick Tests**: Use `--train_subset 0.1 --tune_subset 0.05` for very fast testing
4. **Skip Tuning**: Use `--skip_tuning` to use default or specified hyperparameters

## Files Generated

After running, you'll have:
- **`checkpoints_subset/`** - Model checkpoints from all runs
- **`checkpoints_subset/tuning_results.json`** - All hyperparameter search results
- **`test_results/`** - Test set predictions and visualizations
- **`result.txt`** - Performance metrics (MSE, MAE, RSE)

## Code Quality Notes

While developing this script, I found one bug in the existing codebase:

**Bug in train_etth2.py (lines 137, 143)**: When using DILATE loss, the code may attempt to call `.item()` on `None` values. This won't affect your usage of the new script since it doesn't use DILATE loss by default. See `CODE_ISSUES_FOUND.md` for details.

## Need Help?

Check these files:
- `README_SUBSET_TRAINING.md` - Full documentation with examples
- `examples_subset_training.bat` - Interactive examples
- `CODE_ISSUES_FOUND.md` - Known issues in existing code

## Example Commands

```bash
# Quick 5-minute test
python train_subset_with_tuning.py --train_subset 0.1 --tune_subset 0.05 --train_epochs 5 --num_workers 0

# Recommended settings
python train_subset_with_tuning.py --train_subset 0.25 --tune_subset 0.10 --num_workers 0

# Full training with extensive tuning
python train_subset_with_tuning.py --train_subset 1.0 --tune_subset 0.2 --train_epochs 20 --num_workers 0

# Skip tuning, specify hyperparameters
python train_subset_with_tuning.py --train_subset 0.5 --skip_tuning --learning_rate 0.001 --d_model 32 --batch_size 64 --drop 0.1
```

Enjoy faster Pathformer training! 🚀
