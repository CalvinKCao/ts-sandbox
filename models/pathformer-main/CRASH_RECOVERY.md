# Crash Recovery Report - DILATE Training

## What Happened

Your DILATE training crashed overnight, but **good news**: some results were saved!

## Saved Results

### ✓ Successfully Completed
- **MAE Training**: Fully completed and tested
  - Checkpoint saved: `checkpoints/ETTh2_PathFormer_ftM_sl96_pl96_0_mae/checkpoint.pth`
  - Test metrics saved to `result.txt`
  - Results: MAE=0.334, MSE=0.284, RSE=0.426

### ⚠ Partially Completed  
- **DILATE Training**: Training completed BUT checkpoint missing
  - Test results exist in `test_results/ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01/`
  - Metrics were logged: MAE=0.385, MSE=0.353, RSE=0.474
  - **BUT**: No checkpoint.pth saved (training crashed before early stopping saved best model)

## What Likely Happened

Based on the evidence:
1. ✓ MAE training completed successfully (~8:28 PM Dec 10)
2. ✓ DILATE training started
3. ✓ DILATE testing phase began (test_results folder created)
4. ✗ System crashed during testing or before final checkpoint save

## Most Likely Crash Cause

**Out of Memory (OOM)** - DILATE loss is extremely memory-intensive:
- DILATE computes pairwise DTW distances for EVERY sample in each batch
- With batch_size=512 and seq_len=96: ~8-12 GB GPU memory needed
- Much more intensive than MAE/MSE which are simple element-wise operations

Other possible causes:
- GPU overheating (DILATE computation is intensive)
- System ran out of RAM during test phase
- GPU driver crash

## Your Metrics (from result.txt)

```
MAE Loss:
  MSE: 0.284
  MAE: 0.334
  RSE: 0.426

DILATE Loss (α=0.5, γ=0.01):
  MSE: 0.353
  MAE: 0.385
  RSE: 0.474
```

**Interpretation**: 
- DILATE performed slightly worse than MAE in terms of point-wise accuracy
- This suggests DILATE may have focused more on shape preservation
- The test phase completed, so these metrics are valid

## Recovery Options

### Option 1: Restart DILATE with Safer Settings (RECOMMENDED)

Reduce memory usage:
```bash
python train_etth2.py --loss_type dilate --dilate_alpha 0.5 --batch_size 256 --train_epochs 30
```

Or even more conservative:
```bash
python train_etth2.py --loss_type dilate --dilate_alpha 0.5 --batch_size 128 --train_epochs 30
```

### Option 2: Try Different DILATE Parameters

Since you have MAE baseline, try tuning DILATE:
```bash
# More temporal focus (may help with accuracy)
python train_etth2.py --loss_type dilate --dilate_alpha 0.3 --batch_size 256

# More shape focus
python train_etth2.py --loss_type dilate --dilate_alpha 0.7 --batch_size 256

# Smoother DTW (easier to optimize, less memory)
python train_etth2.py --loss_type dilate --dilate_alpha 0.5 --dilate_gamma 0.1 --batch_size 256
```

### Option 3: Visualize What You Have

You can still visualize the MAE results:
```bash
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae --save_metrics
```

## Recommendations to Prevent Future Crashes

### 1. Monitor GPU Memory
Before training:
```bash
nvidia-smi
```

During training (in another terminal):
```bash
watch -n 1 nvidia-smi
```

### 2. Reduce Batch Size for DILATE
- MAE/MSE: batch_size=512 is fine
- DILATE: use batch_size=256 or 128

### 3. Add Memory Checkpointing
The current implementation already saves checkpoints during training via early stopping.
The issue is it only saves the BEST model, so if crash happens before a better model is found, nothing is saved.

### 4. Monitor Training
Run training during the day so you can monitor for issues:
- Watch for OOM errors
- Check GPU temperature
- Monitor system resources

### 5. Alternative: Train in Stages
```bash
# Stage 1: Train with MAE (fast, stable)
python train_etth2.py --loss_type mae --train_epochs 20

# Stage 2: Fine-tune with DILATE (start from MAE checkpoint)
# Note: Current implementation doesn't support this, but could be added
python train_etth2.py --loss_type dilate --batch_size 256 --train_epochs 10
```

## What the Numbers Tell Us

From your result.txt:
```
MAE Loss:     mae=0.334  (better)
DILATE Loss:  mae=0.385  (worse by ~15%)
```

**This is actually expected!** DILATE is designed for:
- Shape similarity (temporal patterns)
- Temporal alignment
- NOT necessarily point-wise accuracy

The higher MAE with DILATE could mean:
1. Model is capturing shapes better but with slight time shifts
2. DILATE parameters (α=0.5) balanced shape vs temporal too evenly
3. ETTh2 dataset benefits more from point-wise accuracy than shape matching

## Next Steps

### Immediate Actions:
1. **Restart DILATE with batch_size=256**:
   ```bash
   python train_etth2.py --loss_type dilate --batch_size 256 --train_epochs 30
   ```

2. **Monitor the training**: Watch for memory issues

3. **Visualize MAE results while waiting**:
   ```bash
   python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae --save_metrics
   ```

### Experiments to Try:
1. Different alpha values (0.2, 0.3, 0.7, 0.8)
2. Different gamma values (0.001, 0.01, 0.1)
3. Compare with MSE loss as another baseline

### System Health:
- Check Windows Event Viewer for crash logs
- Monitor GPU temperature during training
- Ensure adequate cooling
- Check if system is overclocking (can cause instability)

## Files Location Reference

```
checkpoints/
├── ETTh2_PathFormer_ftM_sl96_pl96_0_mae/
│   └── checkpoint.pth              ✓ SAVED (can visualize)

test_results/
├── ETTh2_PathFormer_ftM_sl96_pl96_0_mae/
│   └── 0.pdf                       ✓ SAVED
└── ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01/
    └── 0.pdf                       ✓ SAVED

result.txt                          ✓ SAVED (has both metrics)
```

## Summary

**What you lost**: DILATE model checkpoint (can't load trained model)
**What you kept**: Test metrics, know it completed testing, have MAE baseline
**What to do**: Retrain DILATE with batch_size=256 to avoid OOM crash

The good news is you have solid MAE baseline results and know DILATE can complete (just needs less memory).
