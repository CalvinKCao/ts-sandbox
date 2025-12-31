# ETTh1 Paper-Exact Training Configuration

## Summary of Paper Analysis

I've thoroughly read the Pathformer paper (Chen et al., 2024, ICLR 2024) and extracted the exact training specifications for ETTh1. **The text file is readable and contains all important details - no garbled text was found that obscures critical information.**

## Exact Paper Specifications (Section 4.1.2)

### Global Settings (Applied to All Datasets)
- **Optimizer**: Adam
- **Learning Rate**: 10^-3 (0.001)
- **Loss Function**: L1 Loss (MAE loss)
- **Early Stopping**: Within 10 epochs
- **Framework**: PyTorch
- **Hardware**: NVIDIA A800 80GB GPU

### Architecture
- **Model**: Pathformer with 3 Adaptive Multi-Scale Blocks (AMS Blocks)
- **Patch Sizes per Block**: 4 different patch sizes
- **Patch Size Pool**: {2, 3, 6, 12, 16, 24, 32}

### ETTh1 Dataset Specifications (from Table 1 & Appendix A.1.1)
- **Input Length (H)**: 96
- **Prediction Lengths (F)**: {96, 192, 336, 720}
- **Variables**: 7
- **Timestamps**: 17,420
- **Split Ratio**: 6:2:2 (train:val:test)
- **Features**: Multivariate (M)

### Expected Results (Table 1 - ETTh1)

| Pred Length | MSE   | MAE   |
|-------------|-------|-------|
| 96          | 0.382 | 0.400 |
| 192         | 0.440 | 0.427 |
| 336         | 0.454 | 0.432 |
| 720         | 0.479 | 0.461 |

## Key Differences Found in Existing Script

I compared the paper specifications with the existing `scripts/multivariate/ETTh1.sh` script:

### ❗ CRITICAL ISSUE: Learning Rate Discrepancy
- **Paper states**: Learning rate = 10^-3 = **0.001**
- **Existing script uses**: **0.0005** (half of paper value)

This is likely the main reason you cannot replicate results!

### Other Configuration Details from Script
The script uses these additional hyperparameters (not explicitly detailed in paper main text):
- `d_model`: 4
- `d_ff`: 64
- `train_epochs`: 30 (paper only mentions early stopping at 10 epochs)
- `patience`: 10
- `lradj`: 'TST' (learning rate adjustment strategy)
- `batch_size`: 128 (for pred_len 96, 192), 512 (for pred_len 336, 720)
- `k`: 3 (top-K patch sizes selected) for most, 2 for pred_len=720
- `residual_connection`: 1 (enabled) for pred_len 96, 192; 0 (disabled) for pred_len 336, 720
- `revin`: 1 (RevIN normalization enabled)

### Patch Size Configurations per Prediction Length
The script uses slightly different patch sizes for different prediction lengths:
- **pred_len 96, 192, 720**: `[16,12,8,32,12,8,6,4,8,6,4,2]`
- **pred_len 336**: `[16,12,8,32,12,8,6,16,8,6,4,16]`

## Implementation Notes

### What the Paper Says About Implementation
From Section 4.1 Implementation Details:
> "Pathformer utilizes the Adam optimizer (Kingma & Ba, 2015) with a learning rate set at 10−3. 
> The default loss function employed is L1 Loss, and we implement early stopping within 10 epochs 
> during the training process. All experiments are conducted using PyTorch and executed on an 
> NVIDIA A800 80GB GPU. Pathformer is composed of 3 Adaptive Multi-Scale Blocks (AMS Blocks). 
> Each AMS Block contains 4 different patch sizes. These patch sizes are selected from a pool of 
> commonly used options, namely {2, 3, 6, 12, 16, 24, 32}."

### Instance Normalization
The paper mentions (from Section 3):
> "Instance Norm (Kim et al., 2022) is a normalization technique employed to address the 
> distribution shift between training and testing data."

This corresponds to `--revin 1` in the code.

### Learning Rate Scheduler
The paper mentions "TST" learning rate adjustment (`--lradj TST`), which is not detailed in the paper but appears in the code.

## Recommendations to Replicate Paper Results

1. **✅ USE LEARNING RATE 0.001** (not 0.0005)
2. ✅ Use L1 Loss (MAE) - set `--metric mae`
3. ✅ Use Adam optimizer (default in code)
4. ✅ Enable early stopping with patience=10
5. ✅ Use the exact patch size configurations from the script
6. ✅ Ensure RevIN is enabled (`--revin 1`)
7. ✅ Use batch sizes: 128 for shorter horizons, 512 for longer
8. ✅ Follow the residual_connection settings per prediction length

## Created Training Script

I've created `train_etth1_paper_exact.bat` which:
- Uses **learning rate 0.001** (paper-specified)
- Trains all 4 prediction lengths (96, 192, 336, 720)
- Uses exact hyperparameters from paper and script
- Logs output to separate files for each prediction length

## Data Path Setup

Ensure your data is located at:
```
./dataset/ETT/ETTh1.csv
```

If your data is elsewhere, you'll need to modify the paths in the batch script.

## No Garbled Text Found

The paper PDF was cleanly converted to text. All critical information is readable:
- ✅ Mathematical equations are interpretable (with some special characters like √)
- ✅ Tables with results are complete and accurate
- ✅ Implementation details are clear
- ✅ Hyperparameters are explicitly stated
- ✅ No corruption in the methodology sections

The only minor formatting issues are decorative (like "çç√√" in diagrams), which don't affect understanding.
