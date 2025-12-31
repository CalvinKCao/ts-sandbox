# FFT-DILATE Loss: Frequency-Selective DILATE Loss

## Overview

The FFT-DILATE loss applies DILATE loss (shape-aware) only to high-frequency components and regular MAE/MSE loss to low-frequency components. This is useful when you want shape-preservation for rapid fluctuations but not for slow trends.

## How It Works

1. **FFT Transform**: Converts predictions and targets to frequency domain using Fast Fourier Transform
2. **Frequency Separation**: Uses a percentile threshold to identify high vs low frequencies
3. **Selective Loss Application**:
   - **High frequencies** (above threshold) → DILATE loss (shape + temporal alignment)
   - **Low frequencies** (below threshold) → MAE loss (point-wise error)
4. **Energy-Weighted Combination**: Combines losses weighted by signal energy

## Usage

### Command Line Arguments

```bash
python run.py \
  --loss_type fft_dilate \
  --freq_threshold 80.0 \
  --dilate_alpha 0.5 \
  --dilate_gamma 0.01
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--loss_type` | str | 'mae' | Set to `'fft_dilate'` to enable |
| `--freq_threshold` | float | 80.0 | Percentile threshold (0-100). Higher = fewer high-freq components |
| `--dilate_alpha` | float | 0.5 | DILATE weight: shape loss vs temporal loss (0-1) |
| `--dilate_gamma` | float | 0.01 | Soft-DTW smoothing parameter |

### Frequency Threshold Examples

- `--freq_threshold 90.0` → Top 10% frequencies are "high" (very selective)
- `--freq_threshold 80.0` → Top 20% frequencies are "high" (recommended)
- `--freq_threshold 70.0` → Top 30% frequencies are "high" (aggressive)
- `--freq_threshold 50.0` → Top 50% frequencies are "high" (half and half)

## When to Use FFT-DILATE

### ✅ Good Use Cases

1. **Time series with distinct frequency components**
   - E.g., daily pattern + hourly fluctuations
   - Want to preserve shape of rapid fluctuations
   - But allow point-wise errors in slow trends

2. **When regular DILATE is too aggressive**
   - DILATE loss enforces shape matching everywhere
   - FFT-DILATE applies it only where needed (high frequencies)

3. **Noisy high-frequency signals**
   - Focus shape-preservation on meaningful rapid changes
   - Let low frequencies be handled simply

### ❌ When NOT to Use

1. **Smooth, low-frequency data**
   - If your data has no high-frequency components
   - Regular MAE/MSE is sufficient

2. **When all frequencies matter equally**
   - Use regular DILATE loss instead

3. **Very short sequences**
   - FFT needs sufficient length to separate frequencies
   - Minimum ~32-96 timesteps recommended

## Quick Start

### Example 1: Train on ETTh1 with FFT-DILATE

```batch
python run.py ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_96_fft ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --loss_type fft_dilate ^
  --freq_threshold 80.0 ^
  --dilate_alpha 0.5 ^
  --dilate_gamma 0.01 ^
  --learning_rate 0.001 ^
  --batch_size 128 ^
  --train_epochs 30
```

### Example 2: Test Different Thresholds

Run the provided batch script:
```batch
train_etth1_fft_dilate.bat
```

This trains 3 models with different frequency thresholds (70%, 80%, 90%) to compare.

## Implementation Details

### Code Location

- Main implementation: `dilate_loss_wrapper.py`
- Class: `FrequencySelectiveDilateLoss`
- Integrated into: `CombinedLoss` class

### Loss Computation

```python
total_loss = (low_freq_energy / total_energy) * MAE(low_freq) + 
             (high_freq_energy / total_energy) * DILATE(high_freq)
```

Where:
- Energy = mean squared signal amplitude
- Low/high frequencies determined by FFT + percentile threshold

### Return Values

When using FFT-DILATE, the loss function returns:
```python
(total_loss, low_freq_loss, high_freq_loss, freq_info)
```

Where `freq_info` contains:
- `high_freq_ratio`: Fraction of frequencies classified as "high"
- `shape_loss`: DILATE shape component (if applicable)
- `temporal_loss`: DILATE temporal component (if applicable)

## Testing

Test the implementation:
```bash
cd models/pathformer-main
python dilate_loss_wrapper.py
```

This runs tests with different frequency thresholds and shows:
- Loss values for each component
- Frequency separation statistics
- Comparison across thresholds

## Comparison with Other Loss Types

| Loss Type | Low Freq | High Freq | Use Case |
|-----------|----------|-----------|----------|
| **MAE** | Point-wise | Point-wise | Fast, simple, general |
| **MSE** | Point-wise | Point-wise | Penalizes large errors more |
| **DILATE** | Shape-aware | Shape-aware | Preserves temporal patterns everywhere |
| **FFT-DILATE** | Point-wise | Shape-aware | **Best of both worlds** |

## Tips for Hyperparameter Tuning

1. **Start with `freq_threshold=80.0`** (top 20% frequencies)
2. **If high-frequency shape is not preserved well**:
   - Lower threshold (e.g., 70.0) to include more frequencies
   - Increase `dilate_alpha` to emphasize shape more
3. **If low-frequency trend is poor**:
   - Raise threshold (e.g., 90.0) to be more selective
   - Check if base loss should be MSE instead of MAE
4. **Monitor the `high_freq_ratio` in output**:
   - Should typically be 0.10 to 0.30 (10-30% of frequencies)
   - Too high/low indicates threshold needs adjustment

## Performance Notes

- FFT adds minimal overhead (~2-5% slower than regular loss)
- Memory usage same as regular DILATE
- GPU-accelerated FFT (torch.fft) is very efficient
- Works with any sequence length (automatically handles different sizes)
