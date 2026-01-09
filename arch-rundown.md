

### Project Structure
```text
models/diffusion_tsf/
├── config.py           # DiffusionTSFConfig: hyperparameters for model and diffusion
├── preprocessing.py    # Standardizer, TimeSeriesTo2D (encoding/decoding), VerticalGaussianBlur
├── unet.py             # ConditionalUNet2D: Default U-Net backbone
├── transformer.py      # DiffusionTransformer: DiT-style backbone option
├── diffusion.py        # DiffusionScheduler: DDPM/DDIM forward and reverse processes
├── guidance.py         # GuidanceModel: Stage 1 predictors for hybrid forecasting (iTransformer, Linear, etc.)
├── model.py            # DiffusionTSF: Main model wrapper combining all components
├── dataset.py          # Synthetic data generation and basic 1D augmentations
├── metrics.py          # Shape-preservation and standard TSF metrics (MSE/MAE)
├── train_electricity.py # Main training script with Optuna support (supports all datasets)
└── visualize.py        # Utilities for plotting 2D representations and forecasts
```

### Supported Datasets

The training script supports multiple time series datasets via the `--dataset` flag:

| Dataset | Columns | Sampling | Seasonal Period | Description |
|---------|---------|----------|-----------------|-------------|
| `electricity` | 321 | Hourly | 96 | Electricity consumption from 321 clients |
| `ETTh1` | 7 | Hourly | 24 | Electricity Transformer Temperature (hourly) |
| `ETTh2` | 7 | Hourly | 24 | Electricity Transformer Temperature (hourly) |
| `ETTm1` | 7 | 15-min | 96 | Electricity Transformer Temperature (15-min) |
| `ETTm2` | 7 | 15-min | 96 | Electricity Transformer Temperature (15-min) |
| `exchange_rate` | 8 | Daily | 5 | Exchange rates of 8 countries |
| `illness` | 7 | Weekly | 52 | National illness (ILI) patient counts |
| `traffic` | 861 | Hourly | 24 | Road occupancy rates from 861 sensors |
| `weather` | 21 | 10-min | 144 | Weather observations (21 meteorological features) |

**Usage Examples:**
```bash
# Train on ETTh1 (univariate)
python train_electricity.py --dataset ETTh1

# Train on weather (multivariate, all 21 features)
python train_electricity.py --dataset weather --multivariate

# Quick test on traffic
python train_electricity.py --dataset traffic --quick --multivariate
```

### Multivariate Support

The model supports **multivariate time series forecasting** by treating multiple variables as separate image channels (similar to RGB channels in a photo).

1. **Configuration:**
   - `num_variables: int = 1` in `DiffusionTSFConfig` - Number of time series variables (1 = univariate, >1 = multivariate)
   - `--multivariate` CLI flag in `train_electricity.py` enables loading all columns from the dataset

2. **Data Format:**
   - **Univariate:** `(batch, seq_len)` → 2D: `(batch, 1, height, seq_len)`
   - **Multivariate:** `(batch, num_vars, seq_len)` → 2D: `(batch, num_vars, height, seq_len)`

3. **Architecture Impact:**
   - Each variable gets its own 2D stripe/occupancy map channel
   - Auxiliary channels (coordinate, time_ramp, time_sine) are **shared** across all variables
   - **Input channels** to backbone (initial conv): `noisy_channels + conditioning_channels`
     - `noisy_channels` = `num_variables + num_aux_channels`
     - `conditioning_channels` = `num_variables` (for `visual_concat`) OR `64` (for `vector_embedding`)
   - **Output channels** from backbone: `num_variables` (predicts noise for each variable)
   - Helper properties: `config.backbone_in_channels`, `config.num_aux_channels`, `config.visual_cond_channels`

4. **Channel Order:** `[Variable_0, Variable_1, ..., Variable_N, Vertical_Coord, Time_Ramp, Time_Sine, Guidance_0, ..., Guidance_N (if enabled)]`

---

### Hybrid "Visual Guide" Forecasting

The model supports a **two-stage hybrid forecasting** approach where a deterministic Stage 1 predictor provides coarse trend guidance, and the diffusion model refines it with texture and local details.

1. **Configuration:**
   - `use_guidance_channel: bool = False` in `DiffusionTSFConfig` - Enable/disable guidance channel
   - When enabled, adds `num_variables` extra input channels to the backbone

2. **Guidance Models (`guidance.py`):**
   - **`GuidanceModel` Protocol:** Defines the interface `get_forecast(past, forecast_length) -> future`
   - **Built-in Implementations:**
     - `LastValueGuidance`: Naive baseline - repeats last observed value
     - `LinearRegressionGuidance`: Fits linear trend on lookback window and extrapolates
     - `iTransformerGuidance`: Wrapper for pre-trained iTransformer checkpoints
   - **Factory Function:** `create_guidance_model(guidance_type, **kwargs)`

3. **Pipeline Flow:**
   ```
   Past Window → [Stage 1 Predictor] → Coarse 1D Forecast
                                            ↓
                                     [TimeSeriesTo2D + Blur]
                                            ↓
                                     "Ghost Image" (2D)
                                            ↓
   [Noisy Future + Aux Channels + Ghost Image] → [U-Net] → Refined Forecast
   ```

4. **Channel Order (with guidance):**
   ```
   [Noisy_Var_0, ..., Noisy_Var_N,   # num_variables channels
    Vertical_Coord (if enabled),      # 1 channel
    Time_Ramp (if enabled),           # 1 channel  
    Time_Sine (if enabled),           # 1 channel
    Guide_Var_0, ..., Guide_Var_N]    # num_variables channels (if use_guidance_channel)
   ```

5. **Usage:**
   ```python
   from models.diffusion_tsf.config import DiffusionTSFConfig
   from models.diffusion_tsf.model import DiffusionTSF
   from models.diffusion_tsf.guidance import LinearRegressionGuidance, iTransformerGuidance
   
   # Option 1: Use default LinearRegressionGuidance
   config = DiffusionTSFConfig(use_guidance_channel=True)
   model = DiffusionTSF(config)  # Auto-creates LinearRegressionGuidance
   
   # Option 2: Provide custom guidance model
   guidance = LinearRegressionGuidance(use_last_n=96)
   model = DiffusionTSF(config, guidance_model=guidance)
   
   # Option 3: Swap in iTransformer after loading checkpoint
   itrans = load_pretrained_itransformer(...)
   guidance = iTransformerGuidance(itrans, seq_len=512, pred_len=96)
   model.set_guidance_model(guidance)
   ```

6. **Output Includes:**
   - `guidance_2d`: The 2D ghost image used for conditioning
   - `guidance_1d`: The decoded 1D coarse forecast (for comparison/analysis)

7. **Data Split Requirements:**
   - When using iTransformer guidance, **CHRONOLOGICAL splits must be used** to prevent data leakage
   - iTransformer uses: Train (first 70%), Val (next 10%), Test (last 20%)
   - Diffusion model automatically uses the same split when `--use-guidance --guidance-type itransformer`
   - This ensures the diffusion model's training/validation data doesn't overlap with iTransformer's training data
   - Visualization script uses the TEST set (last 20%) when evaluating iTransformer-guided models

---

### Phase 1: Data Preprocessing & 2D Mapping

1. **Normalization:** Implement a standardizer that scales input windows using local mean and standard deviation.

2. **Representation Modes:** The model supports two encoding modes controlled by `representation_mode` in `DiffusionTSFConfig` and `DatasetConfig`:
   - **PDF Mode (default, "stripe/one-hot"):** Original probability density representation
   - **CDF Mode ("occupancy map"):** Cumulative distribution occupancy representation

3. **2D Encoding (PDF Mode - The "Stripe" Method):** Map numerical values to a grid of height H=128 and width W=L (sequence length).

    - For each time step t, calculate the pixel index: yt​=clip(σxt​−μ​⋅MSH​+2H​,0,H−1), where MS (Maximum Scale) is 3.5.

    - **Representation:** Create a binary image where only the pixel at (t,yt​) is 1.

4. **2D Encoding (CDF Mode - Occupancy Method):**
    - Normalize with existing standardizer, clamp to `[-max_scale, max_scale]`.
    - Map each time step to an integer bin `y` in `[0, H-1]`.
    - Fill all pixels `0..y` as 1.0 (occupancy), rest 0.0.
    - **Why:** Occupancy/CDF mode makes the mass cumulative per column; blur turns the hard edge into a differentiable sigmoid-like transition while preserving the original stripe path for comparison.

5. **Gaussian Blur (Both Modes):** Apply a **1D Vertical Gaussian blur** (kernel size: Height=31, Width=1) or a highly anisotropic 2D blur (e.g., 31x1). This must strictly blur **only along the value axis** to create a smooth probability density (PDF) or occupancy boundary (CDF) without smearing the temporal geometric patterns (W-shapes, sharp edges) across time steps.

    - **PDF Mode:** Creates probability density distribution
    - **CDF Mode:** Softens the step into a smooth boundary; for diffusion, the blurred occupancy is clamped to `[0, 1]` and shifted to `[-1, 1]` (no extra gain factor).

---

### Phase 2: Architecture - Backbones & Conditioning

1. **Model Selection:** Controlled by `model_type` (default: `"unet"`). 
   - **U-Net:** Uses Residual blocks, `GroupNorm`, and skip-connections. Best for preserving local spatial hierarchies.
   - **Transformer (DiT):** A patch-based Transformer encoder. Splits the 2D image into patches, flattens them, and adds learned positional embeddings.

2. **Conditioning (Past Context):**
    - **U-Net Path:** Controlled by `conditioning_mode` in `DiffusionTSFConfig`.
        - **`visual_concat` (Default):** Directly concatenates the past 2D image (ground truth past + zeros for future) to the input along the channel dimension. This allows the model to explicitly "see" the past trajectory pixels. Bypasses `ConditioningEncoder` to save compute.
        - **`vector_embedding`:** Uses a separate `ConditioningEncoder` to extract local/global features from the past image, which are then concatenated along the channel dimension. (Original backward-compatible mode).
    - **Transformer Path:** Uses **special tokens**. The global historical context is projected into a "context token" and prepended to the patch sequence, similar to a `[CLS]` token.

3. **Handling Non-Square Images:**
    - Since forecast length and image height are often different, the U-Net uses `F.interpolate` during upsampling steps if spatial dimensions between skip connections and upsampled features don't match exactly. The Transformer handles this via flexible patch-based sequence lengths and slicing learned positional embeddings.

4. **Spatial & Temporal Coordinate Channels:**
    - **Vertical Coordinate Channel (`use_coordinate_channel`):** Adds a channel with a gradient from +1 (top) to -1 (bottom), providing the backbone with explicit value-axis position awareness.
    - **Horizontal Time Channels:** Two independently controllable channels for explicit temporal position awareness (fixes "phase drift" in U-Net forecasts):
        - **Linear Ramp (`use_time_ramp`):** Values from -1.0 (start of window) to +1.0 (end of window). Tells the model "how far along" it is in the forecast (a "progress bar").
        - **Sine Wave (`use_time_sine`):** `sin(2π * t / seasonal_period)` where `t` is the column index. Provides periodic/seasonal awareness (a "clock"). Default `seasonal_period=96` for hourly data with daily cycles.
    - **Value Channel (`use_value_channel`):** Shows the last `forecast_length` values from the past as a simple 1D→2D representation. Each column contains the normalized value at that timestep broadcast across all rows (height). For example, if `past=[v0..v511]` and `forecast_len=96`, the value channel shows `[v416..v511]` - the 96 timesteps immediately before the forecast. This provides recent value context without leaking future information (same values used at train and inference time).
    - These channels are concatenated to the noisy image and past conditioning before being fed to the backbone.
    - **Channel Order:** `[Noisy_Image, Vertical_Coord (if enabled), Time_Ramp (if enabled), Time_Sine (if enabled), Value_Channel (if enabled)]`

5. **Diffusion Framework:** * Use the **DDPM (Denoising Diffusion Probabilistic Models)** framework.
    - Set T=1000 diffusion steps with a linear, cosine, sigmoid, or quadratic noise schedule.
    - Supports **Classifier-Free Guidance (CFG)** during training (via dropout) and inference (via `cfg_scale`).
    - Supports **DDIM (Denoising Diffusion Implicit Models)** for accelerated sampling (typically 50 steps).
    - Input to the Backbone: (Noisy Future Image + Coordinate Channels + Time Channels + Diffusion Timestep Embedding + Past Context Encoding).
    - Output: Predicted noise ϵθ​.
        

---

### Phase 3: Training and Inference

1. **Loss Function:**
    - **L2 Loss:** Computed between the added noise and predicted noise in the 2D space.
    - **EMD Loss:** Optional Earth Mover's Distance loss weighted by `emd_lambda` to encourage better probability mass distribution.

2. **Reverse Process (Sampling):**
    - **DDPM:** Full iterative denoising (T=1000).
    - **DDIM:** Accelerated sampling with fewer steps (e.g., 50) and a stochasticity parameter `eta` (0 = deterministic).
    - **CFG Sampling:** Uses the formula `noise_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)` to amplify the conditioning signal.

3. **Decoding (2D to 1D):**
    - After T steps, you will have a 2D probability/occupancy map.
    - **PDF Mode:** Uses **Softmax Expectation**. For each column, apply softmax (sharpened by `decode_temperature`) and calculate the expected value: xt′​=∑i=0H−1​P(i)⋅Value(i).
    - **CDF Mode:**
        - Bring diffusion output back to `[0, 1]` via `(x + 1)/2`, clamp non-negative.
        - Optionally apply horizontal **decode smoothing** if enabled.
        - Sum each column; clamp to `[0, H]`, normalize by `H`, then map back to the normalized value range `[-max_scale, max_scale]`.
    - Final denormalization to the original scale uses the mean and std stored in the `Standardizer`.

### Phase 4: Implementation Requirements

- **Framework:** PyTorch.
- **Augmentations:**
    - **1D (raw series):** Scaling, time-warp (interpolation/stretch/compress), and time-stretch (holds/averages then resizes back).
    - **2D (image space):** **Cutout** (random rectangular masks) applied to the 2D representations to improve robustness.
- **Performance Metric:**
    - In addition to MSE/MAE, use a **Shape-Preservation Metric**.
    - This compares **first-order gradients** (xt​−xt−1​) of the prediction vs. ground truth.
    - Metrics include Gradient MAE, Pearson Correlation of gradients, and Sign Agreement.
- **Config Management:** All hyperparameters are centralized in `DiffusionTSFConfig` using Python dataclasses. Key flags:
    - `conditioning_mode`: Selects between `"visual_concat"` (pixel-level visibility) and `"vector_embedding"` (encoded features).
    - `use_guidance_channel`: Enable hybrid forecasting with Stage 1 predictor guidance (default: False).
- Before you are finished, run a test backward and forward pass on a tiny toy dataset to ensure everything works smoothly.
- before starting, examine @models/ViTime-main/Yang et al. - 2025 - ViTime Foundation Model for Time Series Forecasting Powered by Vision Intelligence.txt and take some notes on the similar implementation. this is not the exact thing i am trying to reimplement but the whole image representation thing is inspired by it so it may give you a better idea of what im looking for. again, this prompt overrides anything that paper says though. after examining the paper, check in with me before starting - i.e. do you have any questions before starting?
