

### Project Structure
```text
models/diffusion_tsf/
├── config.py           # DiffusionTSFConfig: hyperparameters for model and diffusion
├── preprocessing.py    # Standardizer, TimeSeriesTo2D (encoding/decoding), VerticalGaussianBlur
├── unet.py             # ConditionalUNet2D: Default U-Net backbone
├── transformer.py      # DiffusionTransformer: DiT-style backbone option
├── diffusion.py        # DiffusionScheduler: DDPM/DDIM forward and reverse processes
├── model.py            # DiffusionTSF: Main model wrapper combining all components
├── dataset.py          # Synthetic data generation and basic 1D augmentations
├── metrics.py          # Shape-preservation and standard TSF metrics (MSE/MAE)
├── train_electricity.py # Main training script with Optuna support
└── visualize.py        # Utilities for plotting 2D representations and forecasts
```

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
    - **U-Net Path:** Uses a `ConditioningEncoder` to extract local/global features from the past image, which are then **concatenated** along the channel dimension.
    - **Transformer Path:** Uses **special tokens**. The global historical context is projected into a "context token" and prepended to the patch sequence, similar to a `[CLS]` token.

3. **Handling Non-Square Images:**
    - Since forecast length and image height are often different, the U-Net uses `F.interpolate` during upsampling steps if spatial dimensions between skip connections and upsampled features don't match exactly. The Transformer handles this via flexible patch-based sequence lengths and slicing learned positional embeddings.

4. **Diffusion Framework:** * Use the **DDPM (Denoising Diffusion Probabilistic Models)** framework.
    - Set T=1000 diffusion steps with a linear, cosine, sigmoid, or quadratic noise schedule.
    - Supports **Classifier-Free Guidance (CFG)** during training (via dropout) and inference (via `cfg_scale`).
    - Supports **DDIM (Denoising Diffusion Implicit Models)** for accelerated sampling (typically 50 steps).
    - Input to the Backbone: (Noisy Future Image + Time Embedding + Past Context Encoding).
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
- **Config Management:** All hyperparameters are centralized in `DiffusionTSFConfig` using Python dataclasses.
- Before you are finished, run a test backward and forward pass on a tiny toy dataset to ensure everything works smoothly.
- before starting, examine @models/ViTime-main/Yang et al. - 2025 - ViTime Foundation Model for Time Series Forecasting Powered by Vision Intelligence.txt and take some notes on the similar implementation. this is not the exact thing i am trying to reimplement but the whole image representation thing is inspired by it so it may give you a better idea of what im looking for. again, this prompt overrides anything that paper says though. after examining the paper, check in with me before starting - i.e. do you have any questions before starting?
