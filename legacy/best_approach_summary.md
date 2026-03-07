# Diffusion-TSF: Best Approach & Research Summary

This document summarizes the current state-of-the-art approach in this repository (implemented in `slurm_train_7var.sh` and `train_7var_pipeline.py`) and catalogs the various experiments that resulted in performance degradation or excessive computational cost.

---

## 1. Current Best Approach: The 7-Variate Guided Pipeline

The most successful configuration currently utilizes a **Hybrid Visual-Guided Diffusion** model trained on partitioned 7-variate subsets.

### Architecture & Hyperparameters
- **Backbone**: 2D Conditional U-Net.
  - **Channels**: `[64, 128, 256]` (3 levels).
  - **Residual Blocks**: 2 per level.
  - **Attention**: Bottleneck only (level 2) to manage memory.
- **2D Representation**: **CDF / Occupancy Map Mode**.
  - **Dimensions**: $128$ (Height) $\times 1216$ (Total Width: $1024$ Lookback + $192$ Forecast).
  - **Encoding**: Values are mapped to a vertical grid; all pixels from the bottom up to the value are filled (1.0).
  - **Blur**: 1D Vertical Gaussian Blur (Kernel: $31$, Sigma: $1.0$) applied strictly along the value axis.
- **Conditioning**:
  - **Visual Concat**: The past 2D representation is concatenated directly to the U-Net input.
  - **Hybrid 1D Context**: Cross-attention on raw 1D time-series features (normalized values + time index).
  - **Stage 1 Guidance**: iTransformer-generated "ghost images" are concatenated to the future noise to provide trend priors.
- **Inference**:
  - **DDIM Sampling**: 50 steps (deterministic, $\eta=0$).
  - **Optimized Scheme**: Diffuse only the forecast horizon ($192$ width) during inference, rather than the full $1216$ window.

### Training Strategy
1. **Synthetic Pre-training**: 1M samples generated via multivariate augmentation.
2. **Multi-Stage Pre-training**: First train iTransformer (Stage 1), then train Diffusion (Stage 2) using iTransformer guidance.
3. **Subset Partitioning**: Large datasets (e.g., Traffic with 862 variables) are partitioned into random, non-overlapping 7-variate subsets (capped at 5 subsets per dataset) to leverage the 7-variate pretrained backbone without the information loss of clustering.

---

## 2. Failed, Degraded, or Abandoned Approaches

The following features were implemented but found to be suboptimal or counterproductive.

| Feature / Approach | Result | Reason for Abandonment |
| :--- | :--- | :--- |
| **PDF (Stripe) Representation** | **Degraded** | Sparse "one-hot" stripes are harder for the diffusion model to learn than continuous occupancy boundaries (CDF). |
| **Joint Lookback + Forecast Diffusion** | **Too Expensive** | Diffusing the entire window ($1216$ width) at inference time is extremely slow and provides marginal continuity gain over the optimized Forecast-only mode. |
| **Lower Resolutions ($H=64$)** | **Degraded** | $64$ vertical pixels result in significant quantization error, losing fine-grained texture in the time series. |
| **CCM (Channel Clustering)** | **Degraded** | Aggregating hundreds of channels into 7 "super-channels" loses too much variable-specific information. Partitioning is superior. |
| **Vector Embedding Conditioning** | **Suboptimal** | Using a separate encoder for past context was less effective than direct pixel-level concatenation (`visual_concat`). |
| **DiT (Transformer) Backbone** | **Suboptimal** | Transformers required more data and compute to reach the same performance as the U-Net's spatial inductive bias for this task. |
| **Time Ramp / Sine Channels** | **Inconsistent** | Explicit temporal coordinate channels occasionally caused "overfitting" to the training window phase rather than learning the dynamics. |

---

## 3. Normalization Strategies

The pipeline employs a multi-level normalization strategy to ensure stability and cross-dataset compatibility.

### Time-Series Normalization (Instance / Standardization)
- **Local Standardization**: Each lookback window is normalized independently using its own mean ($\mu$) and standard deviation ($\sigma$).
- **Future Alignment**: The forecast horizon is normalized using the statistics derived from the *lookback window* (not the future itself) to prevent data leakage.
- **Why**: This acts similarly to **Instance Normalization**, removing window-specific scale differences and making the model "scale-invariant."

### Architectural Normalization
- **U-Net (Diffusion)**: Uses **GroupNorm** (default: `num_groups=32`).
  - **Why**: GroupNorm is more stable for small batch sizes (often 4-16 in this project) than BatchNorm and preserves spatial dependencies better in 2D representations.
- **iTransformer (Stage 1)**: Uses **LayerNorm**.
  - **Why**: Standard for Transformer architectures to normalize across the feature dimension.

---

## 4. Detailed Pipeline Flowchart

```text
+---------------------------------------------------------------------------------+
|                                PRE-TRAINING PHASE                               |
+---------------------------------------------------------------------------------+
| [Synthetic Data Gen] -> 1M Samples (7-variate, Lookback 1024, Forecast 192)     |
|          |                                                                      |
|          v                                                                      |
| [Stage 1: iTransformer] -> [LayerNorm] -> Pre-train on 1M samples               |
|          |                                                                      |
|          v                                                                      |
| [Stage 2: Diffusion]    -> [GroupNorm] -> Pre-train on 1M samples               |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                                FINE-TUNING PHASE                                |
+---------------------------------------------------------------------------------+
| [Dataset Partitioning]  -> Split high-variate datasets into 7-var subsets       |
| [HP Tuning]             -> Tune LR and Batch Size per subset                    |
| [Fine-tune]             -> Adjust pre-trained weights to real-world data        |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                                INFERENCE PIPELINE                               |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  [ 1D Lookback ] --(Calc Local Mean/Std)--> [ Stats (mu, sigma) ]               |
|          |                  |                                                   |
|          |                  +--------------(Apply to Forecast)-----------+       |
|          |                                                               |       |
|          v                                                               v       |
|  [ Normalize ] <-------------------------------------------+    [ Target Window ]|
|  (Instance Norm)                                           |             |       |
|          |                                                 |             |       |
|          v                                                 |             v       |
|  [ TimeSeriesTo2D (CDF) ]                                  |    [ Final 1D Pred ]|
|          |                                                 |             ^       |
|          v                                                 |             |       |
|  [ Vertical Gaussian Blur ]                                |      (Denormalize)  |
|          |                                                 |             |       |
|          v                                                 |             |       |
|  [ iTransformer ] -------> [ Coarse 1D Forecast ]          |             |       |
|  (LayerNorm)               [ (Normalized)       ]          |             |       |
|          |                            |                    |             |       |
|          |                            v                    |             |       |
|          |                  [ TimeSeriesTo2D (CDF) ]       |             |       |
|          |                            |                    |             |       |
|          |                            v                    |             |       |
|          |                  [ "Ghost Image" Guide ]        |             |       |
|          |                            |                    |             |       |
|          +----------------------------|--------------------------+       |       |
|                                       v                          |       |       |
|  [ Noise ] -> [ DDIM Sampler ] <-> [ U-Net Backbone ]            |       |       |
|                    ^                  | (Conditioning)           |       |       |
|                    |                  | - [GroupNorm]            |       |       |
|                    |                  | - Visual Concat          |       |       |
|                    |                  | - Hybrid 1D Context      |       |       |
|                    |                  | - Vertical Coord Grid    |       |       |
|                    |                  +--------------------------+       |       |
|                    v                                                     |       |
|          [ 2D Occupancy Map ]                                            |       |
|                    |                                                     |       |
|                    v                                                     |       |
|          [ 2D to 1D Decoder ] (Column-wise Mean/Sum) ---------------------+       |
|                                                                                 |
+---------------------------------------------------------------------------------+
```
