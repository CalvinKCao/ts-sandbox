# Onboarding Guide for Future AI Agents

## Project Summary
This project implements a **Diffusion-based Time Series Forecasting (TSF)** model, inspired by the ViTime paper. It treats time series forecasting as a computer vision problem by converting 1D time series into 2D "stripe" images and using a U-Net backbone with DDPM/DDIM diffusion to generate future predictions.

## Key Goals
- **Multivariate Forecasting:** Supports forecasting multiple variables simultaneously.
- **Universal Pre-training:** A 4-phase training pipeline:
    1.  Pre-train iTransformer (Stage 1 predictor) on synthetic data.
    2.  Pre-train Diffusion model (Stage 2) on synthetic data, using iTransformer guidance.
    3.  Fine-tune iTransformer on real data.
    4.  Fine-tune Diffusion model on real data.
- **Rich Synthetic Data:** Uses a sophisticated multivariate augmentation system to generate realistic synthetic pre-training data.

## File Structure & Key Components

### Models
- `models/diffusion_tsf/`: Core diffusion model logic.
    - `diffusion_model.py`: Main `DiffusionTSF` class. Handles 2D encoding/decoding, forward pass (training), and generation. **Key Architectural Change:** Uses a unified "L+F" (Lookback + Forecast) time axis for channel representation.
    - `unet.py`: Conditional U-Net backbone (2D).
    - `transformer.py`: Alternative DiT backbone (not primary).
    - `dataset.py`: `ElectricityDataset` for real data and `get_synthetic_dataloader`.
    - `realts.py`: Synthetic data generator classes.
    - `augmentation.py`: **New** multivariate augmentation logic (Algorithm 1, 2, 3 from paper).
    - `train_universal.py`: **Main Entry Point**. Orchestrates the 4-phase training pipeline.
    - `config.py`: Configuration dataclasses.

- `models/iTransformer/`: The iTransformer model used as a Stage 1 predictor (Guidance).
    - `model/iTransformer.py`: The model architecture.
    - `experiments/exp_long_term_forecasting.py`: Training wrapper.

### Scripts
- `train_universal.py`: The master script to run the full pipeline.
    - Usage: `python -m models.diffusion_tsf.train_universal --dataset ETTh2`
    - Smoke Test: `python -m models.diffusion_tsf.train_universal --dataset ETTh2 --smoke-test` (Runs minimal configuration for debugging).

## Architecture Highlights

### 1. Unified L+F Channel Scheme
Instead of stacking "guidance" and "lookback" as separate channels with misaligned time axes, we now use a single time axis of length `Lookback + Forecast`.
- **Input Canvas:** Shape `(Batch, Channels, Height, L+F)`.
- **Past Part (0 to L):** Contains the ground truth past data.
- **Future Part (L to L+F):** Contains the noisy future (during training/inference).
- **Auxiliary Channels:** Coordinate grids, time ramps, and sine waves are injected across the full `L+F` width.
- **Guidance:** Stage 1 forecasts are converted to 2D and placed in the "Future" part of the canvas.

### 2. Multivariate Augmentation
Located in `models/diffusion_tsf/augmentation.py`.
- Generates synthetic multivariate time series by coupling independent random processes.
- Uses "Impact Functions" to model causal effects between variables.
- Used by `RealTS` dataset when `num_variables > 1`.

## Gotchas & Tips
- **Module Imports:** When running scripts inside `models/diffusion_tsf/`, always run from the project root using `python -m models.diffusion_tsf.script_name` to avoid `ImportError` and name collisions (especially with `iTransformer`'s `model` module).
- **iTransformer Path:** `train_universal.py` manually appends `models/iTransformer` to `sys.path`. Be careful with relative imports in that submodule.
- **Smoke Test:** Always run the smoke test before full training to verify pipeline integrity. It uses tiny batch sizes and models to run quickly on CPU.
- **OOM Issues:** The "L+F" scheme creates wide images (e.g., 512+96 = 608 pixels wide). This consumes significant GPU memory. We use `attention_levels=[2]` (only bottleneck) and small batch sizes to mitigate this.

## Recent Changes
- Renamed `model.py` to `diffusion_model.py` to avoid conflict with `iTransformer/model`.
- Implemented `train_universal.py` replacing the shell script approach.
- Fixed `DiffusionTSF` to properly pad conditioning inputs to match the new `L+F` width.
