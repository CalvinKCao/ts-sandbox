# Architecture snapshot (`experiment/latent-only`)

- **Training:** Latent diffusion only. Entry points: `train_ci_latent_etth2.py`, `train_ci_latent_etth1.py` (7-var CI), `train_latent_experiment.py` (1-var).
- **Model:** `TimeSeriesVAE` → `LatentDiffusionTSF` (`latent_diffusion_model.py`) → `ConditionalUNet2D` (`unet.py`) on latent tensors; `DiffusionScheduler` for DDPM/DDIM.
- **Guidance:** iTransformer (and `CIiTransformerGuidance` for multivariate batching) produces ghost 2D maps; encoded to latent for conditioning.
- **Removed from branch:** `DiffusionTSF`, pixel-space training (`train_multivariate_pipeline`), DiT backbone module, `visualize_comparison`, `storage_paths`.
