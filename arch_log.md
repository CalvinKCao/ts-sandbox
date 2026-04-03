# Architecture log (append-only)

## 2026-04-02 — experiment/latent-only
- Dropped pixel-space diffusion stack (`diffusion_model.py`, `train_multivariate_pipeline.py`, `transformer.py`, viz + storage_paths helpers).
- **Unchanged:** CI multivariate latent pipelines (`train_ci_latent_etth1.py`, `train_ci_latent_etth2.py`), `LatentDiffusionTSF`, `unet.py`, VAE, Slurm `slurm_ci_latent_*.sh`.
- Killarney setup generates `slurm_ci_latent_killarney.sh` targeting `train_ci_latent_etth2`.
