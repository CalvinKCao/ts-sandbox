# Architecture log (append-only)

## 2026-04-04 — CI latent multi-dataset finetune
- `train_ci_latent_etth2.py`: `--dataset` (ETTh1/h2, ETTm1/m2, exchange_rate), `--shared-ckpt-dir` (stages 1–2), `--run-ckpt-dir` (per-`run_tag` finetunes), `--exchange-seed` / `--variate-indices`, stage-4 **default 12 Optuna trials** with wider search (lr, batch 1–16, wd, grad_clip, max_epochs, patience, min_delta).
- `latent_experiment_common.py`: registry rows add ETTm1/m2 + exchange_rate + `itransformer` embed `freq` (`t` for 15-min).
- `slurm_ci_latent_multidataset.sh` replaces separate per-ETT Slurm for the multi-benchmark; removed `slurm_ci_latent_etth1.sh` (ablation still via `train_ci_latent_etth1.py`).

## 2026-04-02 — experiment/latent-only
- Dropped pixel-space diffusion stack (`diffusion_model.py`, `train_multivariate_pipeline.py`, `transformer.py`, viz + storage_paths helpers).
- **Unchanged:** CI multivariate latent pipelines (`train_ci_latent_etth1.py`, `train_ci_latent_etth2.py`), `LatentDiffusionTSF`, `unet.py`, VAE, Slurm `slurm_ci_latent_*.sh`.
- Killarney setup generates `slurm_ci_latent_killarney.sh` targeting `train_ci_latent_etth2`.

## 2026-04-02 — CI latent Slurm split jobs
- `slurm_ci_latent_common.inc.sh`: shared venv/dataset/`run_py` for CI latent Slurm scripts.
- `slurm_ci_latent_bootstrap.sh`: stages 0-2 only (2-day wall default).
- `slurm_ci_latent_finetune_dataset.sh`: one dataset, stages 3-4; `CI_DATASET` (+ optional `CI_EXCHANGE_SEED`).
- `submit_ci_latent_multidataset_jobs.sh`: login-node driver — `sbatch` bootstrap + five `afterok` finetune jobs (parallel). Monolithic `slurm_ci_latent_multidataset.sh` kept.
