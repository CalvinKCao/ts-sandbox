# Project Instructions

## First thing in every new conversation
1. If Linux, always `alias rm='trash-put'`.
2. Python env: use **`.venv` at repo root** (`source .venv/bin/activate`). If both `.venv/` and `venv/` exist, prefer `.venv`; the loose `venv/` copy is legacy/duplicate. Cluster Slurm jobs build or reuse venvs under `$SCRATCH` / `$SLURM_TMPDIR`, not this naming.
3. This is a WSL folder; activate the root venv before running commands.
4. Read this file before doing substantial work.

## Security
- Stay inside this project by default.
- Outside-project reads/edits are allowed only when either:
  - the file/path is explicitly attached as context, or
  - the user explicitly grants permission in chat.
- NEVER NEVER run destructive commands (for example: `rm -rf`, `git reset --hard`, force pushes, or destructive DB/file operations) without asking first and getting explicit approval.

## Alliance Canada / Slurm
For anything Slurm/Alliance Canada related, ALWAYS ALWAYS use the `/alliancecan` skill first (`.ai/skills/alliancecan/SKILL.md`).
If `/alliancecan` does not resolve the question, check `wiki_docs/` for cluster-specific details.

## Git hygiene
If you generate obvious junk, oversized artifacts, or useless throwaway files, **add ignore patterns to `.gitignore` in the same change** (or delete them) so they do not get committed.

## Wrap-up
After multi-step work, use the **`/git`** skill (`.ai/skills/git/SKILL.md`) to produce semantic commits and **push** to the tracked branch.

## ML smoke tests (not unit-test TDD)
Full test suites are usually impractical for ML. Prefer **smoke tests**: the smallest run that still exercises the real training/eval path end-to-end.

**Goal:** catch dumb failures fast (wrong hyperparams, tensor shape mismatches, broken dataloading, device issues). **Not** to prove the model learns.

**Bare minimum:** **one sample, one epoch** on a **consumer GPU**, finishing in **seconds**. If even one epoch is too heavy for the real loop, **one sample, one batch, one optimizer step** is fine as long as it hits the same forward/backward/optimizer path. Do **not** use “small % of data for multiple epochs” as a stand-in — that can still take hours.

**Design for smoke runs:** add a dedicated smoke flag and/or CLI knobs that slash workload (`--smoke-test`, `--max-samples`, `--max-steps`, batch size 1, etc.). Thread those limits through helpers (datamodule, train step, eval) via parameters or a small smoke config — avoid hard-coding full-scale-only paths. Any non-trivial train/eval entrypoint should reach a **few-second** run when smoke mode is on.

**When to run:** after any change beyond a trivial tweak you are **certain** cannot affect shapes, batching, or the train loop, run the smoke path locally before handing work back.

## Weights & Biases (wandb)
Use `wandb` for training/eval runs tied to this repo.

- Use a stable project name via `wandb.init(...)`.
- Standardize the API key env var as `WANDB_API_KEY` (do not depend on key files in this repo).
- Set `name=` to describe what changed vs prior runs; avoid generic names.
- Log all training-relevant hyperparameters in `wandb.config`.
- When resuming from checkpoint, reuse the same run with saved `run_id` and `resume="allow"` or `resume="must"`.
- After evaluation, `wandb.log` final metrics with clear names (example: `eval/mae`, `eval/mse`).
- Attach visualization artifacts to the same run (for example with `wandb.Image`), and compress/downscale images before upload.
- At run end, log the ENTIRE stdout and stderr output files to wandb as artifacts/files.

Implement these behaviors in training, eval, and visualization scripts whenever experiment code changes.

## General style
- Keep writing terse and detailed; avoid context pollution.
- Comments should sound natural, not AI-generated.
- Do not use flashy/fancy formatting in comments (for example `# ---HEADER---` or `// == PIPELINE PT A ==`).
- Use print statements sparingly; only use separators when splitting very large log sections.
- Never leave wording that makes AI authorship obvious.

# Notes space for agents

**Boundary:** Do not move, copy, merge, or relocate any content from **above** this `# Notes space for agents` heading into this section. Keep structural instructions in their existing sections upstream; use this block only for **new** session-specific notes you add (scratch reminders, one-off context).

## ts-sandbox (repo context)

ML / time-series experiments (diffusion and related). Slurm entrypoints live at repo root (`slurm_*.sh`); training code under `models/`. Alliance-specific paths: `.ai/cluster-paths.md`; use the **`/alliancecan`** skill for cluster work.

After tasks: semantic commit and push via **`/git`** — see **Wrap-up** above. Do not commit scratch outputs, huge logs, checkpoints, or throwaway scripts — see **Git hygiene** above; extend `.gitignore` in the same change when you add that kind of artifact.

## Paper ↔ codebase (NeurIPS 2025)

Source: *Geometry-Aware Time Series Forecasting: Hybrid Diffusion–Transformer Architecture with 2D Occupancy Maps* (`neurips_2025-1.pdf`).

**Motivation.** MSE/MAE regress to the conditional mean → smooth, low-frequency forecasts; neural spectral bias and “double penalty” (sharp peaks penalized in two places when slightly mis-timed) reinforce blur. Goal: keep **global trajectory** from a strong Transformer but recover **local geometry** (steps, spikes, regime edges) that point losses wash out.

**Core method.** **iTransformer** produces a coarse horizon. Series are rendered as **CDF occupancy** images (value axis binned, monotone fill per column; paper uses H=128 over ±3.5σ after per-sequence z-score, plus **31×1 vertical Gaussian blur**, σ=1). Width is a **unified time axis**: lookback + small past/future overlap K + horizon F (paper example 1024 + 8 + 192). A **conditional U-Net** runs **pixel-space diffusion** on the target occupancy, conditioned on **past occupancy** (visual concat), **ghost** occupancy from the iTrans forecast, **1D past** via a small Transformer encoder feeding **cross-attention** in the U-Net, and aux **vertical coordinate + time ramp**. Forecast is read out by aggregating mass per time column. **Latent track:** frozen **TimeSeriesVAE** (two stride-2 stages, 4× compression each on H and W, Cz=4), latent normalized by a scalar σ(μ); **shallower latent U-Net** with ghost latents for cost reduction.

**Training/eval (paper).** **Modified RealTS** synthetic pool (~10k seq) with richer seasonality and irregular phase drift → dataset fine-tune; **Optuna** (~8 trials) over LRs; **DDIM** sampling (~50 steps) at test; **GradMAE** (MAE on first differences) alongside MSE/MAE.

**Empirical story.** Diffusion improves **GradMAE everywhere** vs iTrans-only (less oversmoothing). **MSE/MAE** win on many **low–moderate variate**, structured sets (e.g. strong ETTm1 / Electricity gains, modest ETTh2); **hurts** on Exchange / Weather where a flatter mean forecast wins. **Many variates:** one shared diffusion can **fail to converge**; subset averaging helps vs grouped baselines but scaling is the main open issue.

**Repo map.** `models/diffusion_tsf/train_multivariate_pipeline.py` — end-to-end phases (synthetic pretrain, Optuna, finetune/eval), `--smoke-test`, `--profile-one-epoch`, checkpoint resume, wandb hooks. **`diffusion_model.py` / `diffusion.py` / `unet.py` / `preprocessing.py` / `guidance.py` / `dataset.py` / `realts.py` / `metrics.py` / `config.py`** — occupancy encode/decode, DDPM/DDIM, U-Net + context, iTrans guidance, data, synthetics, metrics. **`models/iTransformer/`** — baseline/coarse forecaster. **Slurm:** `slurm_etth2_compare.sh` (Killarney chained jobs, Gaussian diffusion pretrain + epoch-exported checkpoints + downstream FT; `--smoke`), `slurm_profile_one_epoch.sh`, `slurm_unet_fullvar.sh`. **`.ai/cluster-paths.md` + `/alliancecan` skill** for Alliance paths/account rules.

**Exploration branches.** `binary-exp`, `experiment/latent-only`, and `throwaway/profile-four-phase-1epoch` host alternate heads, latent-focused runs, and one-epoch profiling/smoke plumbing; `main` stays the primary integration line.

**Defaults vs paper.** `DiffusionTSFConfig` may differ from the paper’s published grid (e.g. `lookback_length`/`forecast_length`/`image_height`/`num_diffusion_steps`/`unified_time_axis`). Treat the PDF as the reference experiment; align knobs explicitly when reproducing tables.