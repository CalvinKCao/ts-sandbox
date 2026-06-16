# Weights & Biases (wandb) guide

## Run stem (logs, checkpoints, wandb group)

Each Slurm job submitted via [`submit_grid.sh`](../submit_grid.sh) gets one **run stem** shared across:

| Artifact | Path |
|----------|------|
| Slurm log | `results/logs/{stem}.log` |
| Checkpoints | `results/ckpts/{stem}/` |
| Eval outputs | `results/datasets/{stem}/` |
| wandb **group** | `{stem}` |

**Stem format:** `{MM-DD}-{job_id}-{dataset}-{yaml_stem}`

Example for [`configs/sweep/diff_noise_cosine.yaml`](../configs/sweep/diff_noise_cosine.yaml) on `traffic` with job id `48291`:

```
06-15-48291-traffic-diff_noise_cosine
```

- `{yaml_stem}` is the leaf config filename without `.yaml` (not `experiment.name`).
- Slurm `--job-name` is `{MM-DD}-{dataset}-{yaml_stem}` at submit time; the worker renames it to the full stem once the job id is known.

Local runs without a Slurm-style checkpoint dir fall back to `{MM-DD}-{yaml_stem}-s{seed}`.

## One group per job, one run per phase

Each YAML `phases:` entry becomes its own wandb **run** inside the job group.

| wandb field | Value |
|-------------|-------|
| `group` | Run stem (above) |
| `name` | `{group}-{phase}` (underscores in phase keys become hyphens) |
| `job_type` | Phase key from YAML, e.g. `staged_eval` |
| `tags` | Dataset name; `eval` tag on eval phases |

Example runs for the job above:

```
06-15-48291-traffic-diff_noise_cosine-staged-diffusion-pretrain
06-15-48291-traffic-diff_noise_cosine-itrans-finetune-hp
06-15-48291-traffic-diff_noise_cosine-diffusion-coarse-finetune-hp
...
06-15-48291-traffic-diff_noise_cosine-staged-eval
```

Override `wandb.group` in YAML only when you need a fixed group name; otherwise the checkpoint stem is used automatically.

## Configuration

Base defaults live in [`configs/base/binary_staged.yaml`](../configs/base/binary_staged.yaml):

```yaml
wandb:
  enabled: true
  project: ts-sandbox
  group: null   # auto from run stem
  tags: []
```

Set `WANDB_API_KEY` in the environment before running. If the key is missing or invalid, training continues without wandb.

## Resume

On first run, the pipeline writes `wandb_manifest.json` under the checkpoint dir with:

- `project`, `group`, `tags`
- `phase_runs`: map of phase key → wandb run id

On `--resume`, the manifest is the source of truth: each phase reopens its existing wandb run instead of creating a new one.

## What gets logged

Per phase (via [`wandb_utils.py`](../models/diffusion_tsf/pipeline/wandb_utils.py)):

- Full merged YAML config (and config artifact on first init)
- Phase-specific overrides in `runtime.phase` / `runtime.phase_overrides`
- HP phases: best hyperparameters in run summary (e.g. `hp/itrans_ft_best_lr`)
- Eval phases: metrics via `log_eval_metrics`; JPEG visualizations when enabled
- Git and system info in `runtime`

Stdout/stderr from Slurm jobs are captured in the log file under `results/logs/`.

## Optuna and wandb sweeps

We use **Optuna** for in-phase hyperparameter search (not `wandb.agent`). Important limitations from wandb docs and community:

| Approach | Registered in sweep DB | Sweep-like UI |
|----------|------------------------|---------------|
| `wandb.sweep()` + `wandb.agent()` | Yes | Yes |
| Optuna `WeightsAndBiasesCallback(as_multirun=True)` | No | Yes — parallel coordinates, param importance per trial |
| `wandb.sweep()` + manual `wandb.init(group=sweep_id, config=...)` | No | Partial grouping only |
| UI: select runs → “Create sweep” | New sweep from history | Yes for follow-on Bayes |

**There is no API to append arbitrary externally-launched runs to an existing wandb sweep** without the sweep agent. Bayesian / Hyperband controllers only see trials launched through `wandb.agent`.

**Practical recommendation for this repo:**

1. Keep Optuna as the optimizer inside HP phases.
2. Use wandb **groups** (job stem) and **per-phase runs** for pipeline-level tracking (current setup).
3. For trial-level wandb visibility later, add `optuna_integration.WeightsAndBiasesCallback(..., as_multirun=True)` inside HP objectives — each Optuna trial becomes a separate wandb run with params in `config`, which powers sweep-style panels without enrolling in the sweep controller.
4. To seed a new Bayesian wandb sweep from prior manual runs, use the UI: select runs → Create sweep.

References:

- [Add W&B to your code](https://docs.wandb.ai/models/sweeps/add-w-and-b-to-your-code)
- [Organize runs (grouping)](https://docs.wandb.ai/models/runs/grouping)
- [Optuna WeightsAndBiasesCallback](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.WeightsAndBiasesCallback.html)
- [wandb#11390 — sweeps without agent](https://github.com/wandb/wandb/issues/11390)
