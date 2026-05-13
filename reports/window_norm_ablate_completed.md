# Window norm ablation — completed runs

Frozen hyperparameters (from prior default Slurm runs), **20% random test subset** for eval, full pipeline otherwise. Arms: **wn-a** — per-window norm off, guidance penalty 0; **wn-b** — window norm on, uniform guidance penalty 0.03; **wn-c** — window norm on, spatial ramped guidance penalty (max 0.2).

**Filter:** job directories with Slurm id ≥ `3562485` only.

**Results directory scanned:** `/home/cao/ts-sandbox/results`  

**Completed (6):** 05-13-3562485-wn-a-ETTh1, 05-13-3562486-wn-b-ETTh1, 05-13-3562487-wn-c-ETTh1, 05-13-3562488-wn-a-ETTh2, 05-13-3562500-wn-a-exchange-rate, 05-13-3562502-wn-c-exchange-rate

## Not complete (failed, timed out, or still running)
- `05-13-3562489-wn-b-ETTh2` — failed early (env/import or crash before pipeline end)
- `05-13-3562490-wn-c-ETTh2` — failed early (env/import or crash before pipeline end)
- `05-13-3562491-wn-a-ETTm1` — failed early (env/import or crash before pipeline end)
- `05-13-3562492-wn-b-ETTm1` — failed early (env/import or crash before pipeline end)
- `05-13-3562493-wn-c-ETTm1` — no completion marker (timeout or still running)
- `05-13-3562494-wn-a-ETTm2` — no completion marker (timeout or still running)
- `05-13-3562495-wn-b-ETTm2` — no completion marker (timeout or still running)
- `05-13-3562496-wn-c-ETTm2` — no completion marker (timeout or still running)
- `05-13-3562497-wn-a-weather` — no completion marker (timeout or still running)
- `05-13-3562498-wn-b-weather` — no completion marker (timeout or still running)
- `05-13-3562499-wn-c-weather` — no completion marker (timeout or still running)
- `05-13-3562501-wn-b-exchange-rate` — failed early (env/import or crash before pipeline end)

## Per-run metrics (eval)
### `05-13-3562485-wn-a-ETTh1` *(Duration: 1h 13m 12s)*

| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |
|---------|-----|------------|------------|---------------|---------------|--------------------|
| ETTh1 | wn-a | 0.4365 | 0.4311 | 0.5158 | 0.4777 | -18.17% |

W&B: https://wandb.ai/calvincao/diffusion-tsf/runs/512dshn3

### `05-13-3562486-wn-b-ETTh1` *(Duration: 1h 13m 34s)*

| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |
|---------|-----|------------|------------|---------------|---------------|--------------------|
| ETTh1 | wn-b | 0.4365 | 0.4311 | 0.5025 | 0.4735 | -15.12% |

W&B: https://wandb.ai/calvincao/diffusion-tsf/runs/wexzyzaa

### `05-13-3562487-wn-c-ETTh1` *(Duration: 1h 13m 17s)*

| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |
|---------|-----|------------|------------|---------------|---------------|--------------------|
| ETTh1 | wn-c | 0.4365 | 0.4311 | 0.5520 | 0.4763 | -26.46% |

W&B: https://wandb.ai/calvincao/diffusion-tsf/runs/x4txvpas

### `05-13-3562488-wn-a-ETTh2` *(Duration: 1h 12m 34s)*

| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |
|---------|-----|------------|------------|---------------|---------------|--------------------|
| ETTh2 | wn-a | 0.3304 | 0.3684 | 0.3388 | 0.3771 | -2.54% |

W&B: https://wandb.ai/calvincao/diffusion-tsf/runs/nk16hdfx

### `05-13-3562500-wn-a-exchange-rate` *(Duration: 0h 58m 33s)*

| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |
|---------|-----|------------|------------|---------------|---------------|--------------------|
| exchange_rate | wn-a | 0.1589 | 0.2844 | 0.1755 | 0.3227 | -10.45% |

W&B: https://wandb.ai/calvincao/diffusion-tsf/runs/34nxi393

### `05-13-3562502-wn-c-exchange-rate` *(Duration: 0h 57m 57s)*

| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |
|---------|-----|------------|------------|---------------|---------------|--------------------|
| exchange_rate | wn-c | 0.1589 | 0.2844 | 0.1275 | 0.2474 | +19.76% |

W&B: https://wandb.ai/calvincao/diffusion-tsf/runs/maj2deor

## ETTh1 — compare arms (same dataset)

| Arm | Diffusion MSE | Diffusion MAE | iTrans MSE | iTrans MAE |
|-----|-----------------|---------------|------------|------------|
| wn-a | 0.5158 | 0.4777 | 0.4365 | 0.4311 |
| wn-b | 0.5025 | 0.4735 | 0.4365 | 0.4311 |
| wn-c | 0.5520 | 0.4763 | 0.4365 | 0.4311 |

## Qualitative plots

Regenerate anytime from repo root (GPU recommended), with checkpoints under each run’s `ckpts/`:

```bash
MIN_JOB_ID=3562485 ./utils/window_norm_ablate/visualize_completed.sh
```

Plots land under `results/viz/window_norm_ablate/<run-dir>/` (ignored by git via `**/results/`).

### Comparison PNGs (this workspace)

- `05-13-3562485-wn-a-ETTh1` → `results/viz/window_norm_ablate/05-13-3562485-wn-a-ETTh1/comparison_ETTh1.png`
- `05-13-3562486-wn-b-ETTh1` → `results/viz/window_norm_ablate/05-13-3562486-wn-b-ETTh1/comparison_ETTh1.png`
- `05-13-3562487-wn-c-ETTh1` → `results/viz/window_norm_ablate/05-13-3562487-wn-c-ETTh1/comparison_ETTh1.png`
- `05-13-3562488-wn-a-ETTh2` → `results/viz/window_norm_ablate/05-13-3562488-wn-a-ETTh2/comparison_ETTh2.png`
- `05-13-3562500-wn-a-exchange-rate` → `results/viz/window_norm_ablate/05-13-3562500-wn-a-exchange-rate/comparison_exchange_rate.png`
- `05-13-3562502-wn-c-exchange-rate` → `results/viz/window_norm_ablate/05-13-3562502-wn-c-exchange-rate/comparison_exchange_rate.png`

