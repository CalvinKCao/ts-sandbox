# SimDiff baseline (L=H=96)

Reproduces [Dear-Sloth/SimDiff](https://github.com/Dear-Sloth/SimDiff) on **ETTh1** and **exchange_rate** with `seq_len=pred_len=96`, using paper hyperparameters from `script/etth1.sh` and `script/exchange.sh` (MoM, cosine diffusion, N.I., DPM-Solver, etc.).

## Metrics comparable to `slurm_experimental_4phase.sh`

| Protocol | Detail |
|----------|--------|
| Splits | iTransformer/TimesNet borders (ETT month splits; exchange 70/10/20) |
| Scaling | Train-split z-score per variate (`StandardScaler` / `load_dataset`) |
| Reported MSE/MAE | On **scaled** values (`inverse=False`), same as our diffusion pipeline eval |
| Horizon | 96 steps (user override; paper uses dataset-specific L/H) |

Run comparable eval after training:

```bash
source .venv/bin/activate
python baselines/simdiff/eval_comparable.py --dataset ETTh1
python baselines/simdiff/eval_comparable.py --dataset exchange_rate
```

Outputs: `results_simdiff/{dataset}_96_96_eval.json` with `ts_sandbox_metrics` and `simdiff_native_loader_metrics`.

## Train

```bash
bash scripts/simdiff/etth1_96_96.sh
bash scripts/simdiff/exchange_96_96.sh
# or both:
bash scripts/simdiff/run_both_96_96.sh
```

Cluster: `sbatch slurm_simdiff_repro.sh` (submits ETTh1 + exchange jobs).

## Paper reference (Table 2 MSE, multivariate)

| Dataset | SimDiff MSE (paper) |
|---------|---------------------|
| ETTh1 | 0.394 |
| Exchange | 0.299 |

Paper uses dataset-tuned L/H (e.g. ETTh1 336→168, exchange 96→14); we fix **96→96** for alignment with the experimental pipeline.
