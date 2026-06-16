## Assistant

Updating the reports with metrics from the new run logs — locating logs and reviewing the current report format.

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

Updating the report generator and creating the apples-to-apples comparison table.

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]

[REDACTED]


./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh   --datasets ETTm1,ETTm2,illness,PeMS,dalia,dynamic   --output-dir results/datasets/06-16-mmpd-maskae-grad-accum-150-lr-lo-subset   --resume   # if manifest/partials already exist from the failed init


cd $SCRATCH/ts-sandbox && git pull

# Remaining 6 datasets (resubmit MMPD only if binary jobs already running):
./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh  --datasets ETTm1,ETTm2,illness,PeMS,dalia,dynamic  --output-dir results/datasets/06-16-mmpd-maskae-grad-accum-150-lr-lo-subset  --resume   # if manifest/partials already exist from the failed init

# Or fresh:
./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh  --datasets ETTm1,ETTm2,illness,PeMS,dalia,dynamic  --output-dir results/datasets/06-16-mmpd-maskae-grad-accum-150-lr-lo-subset

Updated and pushed (`1b0e870`). Three reports now include the new runs:

**`reports/binary_mmpd_subset_compare.md`** — apples-to-apples side-by-side (main comparison table)

| Dataset | Flat subsets (3951193–99) | EMA0.99 (3951527–33) | MMPD subset (3951201–07) | Best CRPS |
|---|---|---|---|---|
| ETTh1 | 0.3060 | 0.3021 | **0.2985** | MMPD |
| ETTh2 | 0.2705 | **0.2673** | 0.2705 | EMA0.99 |
| exchange_rate | 0.1660 | 0.1684 | **0.1563** | MMPD |
| weather | 0.1776 | **0.1758** | 0.1911 | EMA0.99 |
| electricity | 0.1602 | **0.1572** | 0.1610 | EMA0.99 |
| traffic | **0.2430** | 0.2436 | 0.2515 | Flat subsets |
| solar_Alabama | 0.1945 | **0.1890** | 0.2013 | EMA0.99 |

**CRPS wins:** EMA0.99 ×4, MMPD (subset) ×2, Flat subsets ×1

**`reports/sweep_grid_report.md`** — added rows for **Flat subsets**, **Flat subsets EMA0.99**, **MMPD (subset)** (kept old **Binary flat** full-variate rows for reference)

**`reports/sweep_grid_report_leaderboard.md`** — same configs ranked per dataset; **Flat subsets EMA0.99** avg Δrank −0.33 vs baseline (on ETTh1/exchange/weather where baseline exists)

Note: merge job `3951208` failed but all 7 MMPD partials are present in `06-13-binary-mmpd-subset-compare`. Regenerate anytime with:

```bash
python reports/generate_sweep_leaderboard.py
```

[REDACTED]