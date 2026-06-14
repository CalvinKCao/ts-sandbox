# YAML-First Sweep Leaderboard

Probabilistic metrics from `dpmpp` sampler with `20` steps. Baseline is `sweep_baseline` (fixed 3e-5 LR, linear noise, epsilon target). **Discrete** is ordinal D3PM CE (`ordinal_d3pm_staged`). **MAE Discrete** is expectation-MAE + uniform `1/H` anchor (`ordinal_d3pm_mae_staged_subsets`). **Binary flat** is flat `0.5` XOR anchor on full variates (`binary_anchor_stationary_flat`). **Flat subsets** family uses ETTh1-capped variate subsets (`binary_anchor_stationary_flat_subsets`). EMA reuse sweep: `diffusion_ema_decay` ∈ {0.90, 0.95, 0.98, 0.995, 0.999}. Grad-accum reuse sweep: effective batch {1.25×, 1.5×, 2.0×} probed max micro-batch. **MS tune** is Optuna `max_scale` search (`hp_max_scale_tuning`, baseline-fixed other HPs). **MMPD (subset)** from `06-13-binary-mmpd-subset-compare` (same subsets as flat runs, 20 samples, full test). Legacy **MMPD** from `06-12-sweep-subset-mmpd` where subset MMPD is unavailable.

**Pre-fix invalid runs** (pipeline bug, fixed in main): `hp_max_scale_tuning` jobs `3943934`–`3943937` never searched `max_scale` (matched `max_scale_by_dataset` by accident). `hp_lr_cosine_warmup2` / `hp_lr_cosine_warmup5` jobs `3943882`–`3943887`, `3943924`–`3943927` never applied cosine+warmup LR scheduler (metrics ≈ `sweep_baseline`). Re-submit: `./submit_hp_max_scale_tuning.sh`, `./submit_hp_lr_cosine_warmup.sh`.

## Average Δrank vs baseline

Δrank = config rank − `sweep_baseline` rank per dataset (negative = better anchor MSE). Avg Δrank averages over datasets where the config ran.

| Rank | Config | avg Δrank | ETTh1 Δrank | ETTh2 Δrank | ETTm1 Δrank | exchange_rate Δrank | weather Δrank | electricity Δrank | traffic Δrank | solar_Alabama Δrank | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Flat subsets EMA0.90** | -6.00 | -3 | — | — | -8 | -7 | — | — | — | **OK** |
| 2 | **Flat subsets EMA0.95** | -3.00 | +4 | — | — | -4 | -9 | — | — | — | **OK** |
| 3 | `diff_ema_decay_099` | -2.75 | -6 | — | -6 | +6 | -5 | — | — | — | **OK** |
| 4 | `hp_lr_cosine_warmup2` | -2.00 | -2 | — | -2 | -2 | -2 | — | — | — | **pre-fix invalid** |
| 5 | **Flat subsets EMA0.98** | -1.67 | +6 | — | — | -3 | -8 | — | — | — | **OK** |
| 6 | `hp_lr_cosine_warmup5` | -1.00 | -1 | — | -1 | -1 | -1 | — | — | — | **pre-fix invalid** |
| 7 | `hp_beta_end_04` | -0.75 | +7 | — | +7 | -11 | -6 | — | — | — | **OK** |
| 8 | **Flat subsets EMA0.99** | -0.67 | -5 | — | — | +7 | -4 | — | — | — | **OK** |
| 9 | **Flat subsets EMA0.995** | +0.00 | +16 | — | — | -6 | -10 | — | — | — | **OK** |
| 10 | `hp_cfg_dropout_02` | +0.00 | +2 | — | +4 | -9 | +3 | — | — | — | **OK** |
| 11 | `sweep_baseline` | +0.00 | 0 | — | 0 | 0 | 0 | — | — | — | **OK** |
| 12 | **Flat subsets** | +1.00 | +1 | — | — | +1 | +1 | — | — | — | **OK** |
| 13 | **Flat subsets accum2.0x** | +1.67 | +8 | — | — | -5 | +2 | — | — | — | **OK** |
| 14 | `h16_16_16` | +3.00 | +3 | — | +1 | +4 | +4 | — | — | — | **OK** |
| 15 | `hp_anchor_lambda_090` | +3.00 | +17 | — | -5 | -12 | +12 | — | — | — | **OK** |
| 16 | **MS tune** | +3.00 | +11 | — | +3 | -7 | +5 | — | — | — | **pre-fix invalid** |
| 17 | **Flat subsets accum1.25x** | +3.33 | +5 | — | — | +8 | -3 | — | — | — | **OK** |
| 18 | `hp_ctxbias_neg01` | +4.00 | +10 | — | +5 | -10 | +11 | — | — | — | **OK** |
| 19 | `hp_anchor_lambda_095` | +4.00 | +12 | — | -7 | +2 | +9 | — | — | — | **OK** |
| 20 | **Flat subsets EMA0.999** | +4.67 | +13 | — | — | +14 | -13 | — | — | — | **OK** |
| 21 | `diff_noise_cosine` | +6.25 | +19 | — | +6 | +12 | -12 | — | — | — | **OK** |
| 22 | `hp_num_steps_800` | +8.25 | +14 | — | -4 | +10 | +13 | — | — | — | **OK** |
| 23 | `hp_num_steps_1200` | +8.25 | +9 | — | +2 | +5 | +17 | — | — | — | **OK** |
| 24 | `hp_dit_dropout_01` | +9.00 | -4 | — | +8 | +17 | +15 | — | — | — | **OK** |
| 25 | `hp_ctxbias_005` | +9.25 | +23 | — | -3 | +9 | +8 | — | — | — | **OK** |
| 26 | **Flat subsets accum1.5x** | +9.33 | +15 | — | — | +3 | +10 | — | — | — | **OK** |
| 27 | `hp_dit_embed288_heads4` | +10.75 | +21 | — | +13 | +20 | -11 | — | — | — | **OK** |
| 28 | `diff_min_snr_gamma_5` | +12.50 | +18 | — | +12 | +13 | +7 | — | — | — | **OK** |
| 29 | `hp_dit_depth4` | +14.00 | +24 | — | +10 | +16 | +6 | — | — | — | **OK** |
| 30 | `diff_prediction_x0` | +15.75 | +20 | — | +9 | +18 | +16 | — | — | — | **OK** |
| 31 | `hp_beta_end_03` | +16.25 | +22 | — | +14 | +15 | +14 | — | — | — | **OK** |
| 32 | `h8_8_8` | +16.50 | +26 | — | +11 | +11 | +18 | — | — | — | **OK** |
| 33 | **Discrete** | +21.00 | +25 | — | — | +19 | +19 | — | — | — | **OK** |
| — | **MMPD (subset)** | — | — | — | — | — | — | — | — | — | ref |

### ETTh1

Baseline `sweep_baseline` rank: **8** / 34 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.3762 | 0.3936 | 0.2985 | — | ref |
| 2 | `diff_ema_decay_099` | 0.3974 | 0.4040 | 0.3021 | -6 | **OK** |
| 3 | **Flat subsets EMA0.99** | 0.3974 | 0.4040 | 0.3021 | -5 | **OK** |
| 4 | `hp_dit_dropout_01` | 0.3998 | 0.4056 | 0.3068 | -4 | **OK** |
| 5 | **Flat subsets EMA0.90** | 0.4052 | 0.4082 | 0.3057 | -3 | **OK** |
| 6 | `hp_lr_cosine_warmup2` | 0.4059 | 0.4085 | 0.3060 | -2 | **pre-fix invalid** |
| 7 | `hp_lr_cosine_warmup5` | 0.4059 | 0.4085 | 0.3060 | -1 | **pre-fix invalid** |
| 8 | `sweep_baseline` | 0.4059 | 0.4085 | 0.3060 | 0 | **OK** |
| 9 | **Flat subsets** | 0.4059 | 0.4085 | 0.3060 | +1 | **OK** |
| 10 | `hp_cfg_dropout_02` | 0.4059 | 0.4131 | 0.3080 | +2 | **OK** |
| 11 | `h16_16_16` | 0.4060 | 0.4086 | 0.3088 | +3 | **OK** |
| 12 | **Flat subsets EMA0.95** | 0.4061 | 0.4083 | 0.3061 | +4 | **OK** |
| 13 | **Flat subsets accum1.25x** | 0.4077 | 0.4116 | 0.3052 | +5 | **OK** |
| 14 | **Flat subsets EMA0.98** | 0.4084 | 0.4087 | 0.3066 | +6 | **OK** |
| 15 | `hp_beta_end_04` | 0.4091 | 0.4090 | 0.3116 | +7 | **OK** |
| 16 | **Flat subsets accum2.0x** | 0.4092 | 0.4105 | 0.3068 | +8 | **OK** |
| 17 | `hp_num_steps_1200` | 0.4092 | 0.4121 | 0.3100 | +9 | **OK** |
| 18 | `hp_ctxbias_neg01` | 0.4112 | 0.4097 | 0.3072 | +10 | **OK** |
| 19 | **MS tune** | 0.4117 | 0.4158 | 0.3114 | +11 | **pre-fix invalid** |
| 20 | `hp_anchor_lambda_095` | 0.4133 | 0.4114 | 0.3066 | +12 | **OK** |
| 21 | **Flat subsets EMA0.999** | 0.4140 | 0.4097 | 0.3087 | +13 | **OK** |
| 22 | `hp_num_steps_800` | 0.4146 | 0.4151 | 0.3098 | +14 | **OK** |
| 23 | **Flat subsets accum1.5x** | 0.4149 | 0.4118 | 0.3066 | +15 | **OK** |
| 24 | **Flat subsets EMA0.995** | 0.4158 | 0.4111 | 0.3090 | +16 | **OK** |
| 25 | `hp_anchor_lambda_090` | 0.4169 | 0.4124 | 0.3060 | +17 | **OK** |
| 26 | `diff_min_snr_gamma_5` | 0.4176 | 0.4144 | 0.3136 | +18 | **OK** |
| 27 | `diff_noise_cosine` | 0.4212 | 0.4196 | 0.3193 | +19 | **OK** |
| 28 | `diff_prediction_x0` | 0.4236 | 0.4101 | 0.3066 | +20 | **OK** |
| 29 | `hp_dit_embed288_heads4` | 0.4276 | 0.4221 | 0.3088 | +21 | **OK** |
| 30 | `hp_beta_end_03` | 0.4333 | 0.4188 | 0.4208 | +22 | **OK** |
| 31 | `hp_ctxbias_005` | 0.4456 | 0.4370 | 0.3157 | +23 | **OK** |
| 32 | `hp_dit_depth4` | 0.4799 | 0.4343 | 0.3173 | +24 | **OK** |
| 33 | **Discrete** | 0.5468 | 0.4577 | 0.7199 | +25 | **OK** |
| 34 | `h8_8_8` | 0.6140 | 0.4792 | 0.3435 | +26 | **OK** |

### ETTh2

Baseline `sweep_baseline` missing. Total configs: 12

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets EMA0.99** | 0.3116 | 0.3500 | 0.2673 | — | **OK** |
| 2 | **Flat subsets EMA0.95** | 0.3121 | 0.3509 | 0.2686 | — | **OK** |
| 3 | **Flat subsets EMA0.90** | 0.3124 | 0.3513 | 0.2684 | — | **OK** |
| 4 | **Flat subsets EMA0.98** | 0.3125 | 0.3507 | 0.2690 | — | **OK** |
| 5 | **Flat subsets EMA0.995** | 0.3134 | 0.3523 | 0.2699 | — | **OK** |
| 6 | **Flat subsets accum1.25x** | 0.3169 | 0.3545 | 0.2754 | — | **OK** |
| 7 | **MMPD (subset)** | 0.3186 | 0.3614 | 0.2705 | — | ref |
| 8 | **Flat subsets** | 0.3199 | 0.3546 | 0.2705 | — | **OK** |
| 9 | **Flat subsets accum2.0x** | 0.3200 | 0.3621 | 0.2780 | — | **OK** |
| 10 | **Flat subsets EMA0.999** | 0.3250 | 0.3647 | 0.2799 | — | **OK** |
| 11 | **Flat subsets accum1.5x** | 0.3275 | 0.3676 | 0.2872 | — | **OK** |
| 12 | **Discrete** | 0.3397 | 0.3582 | 0.6547 | — | **OK** |

### ETTm1

Baseline `sweep_baseline` rank: **9** / 23 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD** | 0.4208 | 0.4122 | 0.3109 | — | ref |
| 2 | `hp_anchor_lambda_095` | 0.4522 | 0.4210 | 0.3200 | -7 | **OK** |
| 3 | `diff_ema_decay_099` | 0.4556 | 0.4231 | 0.3290 | -6 | **OK** |
| 4 | `hp_anchor_lambda_090` | 0.4661 | 0.4317 | 0.3407 | -5 | **OK** |
| 5 | `hp_num_steps_800` | 0.4670 | 0.4274 | 0.3382 | -4 | **OK** |
| 6 | `hp_ctxbias_005` | 0.4681 | 0.4236 | 0.3278 | -3 | **OK** |
| 7 | `hp_lr_cosine_warmup2` | 0.4683 | 0.4259 | 0.3268 | -2 | **pre-fix invalid** |
| 8 | `hp_lr_cosine_warmup5` | 0.4683 | 0.4259 | 0.3268 | -1 | **pre-fix invalid** |
| 9 | `sweep_baseline` | 0.4683 | 0.4259 | 0.3268 | 0 | **OK** |
| 10 | `h16_16_16` | 0.4724 | 0.4307 | 0.3365 | +1 | **OK** |
| 11 | `hp_num_steps_1200` | 0.4754 | 0.4289 | 0.3353 | +2 | **OK** |
| 12 | **MS tune** | 0.4784 | 0.4232 | 0.3374 | +3 | **pre-fix invalid** |
| 13 | `hp_cfg_dropout_02` | 0.4785 | 0.4341 | 0.3281 | +4 | **OK** |
| 14 | `hp_ctxbias_neg01` | 0.4807 | 0.4288 | 0.3299 | +5 | **OK** |
| 15 | `diff_noise_cosine` | 0.4894 | 0.4368 | 0.3287 | +6 | **OK** |
| 16 | `hp_beta_end_04` | 0.4989 | 0.4383 | 0.5073 | +7 | **OK** |
| 17 | `hp_dit_dropout_01` | 0.5045 | 0.4394 | 0.3410 | +8 | **OK** |
| 18 | `diff_prediction_x0` | 0.5051 | 0.4334 | 0.3355 | +9 | **OK** |
| 19 | `hp_dit_depth4` | 0.5066 | 0.4498 | 0.3360 | +10 | **OK** |
| 20 | `h8_8_8` | 0.5236 | 0.4546 | 0.3306 | +11 | **OK** |
| 21 | `diff_min_snr_gamma_5` | 0.5463 | 0.4584 | 0.4031 | +12 | **OK** |
| 22 | `hp_dit_embed288_heads4` | 0.5516 | 0.4596 | 0.3396 | +13 | **OK** |
| 23 | `hp_beta_end_03` | 0.6789 | 0.5006 | 0.9025 | +14 | **OK** |

### exchange_rate

Baseline `sweep_baseline` rank: **14** / 34 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.0810 | 0.1987 | 0.1563 | — | ref |
| 2 | `hp_anchor_lambda_090` | 0.0825 | 0.2014 | 0.1656 | -12 | **OK** |
| 3 | `hp_beta_end_04` | 0.0829 | 0.2035 | 0.2460 | -11 | **OK** |
| 4 | `hp_ctxbias_neg01` | 0.0857 | 0.2042 | 0.1647 | -10 | **OK** |
| 5 | `hp_cfg_dropout_02` | 0.0858 | 0.2053 | 0.1691 | -9 | **OK** |
| 6 | **Flat subsets EMA0.90** | 0.0868 | 0.2071 | 0.1699 | -8 | **OK** |
| 7 | **MS tune** | 0.0871 | 0.2067 | 0.1685 | -7 | **pre-fix invalid** |
| 8 | **Flat subsets EMA0.995** | 0.0875 | 0.2074 | 0.1667 | -6 | **OK** |
| 9 | **Flat subsets accum2.0x** | 0.0875 | 0.2073 | 0.1687 | -5 | **OK** |
| 10 | **Flat subsets EMA0.95** | 0.0875 | 0.2077 | 0.1689 | -4 | **OK** |
| 11 | **Flat subsets EMA0.98** | 0.0878 | 0.2078 | 0.1684 | -3 | **OK** |
| 12 | `hp_lr_cosine_warmup2` | 0.0880 | 0.2078 | 0.1660 | -2 | **pre-fix invalid** |
| 13 | `hp_lr_cosine_warmup5` | 0.0880 | 0.2078 | 0.1660 | -1 | **pre-fix invalid** |
| 14 | `sweep_baseline` | 0.0880 | 0.2078 | 0.1660 | 0 | **OK** |
| 15 | **Flat subsets** | 0.0880 | 0.2078 | 0.1660 | +1 | **OK** |
| 16 | `hp_anchor_lambda_095` | 0.0881 | 0.2096 | 0.1701 | +2 | **OK** |
| 17 | **Flat subsets accum1.5x** | 0.0882 | 0.2071 | 0.1657 | +3 | **OK** |
| 18 | `h16_16_16` | 0.0892 | 0.2090 | 0.1760 | +4 | **OK** |
| 19 | `hp_num_steps_1200` | 0.0893 | 0.2081 | 0.1686 | +5 | **OK** |
| 20 | `diff_ema_decay_099` | 0.0893 | 0.2086 | 0.1684 | +6 | **OK** |
| 21 | **Flat subsets EMA0.99** | 0.0893 | 0.2086 | 0.1684 | +7 | **OK** |
| 22 | **Flat subsets accum1.25x** | 0.0895 | 0.2090 | 0.1698 | +8 | **OK** |
| 23 | `hp_ctxbias_005` | 0.0896 | 0.2107 | 0.1684 | +9 | **OK** |
| 24 | `hp_num_steps_800` | 0.0898 | 0.2115 | 0.1734 | +10 | **OK** |
| 25 | `h8_8_8` | 0.0901 | 0.2102 | 0.1614 | +11 | **OK** |
| 26 | `diff_noise_cosine` | 0.0905 | 0.2094 | 0.1788 | +12 | **OK** |
| 27 | `diff_min_snr_gamma_5` | 0.0907 | 0.2125 | 0.1712 | +13 | **OK** |
| 28 | **Flat subsets EMA0.999** | 0.0909 | 0.2107 | 0.1703 | +14 | **OK** |
| 29 | `hp_beta_end_03` | 0.0911 | 0.2138 | 0.2791 | +15 | **OK** |
| 30 | `hp_dit_depth4` | 0.0913 | 0.2108 | 0.1703 | +16 | **OK** |
| 31 | `hp_dit_dropout_01` | 0.0924 | 0.2134 | 0.1754 | +17 | **OK** |
| 32 | `diff_prediction_x0` | 0.0924 | 0.2130 | 0.1700 | +18 | **OK** |
| 33 | **Discrete** | 0.0924 | 0.2132 | 0.3384 | +19 | **OK** |
| 34 | `hp_dit_embed288_heads4` | 0.0925 | 0.2151 | 0.1707 | +20 | **OK** |

### weather

Baseline `sweep_baseline` rank: **14** / 34 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets EMA0.999** | 0.0945 | 0.2193 | 0.1759 | -13 | **OK** |
| 2 | `diff_noise_cosine` | 0.0962 | 0.2192 | 0.1794 | -12 | **OK** |
| 3 | `hp_dit_embed288_heads4` | 0.0963 | 0.2201 | 0.1734 | -11 | **OK** |
| 4 | **Flat subsets EMA0.995** | 0.0964 | 0.2213 | 0.1752 | -10 | **OK** |
| 5 | **Flat subsets EMA0.95** | 0.0964 | 0.2217 | 0.1792 | -9 | **OK** |
| 6 | **Flat subsets EMA0.98** | 0.0966 | 0.2216 | 0.1760 | -8 | **OK** |
| 7 | **Flat subsets EMA0.90** | 0.0968 | 0.2222 | 0.1803 | -7 | **OK** |
| 8 | `hp_beta_end_04` | 0.0969 | 0.2203 | 0.2248 | -6 | **OK** |
| 9 | `diff_ema_decay_099` | 0.0971 | 0.2220 | 0.1758 | -5 | **OK** |
| 10 | **Flat subsets EMA0.99** | 0.0971 | 0.2220 | 0.1758 | -4 | **OK** |
| 11 | **Flat subsets accum1.25x** | 0.0975 | 0.2229 | 0.1777 | -3 | **OK** |
| 12 | `hp_lr_cosine_warmup2` | 0.0978 | 0.2224 | 0.1776 | -2 | **pre-fix invalid** |
| 13 | `hp_lr_cosine_warmup5` | 0.0978 | 0.2224 | 0.1776 | -1 | **pre-fix invalid** |
| 14 | `sweep_baseline` | 0.0978 | 0.2224 | 0.1776 | 0 | **OK** |
| 15 | **Flat subsets** | 0.0978 | 0.2224 | 0.1776 | +1 | **OK** |
| 16 | **Flat subsets accum2.0x** | 0.0979 | 0.2234 | 0.1858 | +2 | **OK** |
| 17 | `hp_cfg_dropout_02` | 0.0981 | 0.2215 | 0.1795 | +3 | **OK** |
| 18 | `h16_16_16` | 0.0981 | 0.2216 | 0.1767 | +4 | **OK** |
| 19 | **MS tune** | 0.0987 | 0.2185 | 0.1775 | +5 | **pre-fix invalid** |
| 20 | `hp_dit_depth4` | 0.0989 | 0.2227 | 0.1801 | +6 | **OK** |
| 21 | `diff_min_snr_gamma_5` | 0.0992 | 0.2243 | 0.1804 | +7 | **OK** |
| 22 | `hp_ctxbias_005` | 0.0992 | 0.2212 | 0.1746 | +8 | **OK** |
| 23 | `hp_anchor_lambda_095` | 0.0993 | 0.2231 | 0.1779 | +9 | **OK** |
| 24 | **Flat subsets accum1.5x** | 0.0994 | 0.2251 | 0.1759 | +10 | **OK** |
| 25 | `hp_ctxbias_neg01` | 0.0996 | 0.2243 | 0.1778 | +11 | **OK** |
| 26 | `hp_anchor_lambda_090` | 0.1001 | 0.2235 | 0.1757 | +12 | **OK** |
| 27 | `hp_num_steps_800` | 0.1009 | 0.2247 | 0.1818 | +13 | **OK** |
| 28 | `hp_beta_end_03` | 0.1023 | 0.2277 | 0.2997 | +14 | **OK** |
| 29 | `hp_dit_dropout_01` | 0.1031 | 0.2268 | 0.1808 | +15 | **OK** |
| 30 | `diff_prediction_x0` | 0.1036 | 0.2276 | 0.1791 | +16 | **OK** |
| 31 | `hp_num_steps_1200` | 0.1037 | 0.2270 | 0.1832 | +17 | **OK** |
| 32 | `h8_8_8` | 0.1043 | 0.2282 | 0.1785 | +18 | **OK** |
| 33 | **Discrete** | 0.1079 | 0.2280 | 0.3707 | +19 | **OK** |
| 34 | **MMPD (subset)** | 0.1128 | 0.2323 | 0.1911 | — | ref |

### electricity

Baseline `sweep_baseline` missing. Total configs: 12

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.1617 | 0.2088 | 0.1610 | — | ref |
| 2 | **Flat subsets EMA0.95** | 0.1690 | 0.2088 | 0.1560 | — | **OK** |
| 3 | **Flat subsets EMA0.98** | 0.1692 | 0.2091 | 0.1562 | — | **OK** |
| 4 | **Flat subsets EMA0.90** | 0.1693 | 0.2086 | 0.1561 | — | **OK** |
| 5 | **Flat subsets EMA0.995** | 0.1706 | 0.2101 | 0.1566 | — | **OK** |
| 6 | **Flat subsets EMA0.99** | 0.1713 | 0.2111 | 0.1572 | — | **OK** |
| 7 | **Flat subsets** | 0.1735 | 0.2132 | 0.1602 | — | **OK** |
| 8 | **Flat subsets accum1.25x** | 0.1737 | 0.2140 | 0.1600 | — | **OK** |
| 9 | **Flat subsets accum1.5x** | 0.1820 | 0.2213 | 0.1629 | — | **OK** |
| 10 | **Flat subsets accum2.0x** | 0.1843 | 0.2245 | 0.1617 | — | **OK** |
| 11 | **Flat subsets EMA0.999** | 0.1974 | 0.2361 | 0.1619 | — | **OK** |
| 12 | **Discrete** | 0.2972 | 0.2712 | 0.5994 | — | **OK** |

### traffic

Baseline `sweep_baseline` missing. Total configs: 12

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Discrete** | 0.3863 | 0.2382 | 1.2598 | — | **OK** |
| 2 | **Flat subsets EMA0.98** | 0.5152 | 0.3161 | 0.2367 | — | **OK** |
| 3 | **Flat subsets EMA0.95** | 0.5167 | 0.3161 | 0.2376 | — | **OK** |
| 4 | **Flat subsets EMA0.90** | 0.5208 | 0.3168 | 0.2398 | — | **OK** |
| 5 | **Flat subsets EMA0.995** | 0.5219 | 0.3196 | 0.2397 | — | **OK** |
| 6 | **MMPD (subset)** | 0.5225 | 0.3612 | 0.2515 | — | ref |
| 7 | **Flat subsets** | 0.5263 | 0.3252 | 0.2430 | — | **OK** |
| 8 | **Flat subsets EMA0.99** | 0.5296 | 0.3245 | 0.2436 | — | **OK** |
| 9 | **Flat subsets accum1.25x** | 0.5318 | 0.3318 | 0.2481 | — | **OK** |
| 10 | **Flat subsets accum1.5x** | 0.5456 | 0.3373 | 0.2556 | — | **OK** |
| 11 | **Flat subsets accum2.0x** | 0.5998 | 0.3526 | 0.2710 | — | **OK** |
| 12 | **Flat subsets EMA0.999** | 0.8891 | 0.4865 | 0.3351 | — | **OK** |

### solar_Alabama

Baseline `sweep_baseline` missing. Total configs: 12

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets accum1.5x** | 0.2002 | 0.2313 | 0.1789 | — | **OK** |
| 2 | **Flat subsets EMA0.999** | 0.2108 | 0.2369 | 0.1912 | — | **OK** |
| 3 | **Flat subsets EMA0.99** | 0.2123 | 0.2366 | 0.1890 | — | **OK** |
| 4 | **Flat subsets EMA0.98** | 0.2126 | 0.2371 | 0.1881 | — | **OK** |
| 5 | **Flat subsets EMA0.995** | 0.2129 | 0.2393 | 0.1932 | — | **OK** |
| 6 | **Flat subsets EMA0.95** | 0.2136 | 0.2379 | 0.1881 | — | **OK** |
| 7 | **Flat subsets EMA0.90** | 0.2147 | 0.2387 | 0.1884 | — | **OK** |
| 8 | **Flat subsets accum2.0x** | 0.2158 | 0.2394 | 0.1876 | — | **OK** |
| 9 | **Flat subsets** | 0.2170 | 0.2426 | 0.1945 | — | **OK** |
| 10 | **Flat subsets accum1.25x** | 0.2231 | 0.2439 | 0.1977 | — | **OK** |
| 11 | **MMPD (subset)** | 0.2360 | 0.2690 | 0.2013 | — | ref |
| 12 | **Discrete** | 0.2560 | 0.2472 | 1.5170 | — | **OK** |

