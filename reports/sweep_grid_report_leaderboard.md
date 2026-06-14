# YAML-First Sweep Leaderboard

Probabilistic metrics from `dpmpp` sampler with `20` steps. Baseline is `sweep_baseline` (fixed 3e-5 LR, linear noise, epsilon target). **Discrete** is ordinal D3PM CE (`ordinal_d3pm_staged`). **MAE Discrete** is expectation-MAE + uniform `1/H` anchor (`ordinal_d3pm_mae_staged_subsets`). **Binary flat** is flat `0.5` XOR anchor on full variates (`binary_anchor_stationary_flat`). **Flat subsets** / **Flat subsets EMA0.99** use ETTh1-capped variate subsets (`binary_anchor_stationary_flat_subsets` / `_ema099`). **MS tune** is Optuna `max_scale` search (`hp_max_scale_tuning`, jobs `3943934`–`3943937`). **MMPD (subset)** from `06-13-binary-mmpd-subset-compare` (same subsets as flat runs, 20 samples, full test). Legacy **MMPD** from `06-12-sweep-subset-mmpd` where subset MMPD is unavailable.

## Average Δrank vs baseline

Δrank = config rank − `sweep_baseline` rank per dataset (negative = better anchor MSE). Avg Δrank averages over datasets where the config ran.

| Rank | Config | avg Δrank | ETTh1 Δrank | ETTh2 Δrank | ETTm1 Δrank | exchange_rate Δrank | weather Δrank | electricity Δrank | traffic Δrank | solar_Alabama Δrank | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `diff_ema_decay_099` | -2.50 | -5 | — | -6 | +5 | -4 | — | — | — | **OK** |
| 2 | `hp_lr_cosine_warmup2` | -2.00 | -2 | — | -2 | -2 | -2 | — | — | — | **OK** |
| 3 | `hp_lr_cosine_warmup5` | -1.00 | -1 | — | -1 | -1 | -1 | — | — | — | **OK** |
| 4 | **Flat subsets EMA0.99** | -0.33 | -4 | — | — | +6 | -3 | — | — | — | **OK** |
| 5 | `sweep_baseline` | +0.00 | 0 | — | 0 | 0 | 0 | — | — | — | **OK** |
| 6 | `hp_beta_end_04` | +0.00 | +4 | — | +7 | -6 | -5 | — | — | — | **OK** |
| 7 | **Flat subsets** | +1.00 | +1 | — | — | +1 | +1 | — | — | — | **OK** |
| 8 | `hp_cfg_dropout_02` | +1.00 | +2 | — | +4 | -4 | +2 | — | — | — | **OK** |
| 9 | `hp_anchor_lambda_090` | +2.00 | +10 | — | -5 | -7 | +10 | — | — | — | **OK** |
| 10 | `h16_16_16` | +2.50 | +3 | — | +1 | +3 | +3 | — | — | — | **OK** |
| 11 | `hp_anchor_lambda_095` | +2.75 | +8 | — | -7 | +2 | +8 | — | — | — | **OK** |
| 12 | **MS tune** | +2.75 | +7 | — | +3 | -3 | +4 | — | — | — | **OK** |
| 13 | `hp_ctxbias_neg01` | +3.75 | +6 | — | +5 | -5 | +9 | — | — | — | **OK** |
| 14 | `diff_noise_cosine` | +5.25 | +12 | — | +6 | +10 | -7 | — | — | — | **OK** |
| 15 | `hp_num_steps_800` | +6.00 | +9 | — | -4 | +8 | +11 | — | — | — | **OK** |
| 16 | `hp_num_steps_1200` | +6.50 | +5 | — | +2 | +4 | +15 | — | — | — | **OK** |
| 17 | `hp_ctxbias_005` | +6.75 | +16 | — | -3 | +7 | +7 | — | — | — | **OK** |
| 18 | `hp_dit_dropout_01` | +8.00 | -3 | — | +8 | +14 | +13 | — | — | — | **OK** |
| 19 | `hp_dit_embed288_heads4` | +9.50 | +14 | — | +13 | +17 | -6 | — | — | — | **OK** |
| 20 | `diff_min_snr_gamma_5` | +10.00 | +11 | — | +12 | +11 | +6 | — | — | — | **OK** |
| 21 | `hp_dit_depth4` | +11.25 | +17 | — | +10 | +13 | +5 | — | — | — | **OK** |
| 22 | `diff_prediction_x0` | +12.75 | +13 | — | +9 | +15 | +14 | — | — | — | **OK** |
| 23 | `hp_beta_end_03` | +13.25 | +15 | — | +14 | +12 | +12 | — | — | — | **OK** |
| 24 | `h8_8_8` | +13.75 | +19 | — | +11 | +9 | +16 | — | — | — | **OK** |
| 25 | **Discrete** | +17.00 | +18 | — | — | +16 | +17 | — | — | — | **OK** |
| — | **MMPD (subset)** | — | — | — | — | — | — | — | — | — | ref |

### ETTh1

Baseline `sweep_baseline` rank: **7** / 26 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.3762 | 0.3936 | 0.2985 | — | ref |
| 2 | `diff_ema_decay_099` | 0.3974 | 0.4040 | 0.3021 | -5 | **OK** |
| 3 | **Flat subsets EMA0.99** | 0.3974 | 0.4040 | 0.3021 | -4 | **OK** |
| 4 | `hp_dit_dropout_01` | 0.3998 | 0.4056 | 0.3068 | -3 | **OK** |
| 5 | `hp_lr_cosine_warmup2` | 0.4059 | 0.4085 | 0.3060 | -2 | **OK** |
| 6 | `hp_lr_cosine_warmup5` | 0.4059 | 0.4085 | 0.3060 | -1 | **OK** |
| 7 | `sweep_baseline` | 0.4059 | 0.4085 | 0.3060 | 0 | **OK** |
| 8 | **Flat subsets** | 0.4059 | 0.4085 | 0.3060 | +1 | **OK** |
| 9 | `hp_cfg_dropout_02` | 0.4059 | 0.4131 | 0.3080 | +2 | **OK** |
| 10 | `h16_16_16` | 0.4060 | 0.4086 | 0.3088 | +3 | **OK** |
| 11 | `hp_beta_end_04` | 0.4091 | 0.4090 | 0.3116 | +4 | **OK** |
| 12 | `hp_num_steps_1200` | 0.4092 | 0.4121 | 0.3100 | +5 | **OK** |
| 13 | `hp_ctxbias_neg01` | 0.4112 | 0.4097 | 0.3072 | +6 | **OK** |
| 14 | **MS tune** | 0.4117 | 0.4158 | 0.3114 | +7 | **OK** |
| 15 | `hp_anchor_lambda_095` | 0.4133 | 0.4114 | 0.3066 | +8 | **OK** |
| 16 | `hp_num_steps_800` | 0.4146 | 0.4151 | 0.3098 | +9 | **OK** |
| 17 | `hp_anchor_lambda_090` | 0.4169 | 0.4124 | 0.3060 | +10 | **OK** |
| 18 | `diff_min_snr_gamma_5` | 0.4176 | 0.4144 | 0.3136 | +11 | **OK** |
| 19 | `diff_noise_cosine` | 0.4212 | 0.4196 | 0.3193 | +12 | **OK** |
| 20 | `diff_prediction_x0` | 0.4236 | 0.4101 | 0.3066 | +13 | **OK** |
| 21 | `hp_dit_embed288_heads4` | 0.4276 | 0.4221 | 0.3088 | +14 | **OK** |
| 22 | `hp_beta_end_03` | 0.4333 | 0.4188 | 0.4208 | +15 | **OK** |
| 23 | `hp_ctxbias_005` | 0.4456 | 0.4370 | 0.3157 | +16 | **OK** |
| 24 | `hp_dit_depth4` | 0.4799 | 0.4343 | 0.3173 | +17 | **OK** |
| 25 | **Discrete** | 0.5468 | 0.4577 | 0.7199 | +18 | **OK** |
| 26 | `h8_8_8` | 0.6140 | 0.4792 | 0.3435 | +19 | **OK** |

### ETTh2

Baseline `sweep_baseline` missing. Total configs: 4

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets EMA0.99** | 0.3116 | 0.3500 | 0.2673 | — | **OK** |
| 2 | **MMPD (subset)** | 0.3186 | 0.3614 | 0.2705 | — | ref |
| 3 | **Flat subsets** | 0.3199 | 0.3546 | 0.2705 | — | **OK** |
| 4 | **Discrete** | 0.3397 | 0.3582 | 0.6547 | — | **OK** |

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
| 7 | `hp_lr_cosine_warmup2` | 0.4683 | 0.4259 | 0.3268 | -2 | **OK** |
| 8 | `hp_lr_cosine_warmup5` | 0.4683 | 0.4259 | 0.3268 | -1 | **OK** |
| 9 | `sweep_baseline` | 0.4683 | 0.4259 | 0.3268 | 0 | **OK** |
| 10 | `h16_16_16` | 0.4724 | 0.4307 | 0.3365 | +1 | **OK** |
| 11 | `hp_num_steps_1200` | 0.4754 | 0.4289 | 0.3353 | +2 | **OK** |
| 12 | **MS tune** | 0.4784 | 0.4232 | 0.3374 | +3 | **OK** |
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

Baseline `sweep_baseline` rank: **9** / 26 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.0810 | 0.1987 | 0.1563 | — | ref |
| 2 | `hp_anchor_lambda_090` | 0.0825 | 0.2014 | 0.1656 | -7 | **OK** |
| 3 | `hp_beta_end_04` | 0.0829 | 0.2035 | 0.2460 | -6 | **OK** |
| 4 | `hp_ctxbias_neg01` | 0.0857 | 0.2042 | 0.1647 | -5 | **OK** |
| 5 | `hp_cfg_dropout_02` | 0.0858 | 0.2053 | 0.1691 | -4 | **OK** |
| 6 | **MS tune** | 0.0871 | 0.2067 | 0.1685 | -3 | **OK** |
| 7 | `hp_lr_cosine_warmup2` | 0.0880 | 0.2078 | 0.1660 | -2 | **OK** |
| 8 | `hp_lr_cosine_warmup5` | 0.0880 | 0.2078 | 0.1660 | -1 | **OK** |
| 9 | `sweep_baseline` | 0.0880 | 0.2078 | 0.1660 | 0 | **OK** |
| 10 | **Flat subsets** | 0.0880 | 0.2078 | 0.1660 | +1 | **OK** |
| 11 | `hp_anchor_lambda_095` | 0.0881 | 0.2096 | 0.1701 | +2 | **OK** |
| 12 | `h16_16_16` | 0.0892 | 0.2090 | 0.1760 | +3 | **OK** |
| 13 | `hp_num_steps_1200` | 0.0893 | 0.2081 | 0.1686 | +4 | **OK** |
| 14 | `diff_ema_decay_099` | 0.0893 | 0.2086 | 0.1684 | +5 | **OK** |
| 15 | **Flat subsets EMA0.99** | 0.0893 | 0.2086 | 0.1684 | +6 | **OK** |
| 16 | `hp_ctxbias_005` | 0.0896 | 0.2107 | 0.1684 | +7 | **OK** |
| 17 | `hp_num_steps_800` | 0.0898 | 0.2115 | 0.1734 | +8 | **OK** |
| 18 | `h8_8_8` | 0.0901 | 0.2102 | 0.1614 | +9 | **OK** |
| 19 | `diff_noise_cosine` | 0.0905 | 0.2094 | 0.1788 | +10 | **OK** |
| 20 | `diff_min_snr_gamma_5` | 0.0907 | 0.2125 | 0.1712 | +11 | **OK** |
| 21 | `hp_beta_end_03` | 0.0911 | 0.2138 | 0.2791 | +12 | **OK** |
| 22 | `hp_dit_depth4` | 0.0913 | 0.2108 | 0.1703 | +13 | **OK** |
| 23 | `hp_dit_dropout_01` | 0.0924 | 0.2134 | 0.1754 | +14 | **OK** |
| 24 | `diff_prediction_x0` | 0.0924 | 0.2130 | 0.1700 | +15 | **OK** |
| 25 | **Discrete** | 0.0924 | 0.2132 | 0.3384 | +16 | **OK** |
| 26 | `hp_dit_embed288_heads4` | 0.0925 | 0.2151 | 0.1707 | +17 | **OK** |

### weather

Baseline `sweep_baseline` rank: **8** / 26 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | `diff_noise_cosine` | 0.0962 | 0.2192 | 0.1794 | -7 | **OK** |
| 2 | `hp_dit_embed288_heads4` | 0.0963 | 0.2201 | 0.1734 | -6 | **OK** |
| 3 | `hp_beta_end_04` | 0.0969 | 0.2203 | 0.2248 | -5 | **OK** |
| 4 | `diff_ema_decay_099` | 0.0971 | 0.2220 | 0.1758 | -4 | **OK** |
| 5 | **Flat subsets EMA0.99** | 0.0971 | 0.2220 | 0.1758 | -3 | **OK** |
| 6 | `hp_lr_cosine_warmup2` | 0.0978 | 0.2224 | 0.1776 | -2 | **OK** |
| 7 | `hp_lr_cosine_warmup5` | 0.0978 | 0.2224 | 0.1776 | -1 | **OK** |
| 8 | `sweep_baseline` | 0.0978 | 0.2224 | 0.1776 | 0 | **OK** |
| 9 | **Flat subsets** | 0.0978 | 0.2224 | 0.1776 | +1 | **OK** |
| 10 | `hp_cfg_dropout_02` | 0.0981 | 0.2215 | 0.1795 | +2 | **OK** |
| 11 | `h16_16_16` | 0.0981 | 0.2216 | 0.1767 | +3 | **OK** |
| 12 | **MS tune** | 0.0987 | 0.2185 | 0.1775 | +4 | **OK** |
| 13 | `hp_dit_depth4` | 0.0989 | 0.2227 | 0.1801 | +5 | **OK** |
| 14 | `diff_min_snr_gamma_5` | 0.0992 | 0.2243 | 0.1804 | +6 | **OK** |
| 15 | `hp_ctxbias_005` | 0.0992 | 0.2212 | 0.1746 | +7 | **OK** |
| 16 | `hp_anchor_lambda_095` | 0.0993 | 0.2231 | 0.1779 | +8 | **OK** |
| 17 | `hp_ctxbias_neg01` | 0.0996 | 0.2243 | 0.1778 | +9 | **OK** |
| 18 | `hp_anchor_lambda_090` | 0.1001 | 0.2235 | 0.1757 | +10 | **OK** |
| 19 | `hp_num_steps_800` | 0.1009 | 0.2247 | 0.1818 | +11 | **OK** |
| 20 | `hp_beta_end_03` | 0.1023 | 0.2277 | 0.2997 | +12 | **OK** |
| 21 | `hp_dit_dropout_01` | 0.1031 | 0.2268 | 0.1808 | +13 | **OK** |
| 22 | `diff_prediction_x0` | 0.1036 | 0.2276 | 0.1791 | +14 | **OK** |
| 23 | `hp_num_steps_1200` | 0.1037 | 0.2270 | 0.1832 | +15 | **OK** |
| 24 | `h8_8_8` | 0.1043 | 0.2282 | 0.1785 | +16 | **OK** |
| 25 | **Discrete** | 0.1079 | 0.2280 | 0.3707 | +17 | **OK** |
| 26 | **MMPD (subset)** | 0.1128 | 0.2323 | 0.1911 | — | ref |

### electricity

Baseline `sweep_baseline` missing. Total configs: 4

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.1617 | 0.2088 | 0.1610 | — | ref |
| 2 | **Flat subsets EMA0.99** | 0.1713 | 0.2111 | 0.1572 | — | **OK** |
| 3 | **Flat subsets** | 0.1735 | 0.2132 | 0.1602 | — | **OK** |
| 4 | **Discrete** | 0.2972 | 0.2712 | 0.5994 | — | **OK** |

### traffic

Baseline `sweep_baseline` missing. Total configs: 4

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Discrete** | 0.3863 | 0.2382 | 1.2598 | — | **OK** |
| 2 | **MMPD (subset)** | 0.5225 | 0.3612 | 0.2515 | — | ref |
| 3 | **Flat subsets** | 0.5263 | 0.3252 | 0.2430 | — | **OK** |
| 4 | **Flat subsets EMA0.99** | 0.5296 | 0.3245 | 0.2436 | — | **OK** |

### solar_Alabama

Baseline `sweep_baseline` missing. Total configs: 4

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets EMA0.99** | 0.2123 | 0.2366 | 0.1890 | — | **OK** |
| 2 | **Flat subsets** | 0.2170 | 0.2426 | 0.1945 | — | **OK** |
| 3 | **MMPD (subset)** | 0.2360 | 0.2690 | 0.2013 | — | ref |
| 4 | **Discrete** | 0.2560 | 0.2472 | 1.5170 | — | **OK** |

