# YAML-First Sweep Leaderboard

Probabilistic metrics from `dpmpp` sampler with `20` steps. Baseline is `sweep_baseline` (fixed 3e-5 LR, linear noise, epsilon target). **Discrete** is ordinal D3PM CE (`ordinal_d3pm_staged`). **MAE Discrete** is expectation-MAE + uniform `1/H` anchor (`ordinal_d3pm_mae_staged_subsets`). **Binary flat** is flat `0.5` XOR anchor on full variates (`binary_anchor_stationary_flat`). **Flat subsets** (`binary_anchor_stationary_flat_subsets`, jobs `3951193`–`3951199`). **Flat subsets EMA0.99** (`3951527`–`3951533`). EMA reuse sweep: `diffusion_ema_decay` ∈ {0.90, 0.95, 0.98, 0.995, 0.999} (jobs `3953317`–`3953351`). Grad-accum reuse sweep: effective batch {1.25×, 1.5×, 2.0×} (jobs `3953944`–`3953964`); LR-band split on 1.5×/2.0× (`3954784`–`3954810`). **Flat subsets guidance accum** {1.5×, 2×, 4×, 8×} (jobs `3961419`–`3961447`). EMA0.99 lookback variants: LB336/H96 and LB96/H720 (`3955091`–`3955098`). **AR accum4x/8x** (`binary_anchor_ar_grad_accum_{400,800}`, LB96/H96). **AR LB336/H96 accum1.5x** (`3961448`–`3961454`); **AR LB96/H720 accum1.5x** (partial: `3961455`, `3961457`, `3961460`). **MS tune** (`hp_max_scale_tuning`, post-fix `3956631` exchange_rate; cosine+warmup post-fix `3956633`–`3956640`). **MMPD (subset)** from `06-13-binary-mmpd-subset-compare` (same subsets as flat runs, 20 samples, full test). Legacy **MMPD** from `06-12-sweep-subset-mmpd` where subset MMPD is unavailable.

**Pre-fix invalid / incomplete:** Jun 12 `hp_max_scale_tuning` jobs `3943934`–`3943937` never searched `max_scale`. Post-fix cosine+warmup `3956633`–`3956640` replaces pre-fix `3943882`–`3943927`. Post-fix MS tune: exchange_rate `3956631` only; ETTh1/ETTm1/weather `3956629`–`3956632` hit 3h wall before eval — still showing pre-fix rows.

## Average Δrank vs baseline

Δrank = config rank − `sweep_baseline` rank per dataset (negative = better anchor MSE). Avg Δrank averages over datasets where the config ran.

| Rank | Config | avg Δrank | ETTh1 Δrank | ETTh2 Δrank | ETTm1 Δrank | exchange_rate Δrank | weather Δrank | electricity Δrank | traffic Δrank | solar_Alabama Δrank | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Flat subsets EMA0.90** | -5.33 | -2 | — | — | -6 | -8 | — | — | — | **OK** |
| 2 | **Flat subsets guidance accum4x** | -4.33 | +7 | — | — | -5 | -15 | — | — | — | **OK** |
| 3 | `diff_ema_decay_099` | -3.00 | -9 | — | -5 | +8 | -6 | — | — | — | **OK** |
| 4 | **Flat subsets EMA0.95** | -2.33 | +5 | — | — | -2 | -10 | — | — | — | **OK** |
| 5 | **Flat subsets accum1.5x LR-hi** | -1.67 | -10 | — | — | +17 | -12 | — | — | — | **OK** |
| 6 | **Flat subsets EMA0.99** | -1.33 | -8 | — | — | +9 | -5 | — | — | — | **OK** |
| 7 | **Flat subsets guidance accum1.5x** | -0.67 | -4 | — | — | +18 | -16 | — | — | — | **OK** |
| 8 | **Flat subsets EMA0.98** | -0.67 | +8 | — | — | -1 | -9 | — | — | — | **OK** |
| 9 | `sweep_baseline` | +0.00 | 0 | — | 0 | 0 | 0 | — | — | — | **OK** |
| 10 | `hp_beta_end_04` | +0.00 | +9 | — | +8 | -10 | -7 | — | — | — | **OK** |
| 11 | **Flat subsets guidance accum2x** | +0.33 | -3 | — | — | +22 | -18 | — | — | — | **OK** |
| 12 | **Binary flat** | +1.00 | +1 | — | — | +1 | +1 | — | — | — | **OK** |
| 13 | `hp_cfg_dropout_02` | +1.00 | +3 | — | +3 | -7 | +5 | — | — | — | **OK** |
| 14 | **Flat subsets EMA0.995** | +2.00 | +21 | — | — | -4 | -11 | — | — | — | **OK** |
| 15 | **Flat subsets** | +2.00 | +2 | — | — | +2 | +2 | — | — | — | **OK** |
| 16 | **Flat subsets accum2.0x LR-hi** | +2.67 | -6 | — | — | +15 | -1 | — | — | — | **OK** |
| 17 | **Flat subsets accum2.0x** | +3.33 | +10 | — | — | -3 | +3 | — | — | — | **OK** |
| 18 | `h16_16_16` | +4.25 | +4 | — | +1 | +6 | +6 | — | — | — | **OK** |
| 19 | **Flat subsets accum1.25x** | +4.33 | +6 | — | — | +10 | -3 | — | — | — | **OK** |
| 20 | `hp_ctxbias_neg01` | +5.75 | +14 | — | +4 | -8 | +13 | — | — | — | **OK** |
| 21 | `hp_anchor_lambda_090` | +5.75 | +22 | — | -4 | -11 | +16 | — | — | — | **OK** |
| 22 | `hp_anchor_lambda_095` | +6.25 | +17 | — | -6 | +3 | +11 | — | — | — | **OK** |
| 23 | `hp_lr_cosine_warmup5` | +6.50 | -5 | — | +5 | +12 | +14 | — | — | — | **OK** |
| 24 | **MAE Discrete** | +6.67 | +25 | — | — | -9 | +4 | — | — | — | **OK** |
| 25 | **Flat subsets EMA0.999** | +7.33 | +18 | — | — | +21 | -17 | — | — | — | **OK** |
| 26 | `hp_lr_cosine_warmup2` | +8.50 | -1 | — | +6 | +14 | +15 | — | — | — | **OK** |
| 27 | **Flat subsets accum2.0x LR-lo** | +9.00 | +24 | — | — | +5 | -2 | — | — | — | **OK** |
| 28 | `diff_noise_cosine` | +9.50 | +26 | — | +7 | +19 | -14 | — | — | — | **OK** |
| 29 | `hp_num_steps_1200` | +10.50 | +11 | — | +2 | +7 | +22 | — | — | — | **OK** |
| 30 | **MS tune** | +10.75 | +15 | — | -2 | +23 | +7 | — | — | — | **pre-fix invalid** |
| 31 | `hp_num_steps_800` | +11.75 | +19 | — | -3 | +13 | +18 | — | — | — | **OK** |
| 32 | **Flat subsets accum1.5x** | +12.00 | +20 | — | — | +4 | +12 | — | — | — | **OK** |
| 33 | `hp_ctxbias_005` | +12.50 | +30 | — | -1 | +11 | +10 | — | — | — | **OK** |
| 34 | `hp_dit_dropout_01` | +12.50 | -7 | — | +9 | +28 | +20 | — | — | — | **OK** |
| 35 | **Flat subsets accum1.5x LR-lo** | +12.67 | +16 | — | — | +26 | -4 | — | — | — | **OK** |
| 36 | `hp_dit_embed288_heads4` | +15.00 | +28 | — | +14 | +31 | -13 | — | — | — | **OK** |
| 37 | `diff_min_snr_gamma_5` | +16.25 | +23 | — | +13 | +20 | +9 | — | — | — | **OK** |
| 38 | `hp_dit_depth4` | +19.00 | +32 | — | +11 | +25 | +8 | — | — | — | **OK** |
| 39 | `h8_8_8` | +21.25 | +34 | — | +12 | +16 | +23 | — | — | — | **OK** |
| 40 | **Flat subsets EMA0.99 LB336/H96** | +21.33 | +13 | — | — | +27 | +24 | — | — | — | **OK** |
| 41 | `hp_beta_end_03` | +21.75 | +29 | — | +15 | +24 | +19 | — | — | — | **OK** |
| 42 | `diff_prediction_x0` | +21.75 | +27 | — | +10 | +29 | +21 | — | — | — | **OK** |
| 43 | **AR LB336/H96 accum1.5x** | +23.33 | +12 | — | — | +32 | +26 | — | — | — | **OK** |
| 44 | **Flat subsets guidance accum8x** | +27.00 | +31 | — | — | +33 | +17 | — | — | — | **OK** |
| 45 | **Discrete** | +29.33 | +33 | — | — | +30 | +25 | — | — | — | **OK** |
| 46 | **Flat subsets EMA0.99 LB96/H720** | +33.00 | +36 | — | — | +35 | +28 | — | — | — | **OK** |
| 47 | **AR LB96/H720 accum1.5x** | +34.50 | +35 | — | — | +34 | — | — | — | — | **OK** |
| — | **MMPD (subset)** | — | — | — | — | — | — | — | — | — | ref |

### ETTh1

Baseline `sweep_baseline` rank: **12** / 48 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.3762 | 0.3936 | 0.2985 | — | ref |
| 2 | **Flat subsets accum1.5x LR-hi** | 0.3943 | 0.4005 | 0.2994 | -10 | **OK** |
| 3 | `diff_ema_decay_099` | 0.3974 | 0.4040 | 0.3021 | -9 | **OK** |
| 4 | **Flat subsets EMA0.99** | 0.3974 | 0.4040 | 0.3021 | -8 | **OK** |
| 5 | `hp_dit_dropout_01` | 0.3998 | 0.4056 | 0.3068 | -7 | **OK** |
| 6 | **Flat subsets accum2.0x LR-hi** | 0.4004 | 0.4034 | 0.3000 | -6 | **OK** |
| 7 | `hp_lr_cosine_warmup5` | 0.4045 | 0.4085 | 0.3064 | -5 | **OK** |
| 8 | **Flat subsets guidance accum1.5x** | 0.4046 | 0.4107 | 0.3069 | -4 | **OK** |
| 9 | **Flat subsets guidance accum2x** | 0.4049 | 0.4141 | 0.3091 | -3 | **OK** |
| 10 | **Flat subsets EMA0.90** | 0.4052 | 0.4082 | 0.3057 | -2 | **OK** |
| 11 | `hp_lr_cosine_warmup2` | 0.4054 | 0.4086 | 0.3069 | -1 | **OK** |
| 12 | `sweep_baseline` | 0.4059 | 0.4085 | 0.3060 | 0 | **OK** |
| 13 | **Binary flat** | 0.4059 | 0.4085 | 0.3060 | +1 | **OK** |
| 14 | **Flat subsets** | 0.4059 | 0.4085 | 0.3060 | +2 | **OK** |
| 15 | `hp_cfg_dropout_02` | 0.4059 | 0.4131 | 0.3080 | +3 | **OK** |
| 16 | `h16_16_16` | 0.4060 | 0.4086 | 0.3088 | +4 | **OK** |
| 17 | **Flat subsets EMA0.95** | 0.4061 | 0.4083 | 0.3061 | +5 | **OK** |
| 18 | **Flat subsets accum1.25x** | 0.4077 | 0.4116 | 0.3052 | +6 | **OK** |
| 19 | **Flat subsets guidance accum4x** | 0.4081 | 0.4169 | 0.3110 | +7 | **OK** |
| 20 | **Flat subsets EMA0.98** | 0.4084 | 0.4087 | 0.3066 | +8 | **OK** |
| 21 | `hp_beta_end_04` | 0.4091 | 0.4090 | 0.3116 | +9 | **OK** |
| 22 | **Flat subsets accum2.0x** | 0.4092 | 0.4105 | 0.3068 | +10 | **OK** |
| 23 | `hp_num_steps_1200` | 0.4092 | 0.4121 | 0.3100 | +11 | **OK** |
| 24 | **AR LB336/H96 accum1.5x** | 0.4093 | 0.4158 | 0.3078 | +12 | **OK** |
| 25 | **Flat subsets EMA0.99 LB336/H96** | 0.4109 | 0.4088 | 0.2951 | +13 | **OK** |
| 26 | `hp_ctxbias_neg01` | 0.4112 | 0.4097 | 0.3072 | +14 | **OK** |
| 27 | **MS tune** | 0.4117 | 0.4158 | 0.3114 | +15 | **pre-fix invalid** |
| 28 | **Flat subsets accum1.5x LR-lo** | 0.4122 | 0.4112 | 0.3080 | +16 | **OK** |
| 29 | `hp_anchor_lambda_095` | 0.4133 | 0.4114 | 0.3066 | +17 | **OK** |
| 30 | **Flat subsets EMA0.999** | 0.4140 | 0.4097 | 0.3087 | +18 | **OK** |
| 31 | `hp_num_steps_800` | 0.4146 | 0.4151 | 0.3098 | +19 | **OK** |
| 32 | **Flat subsets accum1.5x** | 0.4149 | 0.4118 | 0.3066 | +20 | **OK** |
| 33 | **Flat subsets EMA0.995** | 0.4158 | 0.4111 | 0.3090 | +21 | **OK** |
| 34 | `hp_anchor_lambda_090` | 0.4169 | 0.4124 | 0.3060 | +22 | **OK** |
| 35 | `diff_min_snr_gamma_5` | 0.4176 | 0.4144 | 0.3136 | +23 | **OK** |
| 36 | **Flat subsets accum2.0x LR-lo** | 0.4186 | 0.4124 | 0.3104 | +24 | **OK** |
| 37 | **MAE Discrete** | 0.4204 | 0.4116 | 0.7263 | +25 | **OK** |
| 38 | `diff_noise_cosine` | 0.4212 | 0.4196 | 0.3193 | +26 | **OK** |
| 39 | `diff_prediction_x0` | 0.4236 | 0.4101 | 0.3066 | +27 | **OK** |
| 40 | `hp_dit_embed288_heads4` | 0.4276 | 0.4221 | 0.3088 | +28 | **OK** |
| 41 | `hp_beta_end_03` | 0.4333 | 0.4188 | 0.4208 | +29 | **OK** |
| 42 | `hp_ctxbias_005` | 0.4456 | 0.4370 | 0.3157 | +30 | **OK** |
| 43 | **Flat subsets guidance accum8x** | 0.4506 | 0.4376 | 0.3281 | +31 | **OK** |
| 44 | `hp_dit_depth4` | 0.4799 | 0.4343 | 0.3173 | +32 | **OK** |
| 45 | **Discrete** | 0.5468 | 0.4577 | 0.7199 | +33 | **OK** |
| 46 | `h8_8_8` | 0.6140 | 0.4792 | 0.3435 | +34 | **OK** |
| 47 | **AR LB96/H720 accum1.5x** | 0.7242 | 0.5749 | 0.4608 | +35 | **OK** |
| 48 | **Flat subsets EMA0.99 LB96/H720** | 0.7247 | 0.5637 | 0.4868 | +36 | **OK** |

### ETTh2

Baseline `sweep_baseline` missing. Total configs: 24

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets guidance accum2x** | 0.3011 | 0.3397 | 0.2685 | — | **OK** |
| 2 | **Flat subsets guidance accum1.5x** | 0.3016 | 0.3395 | 0.2678 | — | **OK** |
| 3 | **Flat subsets accum1.5x LR-hi** | 0.3017 | 0.3394 | 0.2631 | — | **OK** |
| 4 | **Flat subsets accum2.0x LR-hi** | 0.3029 | 0.3423 | 0.2656 | — | **OK** |
| 5 | **Flat subsets guidance accum4x** | 0.3076 | 0.3538 | 0.2801 | — | **OK** |
| 6 | **Flat subsets EMA0.99** | 0.3116 | 0.3500 | 0.2673 | — | **OK** |
| 7 | **Flat subsets EMA0.95** | 0.3121 | 0.3509 | 0.2686 | — | **OK** |
| 8 | **Flat subsets EMA0.90** | 0.3124 | 0.3513 | 0.2684 | — | **OK** |
| 9 | **Flat subsets EMA0.98** | 0.3125 | 0.3507 | 0.2690 | — | **OK** |
| 10 | **Flat subsets EMA0.995** | 0.3134 | 0.3523 | 0.2699 | — | **OK** |
| 11 | **Flat subsets guidance accum8x** | 0.3148 | 0.3619 | 0.2941 | — | **OK** |
| 12 | **Flat subsets accum1.25x** | 0.3169 | 0.3545 | 0.2754 | — | **OK** |
| 13 | **Flat subsets accum1.5x LR-lo** | 0.3183 | 0.3595 | 0.2739 | — | **OK** |
| 14 | **MAE Discrete** | 0.3183 | 0.3497 | 0.8584 | — | **OK** |
| 15 | **MMPD (subset)** | 0.3186 | 0.3614 | 0.2705 | — | ref |
| 16 | **Binary flat** | 0.3199 | 0.3546 | 0.2705 | — | **OK** |
| 17 | **Flat subsets** | 0.3199 | 0.3546 | 0.2705 | — | **OK** |
| 18 | **Flat subsets accum2.0x** | 0.3200 | 0.3621 | 0.2780 | — | **OK** |
| 19 | **Flat subsets EMA0.999** | 0.3250 | 0.3647 | 0.2799 | — | **OK** |
| 20 | **Flat subsets accum2.0x LR-lo** | 0.3272 | 0.3674 | 0.2841 | — | **OK** |
| 21 | **Flat subsets accum1.5x** | 0.3275 | 0.3676 | 0.2872 | — | **OK** |
| 22 | **AR LB336/H96 accum1.5x** | 0.3373 | 0.3690 | 0.2855 | — | **OK** |
| 23 | **Discrete** | 0.3397 | 0.3582 | 0.6547 | — | **OK** |
| 24 | **AR LB96/H720 accum1.5x** | 0.4503 | 0.4575 | 0.3846 | — | **OK** |

### ETTm1

Baseline `sweep_baseline` rank: **8** / 23 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD** | 0.4208 | 0.4122 | 0.3109 | — | ref |
| 2 | `hp_anchor_lambda_095` | 0.4522 | 0.4210 | 0.3200 | -6 | **OK** |
| 3 | `diff_ema_decay_099` | 0.4556 | 0.4231 | 0.3290 | -5 | **OK** |
| 4 | `hp_anchor_lambda_090` | 0.4661 | 0.4317 | 0.3407 | -4 | **OK** |
| 5 | `hp_num_steps_800` | 0.4670 | 0.4274 | 0.3382 | -3 | **OK** |
| 6 | **MS tune** | 0.4679 | 0.4246 | 0.3176 | -2 | **OK** |
| 7 | `hp_ctxbias_005` | 0.4681 | 0.4236 | 0.3278 | -1 | **OK** |
| 8 | `sweep_baseline` | 0.4683 | 0.4259 | 0.3268 | 0 | **OK** |
| 9 | `h16_16_16` | 0.4724 | 0.4307 | 0.3365 | +1 | **OK** |
| 10 | `hp_num_steps_1200` | 0.4754 | 0.4289 | 0.3353 | +2 | **OK** |
| 11 | `hp_cfg_dropout_02` | 0.4785 | 0.4341 | 0.3281 | +3 | **OK** |
| 12 | `hp_ctxbias_neg01` | 0.4807 | 0.4288 | 0.3299 | +4 | **OK** |
| 13 | `hp_lr_cosine_warmup5` | 0.4857 | 0.4361 | 0.3453 | +5 | **OK** |
| 14 | `hp_lr_cosine_warmup2` | 0.4866 | 0.4360 | 0.3451 | +6 | **OK** |
| 15 | `diff_noise_cosine` | 0.4894 | 0.4368 | 0.3287 | +7 | **OK** |
| 16 | `hp_beta_end_04` | 0.4989 | 0.4383 | 0.5073 | +8 | **OK** |
| 17 | `hp_dit_dropout_01` | 0.5045 | 0.4394 | 0.3410 | +9 | **OK** |
| 18 | `diff_prediction_x0` | 0.5051 | 0.4334 | 0.3355 | +10 | **OK** |
| 19 | `hp_dit_depth4` | 0.5066 | 0.4498 | 0.3360 | +11 | **OK** |
| 20 | `h8_8_8` | 0.5236 | 0.4546 | 0.3306 | +12 | **OK** |
| 21 | `diff_min_snr_gamma_5` | 0.5463 | 0.4584 | 0.4031 | +13 | **OK** |
| 22 | `hp_dit_embed288_heads4` | 0.5516 | 0.4596 | 0.3396 | +14 | **OK** |
| 23 | `hp_beta_end_03` | 0.6789 | 0.5006 | 0.9025 | +15 | **OK** |

### exchange_rate

Baseline `sweep_baseline` rank: **13** / 48 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.0810 | 0.1987 | 0.1563 | — | ref |
| 2 | `hp_anchor_lambda_090` | 0.0825 | 0.2014 | 0.1656 | -11 | **OK** |
| 3 | `hp_beta_end_04` | 0.0829 | 0.2035 | 0.2460 | -10 | **OK** |
| 4 | **MAE Discrete** | 0.0845 | 0.2039 | 0.4216 | -9 | **OK** |
| 5 | `hp_ctxbias_neg01` | 0.0857 | 0.2042 | 0.1647 | -8 | **OK** |
| 6 | `hp_cfg_dropout_02` | 0.0858 | 0.2053 | 0.1691 | -7 | **OK** |
| 7 | **Flat subsets EMA0.90** | 0.0868 | 0.2071 | 0.1699 | -6 | **OK** |
| 8 | **Flat subsets guidance accum4x** | 0.0873 | 0.2072 | 0.1690 | -5 | **OK** |
| 9 | **Flat subsets EMA0.995** | 0.0875 | 0.2074 | 0.1667 | -4 | **OK** |
| 10 | **Flat subsets accum2.0x** | 0.0875 | 0.2073 | 0.1687 | -3 | **OK** |
| 11 | **Flat subsets EMA0.95** | 0.0875 | 0.2077 | 0.1689 | -2 | **OK** |
| 12 | **Flat subsets EMA0.98** | 0.0878 | 0.2078 | 0.1684 | -1 | **OK** |
| 13 | `sweep_baseline` | 0.0880 | 0.2078 | 0.1660 | 0 | **OK** |
| 14 | **Binary flat** | 0.0880 | 0.2078 | 0.1660 | +1 | **OK** |
| 15 | **Flat subsets** | 0.0880 | 0.2078 | 0.1660 | +2 | **OK** |
| 16 | `hp_anchor_lambda_095` | 0.0881 | 0.2096 | 0.1701 | +3 | **OK** |
| 17 | **Flat subsets accum1.5x** | 0.0882 | 0.2071 | 0.1657 | +4 | **OK** |
| 18 | **Flat subsets accum2.0x LR-lo** | 0.0889 | 0.2084 | 0.1655 | +5 | **OK** |
| 19 | `h16_16_16` | 0.0892 | 0.2090 | 0.1760 | +6 | **OK** |
| 20 | `hp_num_steps_1200` | 0.0893 | 0.2081 | 0.1686 | +7 | **OK** |
| 21 | `diff_ema_decay_099` | 0.0893 | 0.2086 | 0.1684 | +8 | **OK** |
| 22 | **Flat subsets EMA0.99** | 0.0893 | 0.2086 | 0.1684 | +9 | **OK** |
| 23 | **Flat subsets accum1.25x** | 0.0895 | 0.2090 | 0.1698 | +10 | **OK** |
| 24 | `hp_ctxbias_005` | 0.0896 | 0.2107 | 0.1684 | +11 | **OK** |
| 25 | `hp_lr_cosine_warmup5` | 0.0898 | 0.2090 | 0.1673 | +12 | **OK** |
| 26 | `hp_num_steps_800` | 0.0898 | 0.2115 | 0.1734 | +13 | **OK** |
| 27 | `hp_lr_cosine_warmup2` | 0.0899 | 0.2094 | 0.1677 | +14 | **OK** |
| 28 | **Flat subsets accum2.0x LR-hi** | 0.0900 | 0.2106 | 0.1690 | +15 | **OK** |
| 29 | `h8_8_8` | 0.0901 | 0.2102 | 0.1614 | +16 | **OK** |
| 30 | **Flat subsets accum1.5x LR-hi** | 0.0901 | 0.2096 | 0.1686 | +17 | **OK** |
| 31 | **Flat subsets guidance accum1.5x** | 0.0904 | 0.2118 | 0.1716 | +18 | **OK** |
| 32 | `diff_noise_cosine` | 0.0905 | 0.2094 | 0.1788 | +19 | **OK** |
| 33 | `diff_min_snr_gamma_5` | 0.0907 | 0.2125 | 0.1712 | +20 | **OK** |
| 34 | **Flat subsets EMA0.999** | 0.0909 | 0.2107 | 0.1703 | +21 | **OK** |
| 35 | **Flat subsets guidance accum2x** | 0.0910 | 0.2111 | 0.1722 | +22 | **OK** |
| 36 | **MS tune** | 0.0910 | 0.2114 | 0.1676 | +23 | **OK** |
| 37 | `hp_beta_end_03` | 0.0911 | 0.2138 | 0.2791 | +24 | **OK** |
| 38 | `hp_dit_depth4` | 0.0913 | 0.2108 | 0.1703 | +25 | **OK** |
| 39 | **Flat subsets accum1.5x LR-lo** | 0.0913 | 0.2109 | 0.1708 | +26 | **OK** |
| 40 | **Flat subsets EMA0.99 LB336/H96** | 0.0922 | 0.2141 | 0.1692 | +27 | **OK** |
| 41 | `hp_dit_dropout_01` | 0.0924 | 0.2134 | 0.1754 | +28 | **OK** |
| 42 | `diff_prediction_x0` | 0.0924 | 0.2130 | 0.1700 | +29 | **OK** |
| 43 | **Discrete** | 0.0924 | 0.2132 | 0.3384 | +30 | **OK** |
| 44 | `hp_dit_embed288_heads4` | 0.0925 | 0.2151 | 0.1707 | +31 | **OK** |
| 45 | **AR LB336/H96 accum1.5x** | 0.1035 | 0.2224 | 0.1867 | +32 | **OK** |
| 46 | **Flat subsets guidance accum8x** | 0.1054 | 0.2328 | 0.1704 | +33 | **OK** |
| 47 | **AR LB96/H720 accum1.5x** | 0.8445 | 0.6937 | 0.6506 | +34 | **OK** |
| 48 | **Flat subsets EMA0.99 LB96/H720** | 0.9771 | 0.7582 | 0.7219 | +35 | **OK** |

### weather

Baseline `sweep_baseline` rank: **19** / 47 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets guidance accum2x** | 0.0938 | 0.2172 | 0.1730 | -18 | **OK** |
| 2 | **Flat subsets EMA0.999** | 0.0945 | 0.2193 | 0.1759 | -17 | **OK** |
| 3 | **Flat subsets guidance accum1.5x** | 0.0945 | 0.2176 | 0.1740 | -16 | **OK** |
| 4 | **Flat subsets guidance accum4x** | 0.0959 | 0.2195 | 0.1756 | -15 | **OK** |
| 5 | `diff_noise_cosine` | 0.0962 | 0.2192 | 0.1794 | -14 | **OK** |
| 6 | `hp_dit_embed288_heads4` | 0.0963 | 0.2201 | 0.1734 | -13 | **OK** |
| 7 | **Flat subsets accum1.5x LR-hi** | 0.0963 | 0.2207 | 0.1746 | -12 | **OK** |
| 8 | **Flat subsets EMA0.995** | 0.0964 | 0.2213 | 0.1752 | -11 | **OK** |
| 9 | **Flat subsets EMA0.95** | 0.0964 | 0.2217 | 0.1792 | -10 | **OK** |
| 10 | **Flat subsets EMA0.98** | 0.0966 | 0.2216 | 0.1760 | -9 | **OK** |
| 11 | **Flat subsets EMA0.90** | 0.0968 | 0.2222 | 0.1803 | -8 | **OK** |
| 12 | `hp_beta_end_04` | 0.0969 | 0.2203 | 0.2248 | -7 | **OK** |
| 13 | `diff_ema_decay_099` | 0.0971 | 0.2220 | 0.1758 | -6 | **OK** |
| 14 | **Flat subsets EMA0.99** | 0.0971 | 0.2220 | 0.1758 | -5 | **OK** |
| 15 | **Flat subsets accum1.5x LR-lo** | 0.0973 | 0.2233 | 0.1775 | -4 | **OK** |
| 16 | **Flat subsets accum1.25x** | 0.0975 | 0.2229 | 0.1777 | -3 | **OK** |
| 17 | **Flat subsets accum2.0x LR-lo** | 0.0976 | 0.2234 | 0.1765 | -2 | **OK** |
| 18 | **Flat subsets accum2.0x LR-hi** | 0.0978 | 0.2226 | 0.1760 | -1 | **OK** |
| 19 | `sweep_baseline` | 0.0978 | 0.2224 | 0.1776 | 0 | **OK** |
| 20 | **Binary flat** | 0.0978 | 0.2224 | 0.1776 | +1 | **OK** |
| 21 | **Flat subsets** | 0.0978 | 0.2224 | 0.1776 | +2 | **OK** |
| 22 | **Flat subsets accum2.0x** | 0.0979 | 0.2234 | 0.1858 | +3 | **OK** |
| 23 | **MAE Discrete** | 0.0980 | 0.2195 | 0.4609 | +4 | **OK** |
| 24 | `hp_cfg_dropout_02` | 0.0981 | 0.2215 | 0.1795 | +5 | **OK** |
| 25 | `h16_16_16` | 0.0981 | 0.2216 | 0.1767 | +6 | **OK** |
| 26 | **MS tune** | 0.0987 | 0.2185 | 0.1775 | +7 | **pre-fix invalid** |
| 27 | `hp_dit_depth4` | 0.0989 | 0.2227 | 0.1801 | +8 | **OK** |
| 28 | `diff_min_snr_gamma_5` | 0.0992 | 0.2243 | 0.1804 | +9 | **OK** |
| 29 | `hp_ctxbias_005` | 0.0992 | 0.2212 | 0.1746 | +10 | **OK** |
| 30 | `hp_anchor_lambda_095` | 0.0993 | 0.2231 | 0.1779 | +11 | **OK** |
| 31 | **Flat subsets accum1.5x** | 0.0994 | 0.2251 | 0.1759 | +12 | **OK** |
| 32 | `hp_ctxbias_neg01` | 0.0996 | 0.2243 | 0.1778 | +13 | **OK** |
| 33 | `hp_lr_cosine_warmup5` | 0.0998 | 0.2244 | 0.1785 | +14 | **OK** |
| 34 | `hp_lr_cosine_warmup2` | 0.1000 | 0.2246 | 0.1784 | +15 | **OK** |
| 35 | `hp_anchor_lambda_090` | 0.1001 | 0.2235 | 0.1757 | +16 | **OK** |
| 36 | **Flat subsets guidance accum8x** | 0.1007 | 0.2260 | 0.1879 | +17 | **OK** |
| 37 | `hp_num_steps_800` | 0.1009 | 0.2247 | 0.1818 | +18 | **OK** |
| 38 | `hp_beta_end_03` | 0.1023 | 0.2277 | 0.2997 | +19 | **OK** |
| 39 | `hp_dit_dropout_01` | 0.1031 | 0.2268 | 0.1808 | +20 | **OK** |
| 40 | `diff_prediction_x0` | 0.1036 | 0.2276 | 0.1791 | +21 | **OK** |
| 41 | `hp_num_steps_1200` | 0.1037 | 0.2270 | 0.1832 | +22 | **OK** |
| 42 | `h8_8_8` | 0.1043 | 0.2282 | 0.1785 | +23 | **OK** |
| 43 | **Flat subsets EMA0.99 LB336/H96** | 0.1053 | 0.2302 | 0.1807 | +24 | **OK** |
| 44 | **Discrete** | 0.1079 | 0.2280 | 0.3707 | +25 | **OK** |
| 45 | **AR LB336/H96 accum1.5x** | 0.1113 | 0.2312 | 0.1932 | +26 | **OK** |
| 46 | **MMPD (subset)** | 0.1128 | 0.2323 | 0.1911 | — | ref |
| 47 | **Flat subsets EMA0.99 LB96/H720** | 0.5848 | 0.5399 | 0.4771 | +28 | **OK** |

### electricity

Baseline `sweep_baseline` missing. Total configs: 23

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **MMPD (subset)** | 0.1617 | 0.2088 | 0.1610 | — | ref |
| 2 | **AR LB336/H96 accum1.5x** | 0.1638 | 0.2048 | 0.1534 | — | **OK** |
| 3 | **Flat subsets accum1.5x LR-hi** | 0.1643 | 0.1995 | 0.1504 | — | **OK** |
| 4 | **Flat subsets accum2.0x LR-hi** | 0.1657 | 0.2003 | 0.1503 | — | **OK** |
| 5 | **Flat subsets guidance accum2x** | 0.1671 | 0.2038 | 0.1537 | — | **OK** |
| 6 | **Flat subsets guidance accum1.5x** | 0.1678 | 0.2023 | 0.1537 | — | **OK** |
| 7 | **Flat subsets EMA0.95** | 0.1690 | 0.2088 | 0.1560 | — | **OK** |
| 8 | **Flat subsets EMA0.98** | 0.1692 | 0.2091 | 0.1562 | — | **OK** |
| 9 | **Flat subsets EMA0.90** | 0.1693 | 0.2086 | 0.1561 | — | **OK** |
| 10 | **Flat subsets EMA0.995** | 0.1706 | 0.2101 | 0.1566 | — | **OK** |
| 11 | **Flat subsets EMA0.99** | 0.1713 | 0.2111 | 0.1572 | — | **OK** |
| 12 | **MAE Discrete** | 0.1714 | 0.2001 | 0.5281 | — | **OK** |
| 13 | **Flat subsets** | 0.1735 | 0.2132 | 0.1602 | — | **OK** |
| 14 | **Flat subsets accum1.25x** | 0.1737 | 0.2140 | 0.1600 | — | **OK** |
| 15 | **Flat subsets guidance accum4x** | 0.1742 | 0.2123 | 0.1595 | — | **OK** |
| 16 | **Flat subsets accum1.5x** | 0.1820 | 0.2213 | 0.1629 | — | **OK** |
| 17 | **Flat subsets accum1.5x LR-lo** | 0.1824 | 0.2215 | 0.1607 | — | **OK** |
| 18 | **Flat subsets accum2.0x** | 0.1843 | 0.2245 | 0.1617 | — | **OK** |
| 19 | **Flat subsets accum2.0x LR-lo** | 0.1962 | 0.2347 | 0.1633 | — | **OK** |
| 20 | **Flat subsets EMA0.999** | 0.1974 | 0.2361 | 0.1619 | — | **OK** |
| 21 | **Flat subsets guidance accum8x** | 0.2156 | 0.2554 | 0.1873 | — | **OK** |
| 22 | **Binary flat** | 0.2598 | 0.2713 | 0.2080 | — | **OK** |
| 23 | **Discrete** | 0.2972 | 0.2712 | 0.5994 | — | **OK** |

### traffic

Baseline `sweep_baseline` missing. Total configs: 26

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Discrete** | 0.3863 | 0.2382 | 1.2598 | — | **OK** |
| 2 | **Binary flat** | 0.4184 | 0.2813 | 0.2121 | — | **OK** |
| 3 | **MAE Discrete** | 0.4356 | 0.2597 | 1.0900 | — | **OK** |
| 4 | **AR LB336/H96 accum1.5x** | 0.4423 | 0.2763 | 0.2092 | — | **OK** |
| 5 | **Flat subsets accum1.5x LR-hi** | 0.4533 | 0.2753 | 0.2086 | — | **OK** |
| 6 | **Flat subsets accum2.0x LR-hi** | 0.4573 | 0.2800 | 0.2111 | — | **OK** |
| 7 | **Flat subsets EMA0.99 LB336/H96** | 0.4791 | 0.2912 | 0.2214 | — | **OK** |
| 8 | **Flat subsets guidance accum1.5x** | 0.4845 | 0.2948 | 0.2244 | — | **OK** |
| 9 | **Flat subsets guidance accum2x** | 0.5007 | 0.3052 | 0.2312 | — | **OK** |
| 10 | **Flat subsets EMA0.98** | 0.5152 | 0.3161 | 0.2367 | — | **OK** |
| 11 | **Flat subsets EMA0.95** | 0.5167 | 0.3161 | 0.2376 | — | **OK** |
| 12 | **Flat subsets EMA0.90** | 0.5208 | 0.3168 | 0.2398 | — | **OK** |
| 13 | **Flat subsets EMA0.995** | 0.5219 | 0.3196 | 0.2397 | — | **OK** |
| 14 | **MMPD (subset)** | 0.5225 | 0.3612 | 0.2515 | — | ref |
| 15 | **Flat subsets** | 0.5263 | 0.3252 | 0.2430 | — | **OK** |
| 16 | **Flat subsets EMA0.99** | 0.5296 | 0.3245 | 0.2436 | — | **OK** |
| 17 | **Flat subsets accum1.25x** | 0.5318 | 0.3318 | 0.2481 | — | **OK** |
| 18 | **Flat subsets guidance accum4x** | 0.5417 | 0.3336 | 0.2605 | — | **OK** |
| 19 | **Flat subsets accum1.5x** | 0.5456 | 0.3373 | 0.2556 | — | **OK** |
| 20 | **Flat subsets accum1.5x LR-lo** | 0.5790 | 0.3489 | 0.2585 | — | **OK** |
| 21 | **Flat subsets accum2.0x** | 0.5998 | 0.3526 | 0.2710 | — | **OK** |
| 22 | **Flat subsets accum2.0x LR-lo** | 0.6120 | 0.3635 | 0.2655 | — | **OK** |
| 23 | **Flat subsets guidance accum8x** | 0.7107 | 0.4152 | 0.3299 | — | **OK** |
| 24 | **Flat subsets EMA0.999** | 0.8891 | 0.4865 | 0.3351 | — | **OK** |
| 25 | **AR LB96/H720 accum1.5x** | 1.3981 | 0.7447 | 0.6410 | — | **OK** |
| 26 | **Flat subsets EMA0.99 LB96/H720** | 1.4590 | 0.7630 | 0.5796 | — | **OK** |

### solar_Alabama

Baseline `sweep_baseline` missing. Total configs: 23

| Rank | Config | anchor_mse | anchor_mae | crps | Δrank | Status |
|---|---|---|---|---|---|---|
| 1 | **Flat subsets accum1.5x** | 0.2002 | 0.2313 | 0.1789 | — | **OK** |
| 2 | **AR LB336/H96 accum1.5x** | 0.2041 | 0.2229 | 0.1796 | — | **OK** |
| 3 | **Flat subsets accum1.5x LR-hi** | 0.2078 | 0.2283 | 0.1861 | — | **OK** |
| 4 | **Flat subsets EMA0.999** | 0.2108 | 0.2369 | 0.1912 | — | **OK** |
| 5 | **Flat subsets accum1.5x LR-lo** | 0.2110 | 0.2379 | 0.1916 | — | **OK** |
| 6 | **Flat subsets accum2.0x LR-hi** | 0.2111 | 0.2321 | 0.1868 | — | **OK** |
| 7 | **Flat subsets EMA0.99** | 0.2123 | 0.2366 | 0.1890 | — | **OK** |
| 8 | **Flat subsets EMA0.98** | 0.2126 | 0.2371 | 0.1881 | — | **OK** |
| 9 | **Flat subsets EMA0.995** | 0.2129 | 0.2393 | 0.1932 | — | **OK** |
| 10 | **Flat subsets guidance accum2x** | 0.2132 | 0.2367 | 0.1879 | — | **OK** |
| 11 | **Flat subsets EMA0.95** | 0.2136 | 0.2379 | 0.1881 | — | **OK** |
| 12 | **Flat subsets accum2.0x LR-lo** | 0.2136 | 0.2398 | 0.1933 | — | **OK** |
| 13 | **Flat subsets guidance accum1.5x** | 0.2144 | 0.2347 | 0.1890 | — | **OK** |
| 14 | **Flat subsets EMA0.90** | 0.2147 | 0.2387 | 0.1884 | — | **OK** |
| 15 | **Flat subsets accum2.0x** | 0.2158 | 0.2394 | 0.1876 | — | **OK** |
| 16 | **Flat subsets guidance accum4x** | 0.2169 | 0.2415 | 0.1924 | — | **OK** |
| 17 | **Binary flat** | 0.2170 | 0.2426 | 0.1945 | — | **OK** |
| 18 | **Flat subsets** | 0.2170 | 0.2426 | 0.1945 | — | **OK** |
| 19 | **Flat subsets accum1.25x** | 0.2231 | 0.2439 | 0.1977 | — | **OK** |
| 20 | **Flat subsets guidance accum8x** | 0.2246 | 0.2498 | 0.2010 | — | **OK** |
| 21 | **MMPD (subset)** | 0.2360 | 0.2690 | 0.2013 | — | ref |
| 22 | **MAE Discrete** | 0.2540 | 0.3142 | 2.4934 | — | **OK** |
| 23 | **Discrete** | 0.2560 | 0.2472 | 1.5170 | — | **OK** |

