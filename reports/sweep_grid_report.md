# YAML-First Sweep Grid Report

Fixed-HP binary sweep (`configs/sweep/`, Jun 12 2026) plus ordinal D3PM staged runs: **Discrete** (CE, `ordinal_d3pm_staged`), **MAE Discrete** (expectation MAE + uniform `1/H` anchor, `ordinal_d3pm_mae_staged_subsets`), **Binary flat** (full variates, `binary_anchor_stationary_flat`), **Flat subsets** (`3951193`–`3951199`), **Flat subsets EMA0.99** (`3951527`–`3951533`), EMA reuse sweep (`ema_sweep_{090,095,098,0995,0999}`, jobs `3953317`–`3953351`), grad-accum reuse sweep (`grad_accum_{125,150,200}`, jobs `3953944`–`3953964`; LR-band split `grad_accum_{150,200}_lr_{lo,hi}`, jobs `3954784`–`3954810`), **Flat subsets guidance accum** {1.5×, 2×, 4×, 8×} (`grad_accum_guidance_{150,200,400,800}`, jobs `3961419`–`3961447`), **Flat subsets accum4x** no guidance (`grad_accum_400`, jobs `3963967`–`3963973`), EMA0.99 lookback variants (`ema099_lb336_hz96`, `ema099_lb96_hz720`, jobs `3955091`–`3955098`), **AR accum4x/8x** (`binary_anchor_ar_grad_accum_{400,800}`, LB96/H96 base), **AR LB336/H96 accum1.5x** (`3961448`–`3961454`), **AR LB96/H720 accum1.5x** (`3961455`, `3961457`, `3961460`), **MS tune** (`hp_max_scale_tuning`), and **MMPD (subset)** (`06-13-binary-mmpd-subset-compare`, jobs `3951201`–`3951207`). Probabilistic metrics: `dpmpp` sampler, 20 steps, 20 samples.

**Pre-fix invalid / incomplete:** Jun 12 `hp_max_scale_tuning` jobs `3943934`–`3943937` never searched `max_scale`. Post-fix cosine+warmup `3956633`–`3956640` replaces pre-fix `3943882`–`3943927`. Post-fix MS tune: exchange_rate `3956631` only; ETTh1/ETTm1/weather `3956629`–`3956632` hit 3h wall before eval — still showing pre-fix rows.

| Dataset | Config | Status | anchor_mse | anchor_mae | crps | sample_mean_mse | Job |
|---|---|---|---|---|---|---|---|
| ETTh1 | **AR LB336/H96 accum1.5x** | **OK** | 0.4093 | 0.4158 | 0.3078 | 0.4001 | 3961448 |
| ETTh1 | **AR LB96/H720 accum1.5x** | **OK** | 0.7242 | 0.5749 | 0.4608 | 0.7137 | 3961455 |
| ETTh1 | **Binary flat** | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3949852 |
| ETTh1 | **Discrete** | **OK** | 0.5468 | 0.4577 | 0.7199 | 1.0323 | 3948454 |
| ETTh1 | **Flat subsets EMA0.90** | **OK** | 0.4052 | 0.4082 | 0.3057 | 0.4102 | 3953317 |
| ETTh1 | **Flat subsets EMA0.95** | **OK** | 0.4061 | 0.4083 | 0.3061 | 0.4117 | 3953324 |
| ETTh1 | **Flat subsets EMA0.98** | **OK** | 0.4084 | 0.4087 | 0.3066 | 0.4133 | 3953331 |
| ETTh1 | **Flat subsets EMA0.99 LB336/H96** | **OK** | 0.4109 | 0.4088 | 0.2951 | 0.3881 | 3955091 |
| ETTh1 | **Flat subsets EMA0.99 LB96/H720** | **OK** | 0.7247 | 0.5637 | 0.4868 | 0.7028 | 3955095 |
| ETTh1 | **Flat subsets EMA0.99** | **OK** | 0.3974 | 0.4040 | 0.3021 | 0.4047 | 3951527 |
| ETTh1 | **Flat subsets EMA0.995** | **OK** | 0.4158 | 0.4111 | 0.3090 | 0.4205 | 3953338 |
| ETTh1 | **Flat subsets EMA0.999** | **OK** | 0.4140 | 0.4097 | 0.3087 | 0.4214 | 3953345 |
| ETTh1 | **Flat subsets accum1.25x** | **OK** | 0.4077 | 0.4116 | 0.3052 | 0.4085 | 3953944 |
| ETTh1 | **Flat subsets accum1.5x LR-hi** | **OK** | 0.3943 | 0.4005 | 0.2994 | 0.3976 | 3954791 |
| ETTh1 | **Flat subsets accum1.5x LR-lo** | **OK** | 0.4122 | 0.4112 | 0.3080 | 0.4179 | 3954784 |
| ETTh1 | **Flat subsets accum1.5x** | **OK** | 0.4149 | 0.4118 | 0.3066 | 0.4145 | 3953951 |
| ETTh1 | **Flat subsets accum2.0x LR-hi** | **OK** | 0.4004 | 0.4034 | 0.3000 | 0.3999 | 3954805 |
| ETTh1 | **Flat subsets accum2.0x LR-lo** | **OK** | 0.4186 | 0.4124 | 0.3104 | 0.4239 | 3954798 |
| ETTh1 | **Flat subsets accum2.0x** | **OK** | 0.4092 | 0.4105 | 0.3068 | 0.4154 | 3953958 |
| ETTh1 | **Flat subsets accum4x** | **OK** | 0.4012 | 0.4053 | 0.3020 | 0.4049 | 3963967 |
| ETTh1 | **Flat subsets guidance accum1.5x** | **OK** | 0.4046 | 0.4107 | 0.3069 | 0.4098 | 3961419 |
| ETTh1 | **Flat subsets guidance accum2x** | **OK** | 0.4049 | 0.4141 | 0.3091 | 0.4075 | 3961427 |
| ETTh1 | **Flat subsets guidance accum4x** | **OK** | 0.4081 | 0.4169 | 0.3110 | 0.4071 | 3961434 |
| ETTh1 | **Flat subsets guidance accum8x** | **OK** | 0.4506 | 0.4376 | 0.3281 | 0.4402 | 3961441 |
| ETTh1 | **Flat subsets** | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3951193 |
| ETTh1 | **MAE Discrete** | **OK** | 0.4204 | 0.4116 | 0.7263 | 1.0794 | 3949859 |
| ETTh1 | **MMPD (subset)** | **ref** | 0.3762 | 0.3936 | 0.2985 | — | 3951201 |
| ETTh1 | **MS tune** | pre-fix invalid | 0.4117 | 0.4158 | 0.3114 | 0.4058 | 3943934 |
| ETTh1 | `diff_ema_decay_099` | **OK** | 0.3974 | 0.4040 | 0.3021 | 0.4047 | 3943854 |
| ETTh1 | `diff_min_snr_gamma_5` | **OK** | 0.4176 | 0.4144 | 0.3136 | 0.4268 | 3943856 |
| ETTh1 | `diff_noise_cosine` | **OK** | 0.4212 | 0.4196 | 0.3193 | 0.4378 | 3943858 |
| ETTh1 | `diff_prediction_x0` | **OK** | 0.4236 | 0.4101 | 0.3066 | 0.4157 | 3943860 |
| ETTh1 | `h16_16_16` | **OK** | 0.4060 | 0.4086 | 0.3088 | 0.4153 | 3943850 |
| ETTh1 | `h8_8_8` | **OK** | 0.6140 | 0.4792 | 0.3435 | 0.5279 | 3943852 |
| ETTh1 | `hp_anchor_lambda_090` | **OK** | 0.4169 | 0.4124 | 0.3060 | 0.4093 | 3943862 |
| ETTh1 | `hp_anchor_lambda_095` | **OK** | 0.4133 | 0.4114 | 0.3066 | 0.4111 | 3943864 |
| ETTh1 | `hp_beta_end_03` | **OK** | 0.4333 | 0.4188 | 0.4208 | 0.5942 | 3943866 |
| ETTh1 | `hp_beta_end_04` | **OK** | 0.4091 | 0.4090 | 0.3116 | 0.4325 | 3943868 |
| ETTh1 | `hp_cfg_dropout_02` | **OK** | 0.4059 | 0.4131 | 0.3080 | 0.4143 | 3943870 |
| ETTh1 | `hp_ctxbias_005` | **OK** | 0.4456 | 0.4370 | 0.3157 | 0.4416 | 3943872 |
| ETTh1 | `hp_ctxbias_neg01` | **OK** | 0.4112 | 0.4097 | 0.3072 | 0.4116 | 3943874 |
| ETTh1 | `hp_dit_depth4` | **OK** | 0.4799 | 0.4343 | 0.3173 | 0.4402 | 3943876 |
| ETTh1 | `hp_dit_dropout_01` | **OK** | 0.3998 | 0.4056 | 0.3068 | 0.4091 | 3943878 |
| ETTh1 | `hp_dit_embed288_heads4` | **OK** | 0.4276 | 0.4221 | 0.3088 | 0.4260 | 3943880 |
| ETTh1 | `hp_lr_cosine_warmup2` | **OK** | 0.4054 | 0.4086 | 0.3069 | 0.4143 | 3956633 |
| ETTh1 | `hp_lr_cosine_warmup5` | **OK** | 0.4045 | 0.4085 | 0.3064 | 0.4132 | 3956637 |
| ETTh1 | `hp_num_steps_1200` | **OK** | 0.4092 | 0.4121 | 0.3100 | 0.4244 | 3943886 |
| ETTh1 | `hp_num_steps_800` | **OK** | 0.4146 | 0.4151 | 0.3098 | 0.4260 | 3943888 |
| ETTh1 | `sweep_baseline` | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3943890 |
| ETTh2 | **AR LB336/H96 accum1.5x** | **OK** | 0.3373 | 0.3690 | 0.2855 | 0.3281 | 3961449 |
| ETTh2 | **AR LB96/H720 accum1.5x** | **OK** | 0.4503 | 0.4575 | 0.3846 | 0.4490 | 3961456 |
| ETTh2 | **Binary flat** | **OK** | 0.3199 | 0.3546 | 0.2705 | 0.3104 | 3949853 |
| ETTh2 | **Discrete** | **OK** | 0.3397 | 0.3582 | 0.6547 | 0.6292 | 3948455 |
| ETTh2 | **Flat subsets EMA0.90** | **OK** | 0.3124 | 0.3513 | 0.2684 | 0.3008 | 3953318 |
| ETTh2 | **Flat subsets EMA0.95** | **OK** | 0.3121 | 0.3509 | 0.2686 | 0.3019 | 3953325 |
| ETTh2 | **Flat subsets EMA0.98** | **OK** | 0.3125 | 0.3507 | 0.2690 | 0.3030 | 3953332 |
| ETTh2 | **Flat subsets EMA0.99** | **OK** | 0.3116 | 0.3500 | 0.2673 | 0.3007 | 3951528 |
| ETTh2 | **Flat subsets EMA0.995** | **OK** | 0.3134 | 0.3523 | 0.2699 | 0.3035 | 3953339 |
| ETTh2 | **Flat subsets EMA0.999** | **OK** | 0.3250 | 0.3647 | 0.2799 | 0.3150 | 3953346 |
| ETTh2 | **Flat subsets accum1.25x** | **OK** | 0.3169 | 0.3545 | 0.2754 | 0.3102 | 3953945 |
| ETTh2 | **Flat subsets accum1.5x LR-hi** | **OK** | 0.3017 | 0.3394 | 0.2631 | 0.2949 | 3954792 |
| ETTh2 | **Flat subsets accum1.5x LR-lo** | **OK** | 0.3183 | 0.3595 | 0.2739 | 0.3053 | 3954785 |
| ETTh2 | **Flat subsets accum1.5x** | **OK** | 0.3275 | 0.3676 | 0.2872 | 0.3249 | 3953952 |
| ETTh2 | **Flat subsets accum2.0x LR-hi** | **OK** | 0.3029 | 0.3423 | 0.2656 | 0.2977 | 3954806 |
| ETTh2 | **Flat subsets accum2.0x LR-lo** | **OK** | 0.3272 | 0.3674 | 0.2841 | 0.3191 | 3954799 |
| ETTh2 | **Flat subsets accum2.0x** | **OK** | 0.3200 | 0.3621 | 0.2780 | 0.3098 | 3953959 |
| ETTh2 | **Flat subsets accum4x** | **OK** | 0.3104 | 0.3476 | 0.2680 | 0.3020 | 3963968 |
| ETTh2 | **Flat subsets guidance accum1.5x** | **OK** | 0.3016 | 0.3395 | 0.2678 | 0.3009 | 3961420 |
| ETTh2 | **Flat subsets guidance accum2x** | **OK** | 0.3011 | 0.3397 | 0.2685 | 0.3004 | 3961428 |
| ETTh2 | **Flat subsets guidance accum4x** | **OK** | 0.3076 | 0.3538 | 0.2801 | 0.3084 | 3961435 |
| ETTh2 | **Flat subsets guidance accum8x** | **OK** | 0.3148 | 0.3619 | 0.2941 | 0.3228 | 3961442 |
| ETTh2 | **Flat subsets** | **OK** | 0.3199 | 0.3546 | 0.2705 | 0.3104 | 3951194 |
| ETTh2 | **MAE Discrete** | **OK** | 0.3183 | 0.3497 | 0.8584 | 0.5778 | 3949860 |
| ETTh2 | **MMPD (subset)** | **ref** | 0.3186 | 0.3614 | 0.2705 | — | 3951202 |
| ETTm1 | **MS tune** | **OK** | 0.4679 | 0.4246 | 0.3176 | 0.4398 | 3956630 |
| ETTm1 | `diff_ema_decay_099` | **OK** | 0.4556 | 0.4231 | 0.3290 | 0.4673 | 3943896 |
| ETTm1 | `diff_min_snr_gamma_5` | **OK** | 0.5463 | 0.4584 | 0.4031 | 0.6940 | 3943898 |
| ETTm1 | `diff_noise_cosine` | **OK** | 0.4894 | 0.4368 | 0.3287 | 0.4614 | 3943900 |
| ETTm1 | `diff_prediction_x0` | **OK** | 0.5051 | 0.4334 | 0.3355 | 0.5193 | 3943902 |
| ETTm1 | `h16_16_16` | **OK** | 0.4724 | 0.4307 | 0.3365 | 0.4706 | 3943892 |
| ETTm1 | `h8_8_8` | **OK** | 0.5236 | 0.4546 | 0.3306 | 0.4884 | 3943894 |
| ETTm1 | `hp_anchor_lambda_090` | **OK** | 0.4661 | 0.4317 | 0.3407 | 0.5055 | 3943904 |
| ETTm1 | `hp_anchor_lambda_095` | **OK** | 0.4522 | 0.4210 | 0.3200 | 0.4448 | 3943906 |
| ETTm1 | `hp_beta_end_03` | **OK** | 0.6789 | 0.5006 | 0.9025 | 3.5316 | 3943908 |
| ETTm1 | `hp_beta_end_04` | **OK** | 0.4989 | 0.4383 | 0.5073 | 1.9746 | 3943910 |
| ETTm1 | `hp_cfg_dropout_02` | **OK** | 0.4785 | 0.4341 | 0.3281 | 0.4541 | 3943912 |
| ETTm1 | `hp_ctxbias_005` | **OK** | 0.4681 | 0.4236 | 0.3278 | 0.4594 | 3943914 |
| ETTm1 | `hp_ctxbias_neg01` | **OK** | 0.4807 | 0.4288 | 0.3299 | 0.4685 | 3943916 |
| ETTm1 | `hp_dit_depth4` | **OK** | 0.5066 | 0.4498 | 0.3360 | 0.4658 | 3943918 |
| ETTm1 | `hp_dit_dropout_01` | **OK** | 0.5045 | 0.4394 | 0.3410 | 0.4988 | 3943920 |
| ETTm1 | `hp_dit_embed288_heads4` | **OK** | 0.5516 | 0.4596 | 0.3396 | 0.5020 | 3943922 |
| ETTm1 | `hp_lr_cosine_warmup2` | **OK** | 0.4866 | 0.4360 | 0.3451 | 0.5155 | 3956634 |
| ETTm1 | `hp_lr_cosine_warmup5` | **OK** | 0.4857 | 0.4361 | 0.3453 | 0.5157 | 3956638 |
| ETTm1 | `hp_num_steps_1200` | **OK** | 0.4754 | 0.4289 | 0.3353 | 0.4862 | 3943928 |
| ETTm1 | `hp_num_steps_800` | **OK** | 0.4670 | 0.4274 | 0.3382 | 0.4872 | 3943930 |
| ETTm1 | `sweep_baseline` | **OK** | 0.4683 | 0.4259 | 0.3268 | 0.4642 | 3943932 |
| electricity | **AR LB336/H96 accum1.5x** | **OK** | 0.1638 | 0.2048 | 0.1534 | 0.1573 | 3961452 |
| electricity | **Binary flat** | **OK** | 0.2598 | 0.2713 | 0.2080 | 0.2579 | 3949856 |
| electricity | **Discrete** | **OK** | 0.2972 | 0.2712 | 0.5994 | 0.7374 | 3948458 |
| electricity | **Flat subsets EMA0.90** | **OK** | 0.1693 | 0.2086 | 0.1561 | 0.1619 | 3953321 |
| electricity | **Flat subsets EMA0.95** | **OK** | 0.1690 | 0.2088 | 0.1560 | 0.1615 | 3953328 |
| electricity | **Flat subsets EMA0.98** | **OK** | 0.1692 | 0.2091 | 0.1562 | 0.1619 | 3953335 |
| electricity | **Flat subsets EMA0.99** | **OK** | 0.1713 | 0.2111 | 0.1572 | 0.1639 | 3951531 |
| electricity | **Flat subsets EMA0.995** | **OK** | 0.1706 | 0.2101 | 0.1566 | 0.1622 | 3953342 |
| electricity | **Flat subsets EMA0.999** | **OK** | 0.1974 | 0.2361 | 0.1619 | 0.1693 | 3953349 |
| electricity | **Flat subsets accum1.25x** | **OK** | 0.1737 | 0.2140 | 0.1600 | 0.1679 | 3953948 |
| electricity | **Flat subsets accum1.5x LR-hi** | **OK** | 0.1643 | 0.1995 | 0.1504 | 0.1571 | 3954795 |
| electricity | **Flat subsets accum1.5x LR-lo** | **OK** | 0.1824 | 0.2215 | 0.1607 | 0.1692 | 3954788 |
| electricity | **Flat subsets accum1.5x** | **OK** | 0.1820 | 0.2213 | 0.1629 | 0.1731 | 3953955 |
| electricity | **Flat subsets accum2.0x LR-hi** | **OK** | 0.1657 | 0.2003 | 0.1503 | 0.1573 | 3954809 |
| electricity | **Flat subsets accum2.0x LR-lo** | **OK** | 0.1962 | 0.2347 | 0.1633 | 0.1723 | 3954802 |
| electricity | **Flat subsets accum2.0x** | **OK** | 0.1843 | 0.2245 | 0.1617 | 0.1699 | 3953962 |
| electricity | **Flat subsets accum4x** | **OK** | 0.1651 | 0.2028 | 0.1530 | 0.1594 | 3963971 |
| electricity | **Flat subsets guidance accum1.5x** | **OK** | 0.1678 | 0.2023 | 0.1537 | 0.1644 | 3961423 |
| electricity | **Flat subsets guidance accum2x** | **OK** | 0.1671 | 0.2038 | 0.1537 | 0.1624 | 3961431 |
| electricity | **Flat subsets guidance accum4x** | **OK** | 0.1742 | 0.2123 | 0.1595 | 0.1689 | 3961438 |
| electricity | **Flat subsets guidance accum8x** | **OK** | 0.2156 | 0.2554 | 0.1873 | 0.2036 | 3961445 |
| electricity | **Flat subsets** | **OK** | 0.1735 | 0.2132 | 0.1602 | 0.1699 | 3951197 |
| electricity | **MAE Discrete** | **OK** | 0.1714 | 0.2001 | 0.5281 | 0.5642 | 3949863 |
| electricity | **MMPD (subset)** | **ref** | 0.1617 | 0.2088 | 0.1610 | — | 3951205 |
| exchange_rate | **AR LB336/H96 accum1.5x** | **OK** | 0.1035 | 0.2224 | 0.1867 | 0.1013 | 3961450 |
| exchange_rate | **AR LB96/H720 accum1.5x** | **OK** | 0.8445 | 0.6937 | 0.6506 | 0.9065 | 3961457 |
| exchange_rate | **Binary flat** | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3949854 |
| exchange_rate | **Discrete** | **OK** | 0.0924 | 0.2132 | 0.3384 | 0.1959 | 3948456 |
| exchange_rate | **Flat subsets EMA0.90** | **OK** | 0.0868 | 0.2071 | 0.1699 | 0.0897 | 3953319 |
| exchange_rate | **Flat subsets EMA0.95** | **OK** | 0.0875 | 0.2077 | 0.1689 | 0.0888 | 3953326 |
| exchange_rate | **Flat subsets EMA0.98** | **OK** | 0.0878 | 0.2078 | 0.1684 | 0.0884 | 3953333 |
| exchange_rate | **Flat subsets EMA0.99 LB336/H96** | **OK** | 0.0922 | 0.2141 | 0.1692 | 0.0896 | 3955092 |
| exchange_rate | **Flat subsets EMA0.99 LB96/H720** | **OK** | 0.9771 | 0.7582 | 0.7219 | 1.0750 | 3955096 |
| exchange_rate | **Flat subsets EMA0.99** | **OK** | 0.0893 | 0.2086 | 0.1684 | 0.0881 | 3951529 |
| exchange_rate | **Flat subsets EMA0.995** | **OK** | 0.0875 | 0.2074 | 0.1667 | 0.0869 | 3953340 |
| exchange_rate | **Flat subsets EMA0.999** | **OK** | 0.0909 | 0.2107 | 0.1703 | 0.0895 | 3953347 |
| exchange_rate | **Flat subsets accum1.25x** | **OK** | 0.0895 | 0.2090 | 0.1698 | 0.0906 | 3953946 |
| exchange_rate | **Flat subsets accum1.5x LR-hi** | **OK** | 0.0901 | 0.2096 | 0.1686 | 0.0880 | 3954793 |
| exchange_rate | **Flat subsets accum1.5x LR-lo** | **OK** | 0.0913 | 0.2109 | 0.1708 | 0.0894 | 3954786 |
| exchange_rate | **Flat subsets accum1.5x** | **OK** | 0.0882 | 0.2071 | 0.1657 | 0.0866 | 3953953 |
| exchange_rate | **Flat subsets accum2.0x LR-hi** | **OK** | 0.0900 | 0.2106 | 0.1690 | 0.0884 | 3954807 |
| exchange_rate | **Flat subsets accum2.0x LR-lo** | **OK** | 0.0889 | 0.2084 | 0.1655 | 0.0856 | 3954800 |
| exchange_rate | **Flat subsets accum2.0x** | **OK** | 0.0875 | 0.2073 | 0.1687 | 0.0888 | 3953960 |
| exchange_rate | **Flat subsets accum4x** | **OK** | 0.0889 | 0.2080 | 0.1673 | 0.0868 | 3963969 |
| exchange_rate | **Flat subsets guidance accum1.5x** | **OK** | 0.0904 | 0.2118 | 0.1716 | 0.0902 | 3961421 |
| exchange_rate | **Flat subsets guidance accum2x** | **OK** | 0.0910 | 0.2111 | 0.1722 | 0.0903 | 3961429 |
| exchange_rate | **Flat subsets guidance accum4x** | **OK** | 0.0873 | 0.2072 | 0.1690 | 0.0853 | 3961436 |
| exchange_rate | **Flat subsets guidance accum8x** | **OK** | 0.1054 | 0.2328 | 0.1704 | 0.0888 | 3961443 |
| exchange_rate | **Flat subsets** | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3951195 |
| exchange_rate | **MAE Discrete** | **OK** | 0.0845 | 0.2039 | 0.4216 | 0.2109 | 3949861 |
| exchange_rate | **MMPD (subset)** | **ref** | 0.0810 | 0.1987 | 0.1563 | — | 3951203 |
| exchange_rate | **MS tune** | **OK** | 0.0910 | 0.2114 | 0.1676 | 0.0871 | 3956631 |
| exchange_rate | `diff_ema_decay_099` | **OK** | 0.0893 | 0.2086 | 0.1684 | 0.0881 | 3943855 |
| exchange_rate | `diff_min_snr_gamma_5` | **OK** | 0.0907 | 0.2125 | 0.1712 | 0.0898 | 3943857 |
| exchange_rate | `diff_noise_cosine` | **OK** | 0.0905 | 0.2094 | 0.1788 | 0.0998 | 3943859 |
| exchange_rate | `diff_prediction_x0` | **OK** | 0.0924 | 0.2130 | 0.1700 | 0.0896 | 3943861 |
| exchange_rate | `h16_16_16` | **OK** | 0.0892 | 0.2090 | 0.1760 | 0.0980 | 3943851 |
| exchange_rate | `h8_8_8` | **OK** | 0.0901 | 0.2102 | 0.1614 | 0.0842 | 3943853 |
| exchange_rate | `hp_anchor_lambda_090` | **OK** | 0.0825 | 0.2014 | 0.1656 | 0.0860 | 3943863 |
| exchange_rate | `hp_anchor_lambda_095` | **OK** | 0.0881 | 0.2096 | 0.1701 | 0.0877 | 3943865 |
| exchange_rate | `hp_beta_end_03` | **OK** | 0.0911 | 0.2138 | 0.2791 | 0.2509 | 3943867 |
| exchange_rate | `hp_beta_end_04` | **OK** | 0.0829 | 0.2035 | 0.2460 | 0.2276 | 3943869 |
| exchange_rate | `hp_cfg_dropout_02` | **OK** | 0.0858 | 0.2053 | 0.1691 | 0.0871 | 3943871 |
| exchange_rate | `hp_ctxbias_005` | **OK** | 0.0896 | 0.2107 | 0.1684 | 0.0863 | 3943873 |
| exchange_rate | `hp_ctxbias_neg01` | **OK** | 0.0857 | 0.2042 | 0.1647 | 0.0837 | 3943875 |
| exchange_rate | `hp_dit_depth4` | **OK** | 0.0913 | 0.2108 | 0.1703 | 0.0891 | 3943877 |
| exchange_rate | `hp_dit_dropout_01` | **OK** | 0.0924 | 0.2134 | 0.1754 | 0.0967 | 3943879 |
| exchange_rate | `hp_dit_embed288_heads4` | **OK** | 0.0925 | 0.2151 | 0.1707 | 0.0910 | 3943881 |
| exchange_rate | `hp_lr_cosine_warmup2` | **OK** | 0.0899 | 0.2094 | 0.1677 | 0.0873 | 3956635 |
| exchange_rate | `hp_lr_cosine_warmup5` | **OK** | 0.0898 | 0.2090 | 0.1673 | 0.0870 | 3956639 |
| exchange_rate | `hp_num_steps_1200` | **OK** | 0.0893 | 0.2081 | 0.1686 | 0.0880 | 3943887 |
| exchange_rate | `hp_num_steps_800` | **OK** | 0.0898 | 0.2115 | 0.1734 | 0.0914 | 3943889 |
| exchange_rate | `sweep_baseline` | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3943891 |
| solar_Alabama | **AR LB336/H96 accum1.5x** | **OK** | 0.2041 | 0.2229 | 0.1796 | 0.2002 | 3961454 |
| solar_Alabama | **Binary flat** | **OK** | 0.2170 | 0.2426 | 0.1945 | 0.2172 | 3949858 |
| solar_Alabama | **Discrete** | **OK** | 0.2560 | 0.2472 | 1.5170 | 3.3009 | 3948460 |
| solar_Alabama | **Flat subsets EMA0.90** | **OK** | 0.2147 | 0.2387 | 0.1884 | 0.2087 | 3953323 |
| solar_Alabama | **Flat subsets EMA0.95** | **OK** | 0.2136 | 0.2379 | 0.1881 | 0.2082 | 3953330 |
| solar_Alabama | **Flat subsets EMA0.98** | **OK** | 0.2126 | 0.2371 | 0.1881 | 0.2082 | 3953337 |
| solar_Alabama | **Flat subsets EMA0.99** | **OK** | 0.2123 | 0.2366 | 0.1890 | 0.2083 | 3951533 |
| solar_Alabama | **Flat subsets EMA0.995** | **OK** | 0.2129 | 0.2393 | 0.1932 | 0.2149 | 3953344 |
| solar_Alabama | **Flat subsets EMA0.999** | **OK** | 0.2108 | 0.2369 | 0.1912 | 0.2116 | 3953351 |
| solar_Alabama | **Flat subsets accum1.25x** | **OK** | 0.2231 | 0.2439 | 0.1977 | 0.2224 | 3953950 |
| solar_Alabama | **Flat subsets accum1.5x LR-hi** | **OK** | 0.2078 | 0.2283 | 0.1861 | 0.2073 | 3954797 |
| solar_Alabama | **Flat subsets accum1.5x LR-lo** | **OK** | 0.2110 | 0.2379 | 0.1916 | 0.2116 | 3954790 |
| solar_Alabama | **Flat subsets accum1.5x** | **OK** | 0.2002 | 0.2313 | 0.1789 | 0.1902 | 3953957 |
| solar_Alabama | **Flat subsets accum2.0x LR-hi** | **OK** | 0.2111 | 0.2321 | 0.1868 | 0.2087 | 3954811 |
| solar_Alabama | **Flat subsets accum2.0x LR-lo** | **OK** | 0.2136 | 0.2398 | 0.1933 | 0.2147 | 3954804 |
| solar_Alabama | **Flat subsets accum2.0x** | **OK** | 0.2158 | 0.2394 | 0.1876 | 0.2045 | 3953964 |
| solar_Alabama | **Flat subsets accum4x** | **OK** | 0.2098 | 0.2325 | 0.1878 | 0.2085 | 3963973 |
| solar_Alabama | **Flat subsets guidance accum1.5x** | **OK** | 0.2144 | 0.2347 | 0.1890 | 0.2104 | 3961426 |
| solar_Alabama | **Flat subsets guidance accum2x** | **OK** | 0.2132 | 0.2367 | 0.1879 | 0.2059 | 3961433 |
| solar_Alabama | **Flat subsets guidance accum4x** | **OK** | 0.2169 | 0.2415 | 0.1924 | 0.2109 | 3961440 |
| solar_Alabama | **Flat subsets guidance accum8x** | **OK** | 0.2246 | 0.2498 | 0.2010 | 0.2228 | 3961447 |
| solar_Alabama | **Flat subsets** | **OK** | 0.2170 | 0.2426 | 0.1945 | 0.2172 | 3951199 |
| solar_Alabama | **MAE Discrete** | **OK** | 0.2540 | 0.3142 | 2.4934 | 5.9180 | 3949865 |
| solar_Alabama | **MMPD (subset)** | **ref** | 0.2360 | 0.2690 | 0.2013 | — | 3951207 |
| traffic | **AR LB336/H96 accum1.5x** | **OK** | 0.4423 | 0.2763 | 0.2092 | 0.4204 | 3961453 |
| traffic | **AR LB96/H720 accum1.5x** | **OK** | 1.3981 | 0.7447 | 0.6410 | 1.3629 | 3961460 |
| traffic | **Binary flat** | **OK** | 0.4184 | 0.2813 | 0.2121 | 0.4138 | 3949857 |
| traffic | **Discrete** | **OK** | 0.3863 | 0.2382 | 1.2598 | 2.1313 | 3948459 |
| traffic | **Flat subsets EMA0.90** | **OK** | 0.5208 | 0.3168 | 0.2398 | 0.5119 | 3953322 |
| traffic | **Flat subsets EMA0.95** | **OK** | 0.5167 | 0.3161 | 0.2376 | 0.5030 | 3953329 |
| traffic | **Flat subsets EMA0.98** | **OK** | 0.5152 | 0.3161 | 0.2367 | 0.4987 | 3953336 |
| traffic | **Flat subsets EMA0.99 LB336/H96** | **OK** | 0.4791 | 0.2912 | 0.2214 | 0.4837 | 3955094 |
| traffic | **Flat subsets EMA0.99 LB96/H720** | **OK** | 1.4590 | 0.7630 | 0.5796 | 1.4363 | 3955098 |
| traffic | **Flat subsets EMA0.99** | **OK** | 0.5296 | 0.3245 | 0.2436 | 0.5202 | 3951532 |
| traffic | **Flat subsets EMA0.995** | **OK** | 0.5219 | 0.3196 | 0.2397 | 0.5076 | 3953343 |
| traffic | **Flat subsets EMA0.999** | **OK** | 0.8891 | 0.4865 | 0.3351 | 0.8390 | 3953350 |
| traffic | **Flat subsets accum1.25x** | **OK** | 0.5318 | 0.3318 | 0.2481 | 0.5312 | 3953949 |
| traffic | **Flat subsets accum1.5x LR-hi** | **OK** | 0.4533 | 0.2753 | 0.2086 | 0.4236 | 3954796 |
| traffic | **Flat subsets accum1.5x LR-lo** | **OK** | 0.5790 | 0.3489 | 0.2585 | 0.5675 | 3954789 |
| traffic | **Flat subsets accum1.5x** | **OK** | 0.5456 | 0.3373 | 0.2556 | 0.5597 | 3953956 |
| traffic | **Flat subsets accum2.0x LR-hi** | **OK** | 0.4573 | 0.2800 | 0.2111 | 0.4306 | 3954810 |
| traffic | **Flat subsets accum2.0x LR-lo** | **OK** | 0.6120 | 0.3635 | 0.2655 | 0.5902 | 3954803 |
| traffic | **Flat subsets accum2.0x** | **OK** | 0.5998 | 0.3526 | 0.2710 | 0.6114 | 3953963 |
| traffic | **Flat subsets accum4x** | **OK** | 0.5011 | 0.3054 | 0.2276 | 0.4742 | 3963972 |
| traffic | **Flat subsets guidance accum1.5x** | **OK** | 0.4845 | 0.2948 | 0.2244 | 0.4628 | 3961424 |
| traffic | **Flat subsets guidance accum2x** | **OK** | 0.5007 | 0.3052 | 0.2312 | 0.4777 | 3961432 |
| traffic | **Flat subsets guidance accum4x** | **OK** | 0.5417 | 0.3336 | 0.2605 | 0.5587 | 3961439 |
| traffic | **Flat subsets guidance accum8x** | **OK** | 0.7107 | 0.4152 | 0.3299 | 0.7121 | 3961446 |
| traffic | **Flat subsets** | **OK** | 0.5263 | 0.3252 | 0.2430 | 0.5130 | 3951198 |
| traffic | **MAE Discrete** | **OK** | 0.4356 | 0.2597 | 1.0900 | 2.3495 | 3949864 |
| traffic | **MMPD (subset)** | **ref** | 0.5225 | 0.3612 | 0.2515 | — | 3951206 |
| weather | **AR LB336/H96 accum1.5x** | **OK** | 0.1113 | 0.2312 | 0.1932 | 0.1086 | 3961451 |
| weather | **Binary flat** | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3949855 |
| weather | **Discrete** | **OK** | 0.1079 | 0.2280 | 0.3707 | 0.2615 | 3948457 |
| weather | **Flat subsets EMA0.90** | **OK** | 0.0968 | 0.2222 | 0.1803 | 0.1003 | 3953320 |
| weather | **Flat subsets EMA0.95** | **OK** | 0.0964 | 0.2217 | 0.1792 | 0.0994 | 3953327 |
| weather | **Flat subsets EMA0.98** | **OK** | 0.0966 | 0.2216 | 0.1760 | 0.0976 | 3953334 |
| weather | **Flat subsets EMA0.99 LB336/H96** | **OK** | 0.1053 | 0.2302 | 0.1807 | 0.1025 | 3955093 |
| weather | **Flat subsets EMA0.99 LB96/H720** | **OK** | 0.5848 | 0.5399 | 0.4771 | 0.5845 | 3955097 |
| weather | **Flat subsets EMA0.99** | **OK** | 0.0971 | 0.2220 | 0.1758 | 0.0970 | 3951530 |
| weather | **Flat subsets EMA0.995** | **OK** | 0.0964 | 0.2213 | 0.1752 | 0.0964 | 3953341 |
| weather | **Flat subsets EMA0.999** | **OK** | 0.0945 | 0.2193 | 0.1759 | 0.0972 | 3953348 |
| weather | **Flat subsets accum1.25x** | **OK** | 0.0975 | 0.2229 | 0.1777 | 0.0986 | 3953947 |
| weather | **Flat subsets accum1.5x LR-hi** | **OK** | 0.0963 | 0.2207 | 0.1746 | 0.0955 | 3954794 |
| weather | **Flat subsets accum1.5x LR-lo** | **OK** | 0.0973 | 0.2233 | 0.1775 | 0.0981 | 3954787 |
| weather | **Flat subsets accum1.5x** | **OK** | 0.0994 | 0.2251 | 0.1759 | 0.0961 | 3953954 |
| weather | **Flat subsets accum2.0x LR-hi** | **OK** | 0.0978 | 0.2226 | 0.1760 | 0.0969 | 3954808 |
| weather | **Flat subsets accum2.0x LR-lo** | **OK** | 0.0976 | 0.2234 | 0.1765 | 0.0973 | 3954801 |
| weather | **Flat subsets accum2.0x** | **OK** | 0.0979 | 0.2234 | 0.1858 | 0.1038 | 3953961 |
| weather | **Flat subsets accum4x** | **OK** | 0.0974 | 0.2225 | 0.1748 | 0.0958 | 3963970 |
| weather | **Flat subsets guidance accum1.5x** | **OK** | 0.0945 | 0.2176 | 0.1740 | 0.0952 | 3961422 |
| weather | **Flat subsets guidance accum2x** | **OK** | 0.0938 | 0.2172 | 0.1730 | 0.0944 | 3961430 |
| weather | **Flat subsets guidance accum4x** | **OK** | 0.0959 | 0.2195 | 0.1756 | 0.0963 | 3961437 |
| weather | **Flat subsets guidance accum8x** | **OK** | 0.1007 | 0.2260 | 0.1879 | 0.1054 | 3961444 |
| weather | **Flat subsets** | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3951196 |
| weather | **MAE Discrete** | **OK** | 0.0980 | 0.2195 | 0.4609 | 0.2603 | 3949862 |
| weather | **MMPD (subset)** | **ref** | 0.1128 | 0.2323 | 0.1911 | — | 3951204 |
| weather | **MS tune** | pre-fix invalid | 0.0987 | 0.2185 | 0.1775 | 0.0978 | 3943937 |
| weather | `diff_ema_decay_099` | **OK** | 0.0971 | 0.2220 | 0.1758 | 0.0970 | 3943897 |
| weather | `diff_min_snr_gamma_5` | **OK** | 0.0992 | 0.2243 | 0.1804 | 0.1014 | 3943899 |
| weather | `diff_noise_cosine` | **OK** | 0.0962 | 0.2192 | 0.1794 | 0.0965 | 3943901 |
| weather | `diff_prediction_x0` | **OK** | 0.1036 | 0.2276 | 0.1791 | 0.1022 | 3943903 |
| weather | `h16_16_16` | **OK** | 0.0981 | 0.2216 | 0.1767 | 0.0986 | 3943893 |
| weather | `h8_8_8` | **OK** | 0.1043 | 0.2282 | 0.1785 | 0.1019 | 3943895 |
| weather | `hp_anchor_lambda_090` | **OK** | 0.1001 | 0.2235 | 0.1757 | 0.1011 | 3943905 |
| weather | `hp_anchor_lambda_095` | **OK** | 0.0993 | 0.2231 | 0.1779 | 0.1012 | 3943907 |
| weather | `hp_beta_end_03` | **OK** | 0.1023 | 0.2277 | 0.2997 | 0.2489 | 3943909 |
| weather | `hp_beta_end_04` | **OK** | 0.0969 | 0.2203 | 0.2248 | 0.1866 | 3943911 |
| weather | `hp_cfg_dropout_02` | **OK** | 0.0981 | 0.2215 | 0.1795 | 0.1017 | 3943913 |
| weather | `hp_ctxbias_005` | **OK** | 0.0992 | 0.2212 | 0.1746 | 0.0985 | 3943915 |
| weather | `hp_ctxbias_neg01` | **OK** | 0.0996 | 0.2243 | 0.1778 | 0.1011 | 3943917 |
| weather | `hp_dit_depth4` | **OK** | 0.0989 | 0.2227 | 0.1801 | 0.1017 | 3943919 |
| weather | `hp_dit_dropout_01` | **OK** | 0.1031 | 0.2268 | 0.1808 | 0.1047 | 3943921 |
| weather | `hp_dit_embed288_heads4` | **OK** | 0.0963 | 0.2201 | 0.1734 | 0.0969 | 3943923 |
| weather | `hp_lr_cosine_warmup2` | **OK** | 0.1000 | 0.2246 | 0.1784 | 0.1004 | 3956636 |
| weather | `hp_lr_cosine_warmup5` | **OK** | 0.0998 | 0.2244 | 0.1785 | 0.1004 | 3956640 |
| weather | `hp_num_steps_1200` | **OK** | 0.1037 | 0.2270 | 0.1832 | 0.1047 | 3943929 |
| weather | `hp_num_steps_800` | **OK** | 0.1009 | 0.2247 | 0.1818 | 0.1017 | 3943931 |
| weather | `sweep_baseline` | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3943933 |
