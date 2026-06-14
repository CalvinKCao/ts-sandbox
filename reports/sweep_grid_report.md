# YAML-First Sweep Grid Report

Fixed-HP binary sweep (`configs/sweep/`, Jun 12 2026) plus ordinal D3PM staged runs: **Discrete** (CE, `ordinal_d3pm_staged`), **MAE Discrete** (expectation MAE + uniform `1/H` anchor, `ordinal_d3pm_mae_staged_subsets`), **Binary flat** (full variates, `binary_anchor_stationary_flat`), **Flat subsets** / **Flat subsets EMA0.99** (ETTh1-capped subsets, jobs `3951193`–`3951199` / `3951527`–`3951533`), and **MMPD (subset)** (`06-13-binary-mmpd-subset-compare`, jobs `3951201`–`3951207`). Probabilistic metrics: `dpmpp` sampler, 20 steps, 20 samples.

| Dataset | Config | Status | anchor_mse | anchor_mae | crps | sample_mean_mse | Job |
|---|---|---|---|---|---|---|---|
| ETTh1 | **Discrete** | **OK** | 0.5468 | 0.4577 | 0.7199 | 1.0323 | 3948454 |
| ETTh1 | **Flat subsets EMA0.99** | **OK** | 0.3974 | 0.4040 | 0.3021 | 0.4047 | 3951527 |
| ETTh1 | **Flat subsets** | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3951193 |
| ETTh1 | **MMPD (subset)** | ref | 0.3762 | 0.3936 | 0.2985 | — | 3951201 |
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
| ETTh1 | `hp_lr_cosine_warmup2` | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3943882 |
| ETTh1 | `hp_lr_cosine_warmup5` | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3943884 |
| ETTh1 | `hp_max_scale_tuning` | **OK** | 0.4117 | 0.4158 | 0.3114 | 0.4058 | 3943934 |
| ETTh1 | `hp_num_steps_1200` | **OK** | 0.4092 | 0.4121 | 0.3100 | 0.4244 | 3943886 |
| ETTh1 | `hp_num_steps_800` | **OK** | 0.4146 | 0.4151 | 0.3098 | 0.4260 | 3943888 |
| ETTh1 | `sweep_baseline` | **OK** | 0.4059 | 0.4085 | 0.3060 | 0.4185 | 3943890 |
| ETTh2 | **Discrete** | **OK** | 0.3397 | 0.3582 | 0.6547 | 0.6292 | 3948455 |
| ETTh2 | **Flat subsets EMA0.99** | **OK** | 0.3116 | 0.3500 | 0.2673 | 0.3007 | 3951528 |
| ETTh2 | **Flat subsets** | **OK** | 0.3199 | 0.3546 | 0.2705 | 0.3104 | 3951194 |
| ETTh2 | **MMPD (subset)** | ref | 0.3186 | 0.3614 | 0.2705 | — | 3951202 |
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
| ETTm1 | `hp_lr_cosine_warmup2` | **OK** | 0.4683 | 0.4259 | 0.3268 | 0.4642 | 3943924 |
| ETTm1 | `hp_lr_cosine_warmup5` | **OK** | 0.4683 | 0.4259 | 0.3268 | 0.4642 | 3943926 |
| ETTm1 | `hp_max_scale_tuning` | **OK** | 0.4784 | 0.4232 | 0.3374 | 0.4835 | 3943936 |
| ETTm1 | `hp_num_steps_1200` | **OK** | 0.4754 | 0.4289 | 0.3353 | 0.4862 | 3943928 |
| ETTm1 | `hp_num_steps_800` | **OK** | 0.4670 | 0.4274 | 0.3382 | 0.4872 | 3943930 |
| ETTm1 | `sweep_baseline` | **OK** | 0.4683 | 0.4259 | 0.3268 | 0.4642 | 3943932 |
| electricity | **Discrete** | **OK** | 0.2972 | 0.2712 | 0.5994 | 0.7374 | 3948458 |
| electricity | **Flat subsets EMA0.99** | **OK** | 0.1713 | 0.2111 | 0.1572 | 0.1639 | 3951531 |
| electricity | **Flat subsets** | **OK** | 0.1735 | 0.2132 | 0.1602 | 0.1699 | 3951197 |
| electricity | **MMPD (subset)** | ref | 0.1617 | 0.2088 | 0.1610 | — | 3951205 |
| exchange_rate | **Discrete** | **OK** | 0.0924 | 0.2132 | 0.3384 | 0.1959 | 3948456 |
| exchange_rate | **Flat subsets EMA0.99** | **OK** | 0.0893 | 0.2086 | 0.1684 | 0.0881 | 3951529 |
| exchange_rate | **Flat subsets** | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3951195 |
| exchange_rate | **MMPD (subset)** | ref | 0.0810 | 0.1987 | 0.1563 | — | 3951203 |
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
| exchange_rate | `hp_lr_cosine_warmup2` | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3943883 |
| exchange_rate | `hp_lr_cosine_warmup5` | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3943885 |
| exchange_rate | `hp_max_scale_tuning` | **OK** | 0.0871 | 0.2067 | 0.1685 | 0.0884 | 3943935 |
| exchange_rate | `hp_num_steps_1200` | **OK** | 0.0893 | 0.2081 | 0.1686 | 0.0880 | 3943887 |
| exchange_rate | `hp_num_steps_800` | **OK** | 0.0898 | 0.2115 | 0.1734 | 0.0914 | 3943889 |
| exchange_rate | `sweep_baseline` | **OK** | 0.0880 | 0.2078 | 0.1660 | 0.0843 | 3943891 |
| solar_Alabama | **Discrete** | **OK** | 0.2560 | 0.2472 | 1.5170 | 3.3009 | 3948460 |
| solar_Alabama | **Flat subsets EMA0.99** | **OK** | 0.2123 | 0.2366 | 0.1890 | 0.2083 | 3951533 |
| solar_Alabama | **Flat subsets** | **OK** | 0.2170 | 0.2426 | 0.1945 | 0.2172 | 3951199 |
| solar_Alabama | **MMPD (subset)** | ref | 0.2360 | 0.2690 | 0.2013 | — | 3951207 |
| traffic | **Discrete** | **OK** | 0.3863 | 0.2382 | 1.2598 | 2.1313 | 3948459 |
| traffic | **Flat subsets EMA0.99** | **OK** | 0.5296 | 0.3245 | 0.2436 | 0.5202 | 3951532 |
| traffic | **Flat subsets** | **OK** | 0.5263 | 0.3252 | 0.2430 | 0.5130 | 3951198 |
| traffic | **MMPD (subset)** | ref | 0.5225 | 0.3612 | 0.2515 | — | 3951206 |
| weather | **Discrete** | **OK** | 0.1079 | 0.2280 | 0.3707 | 0.2615 | 3948457 |
| weather | **Flat subsets EMA0.99** | **OK** | 0.0971 | 0.2220 | 0.1758 | 0.0970 | 3951530 |
| weather | **Flat subsets** | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3951196 |
| weather | **MMPD (subset)** | ref | 0.1128 | 0.2323 | 0.1911 | — | 3951204 |
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
| weather | `hp_lr_cosine_warmup2` | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3943925 |
| weather | `hp_lr_cosine_warmup5` | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3943927 |
| weather | `hp_max_scale_tuning` | **OK** | 0.0987 | 0.2185 | 0.1775 | 0.0978 | 3943937 |
| weather | `hp_num_steps_1200` | **OK** | 0.1037 | 0.2270 | 0.1832 | 0.1047 | 3943929 |
| weather | `hp_num_steps_800` | **OK** | 0.1009 | 0.2247 | 0.1818 | 0.1017 | 3943931 |
| weather | `sweep_baseline` | **OK** | 0.0978 | 0.2224 | 0.1776 | 0.1000 | 3943933 |
