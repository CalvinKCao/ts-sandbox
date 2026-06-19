# YAML-First Sweep Leaderboard

Probabilistic metrics from `dpmpp` sampler with `20` steps. Baseline is `sweep_baseline` (fixed 3e-5 LR, linear noise, epsilon target). **Discrete** is ordinal D3PM CE (`ordinal_d3pm_staged`). **MAE Discrete** is expectation-MAE + uniform `1/H` anchor (`ordinal_d3pm_mae_staged_subsets`). **Binary flat** is flat `0.5` XOR anchor on full variates (`binary_anchor_stationary_flat`). **Flat subsets** (`binary_anchor_stationary_flat_subsets`, jobs `3951193`–`3951199`). **Flat subsets EMA0.99** (`3951527`–`3951533`; Jun 16 remaining datasets `3967251`–`3967256`). EMA reuse sweep: `diffusion_ema_decay` ∈ {0.90, 0.95, 0.98, 0.995, 0.999} (jobs `3953317`–`3953351`). Grad-accum reuse sweep: effective batch {1.25×, 1.5×, 2.0×} (jobs `3953944`–`3953964`); LR-band split on 1.5×/2.0× (`3954784`–`3954810`). **Flat subsets guidance accum** {1.5×, 2×, 4×, 8×} (jobs `3961419`–`3961447`). **2d-guidance** (iTrans 2D ghost + guidance channel, `grad_accum_150_lr_hi_guidance`, jobs `3965290`–`3965296`). **Flat subsets accum4x** no guidance (jobs `3963967`–`3963973`). EMA0.99 lookback variants: LB336/H96 and LB96/H720 (`3955091`–`3955098`). **AR accum4x/8x** (`binary_anchor_ar_grad_accum_{400,800}`, LB96/H96). **AR LB336/H96 accum1.5x** (`3961448`–`3961454`); **AR LB96/H720 accum1.5x** (partial: `3961455`, `3961457`, `3961460`). **MS tune** (`hp_max_scale_tuning`; post-fix `3956631` exchange_rate, `3960878` ETTm1; incomplete `3960877` ETTh1 / `3960879` weather; cosine+warmup post-fix `3956633`–`3956640`). **Flatline/trend ablations** off `grad_accum_150_lr_lo` (jobs `3978767`–`3978776`): no win-norm, LR-lo guidance, no cross-attn, LR-lo MS tune. **MMPD (subset)** from `06-13-binary-mmpd-subset-compare` (Decoder backbone, same subsets as flat runs, 20 samples, full test). **MMPD (MaskedAE)** from `06-15-mmpd-maskae-grad-accum-200-lr-lo-tune` (ETTh1-capped 7 datasets, grad_accum_200_lr_lo, jobs `3965321`–`3965327`) and `06-16-mmpd-maskae-grad-accum-150-lr-lo-subset` (remaining 6 datasets, grad_accum_150_lr_lo MaskAE, jobs `3968154`–`3968538`). Legacy **MMPD** from `06-12-sweep-subset-mmpd` where subset MMPD is unavailable.

**Pre-fix invalid / incomplete:** Jun 12 `hp_max_scale_tuning` jobs `3943934`–`3943937` (and resumes `3947879`–`3947881`) never searched `max_scale`. Post-fix cosine+warmup `3956633`–`3956640` replaces pre-fix `3943882`–`3943927`. Post-fix MS tune: `3956629`–`3956632` (3h wall); Jun 15 resume `3960877`–`3960879`. **OK:** exchange_rate `3956631`, ETTm1 `3960878`. **Incomplete** (tuned `max_scale≈13.43`, eval pending): ETTh1 `3960877`, weather `3960879`.

## Average Δrank vs baseline

Δrank = config rank − `sweep_baseline` rank per dataset (negative = better anchor MSE). Avg Δrank averages over datasets where the config ran.


| Rank | Config                                 | avg Δrank | ETTh1 Δrank | ETTh2 Δrank | ETTm1 Δrank | ETTm2 Δrank | illness Δrank | exchange_rate Δrank | weather Δrank | electricity Δrank | traffic Δrank | PeMS Δrank | solar_Alabama Δrank | dalia Δrank | dynamic Δrank | Status |
| ---- | -------------------------------------- | --------- | ----------- | ----------- | ----------- | ----------- | ------------- | ------------------- | ------------- | ----------------- | ------------- | ---------- | ------------------- | ----------- | ------------- | ------ |
| 1    | **Flat subsets EMA0.90**               | -5.67     | -2          | —           | —           | —           | —             | -6                  | -9            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 2    | **Flat subsets guidance accum4x**      | -5.00     | +7          | —           | —           | —           | —             | -5                  | -17           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 3    | **Flat subsets EMA0.99**               | -2.75     | -9          | —           | -7          | —           | —             | +11                 | -6            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 4    | **Flat subsets EMA0.95**               | -2.67     | +5          | —           | —           | —           | —             | -2                  | -11           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 5    | **2d-guidance**                        | -2.67     | +15         | —           | —           | —           | —             | -7                  | -16           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 6    | **Flat subsets accum1.5x LR-hi**       | -1.67     | -11         | —           | —           | —           | —             | +19                 | -13           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 7    | **Flat subsets accum4x**               | -1.00     | -6          | —           | —           | —           | —             | +7                  | -4            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 8    | **Flat subsets EMA0.98**               | -1.00     | +8          | —           | —           | —           | —             | -1                  | -10           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 9    | **Flat subsets guidance accum1.5x**    | -0.67     | -4          | —           | —           | —           | —             | +20                 | -18           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 10   | **Flat subsets guidance accum2x**      | +0.33     | -3          | —           | —           | —           | —             | +25                 | -21           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 11   | **Binary flat**                        | +1.00     | +1          | —           | —           | —           | —             | +1                  | +1            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 12   | **Flat subsets EMA0.995**              | +1.67     | +21         | —           | —           | —           | —             | -4                  | -12           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 13   | **Flat subsets**                       | +2.00     | +2          | —           | —           | —           | —             | +2                  | +2            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 14   | **Flat subsets accum2.0x LR-hi**       | +3.00     | -7          | —           | —           | —           | —             | +17                 | -1            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 15   | **Flat subsets accum2.0x**             | +3.33     | +10         | —           | —           | —           | —             | -3                  | +3            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 16   | **Flat subsets accum1.25x**            | +5.00     | +6          | —           | —           | —           | —             | +12                 | -3            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 17   | **MAE Discrete**                       | +6.33     | +26         | —           | —           | —           | —             | -11                 | +4            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 18   | **Flat accum1.5x LR-lo no cross-attn** | +7.00     | —           | —           | —           | —           | —             | +4                  | +10           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 19   | **Flat subsets EMA0.999**              | +7.00     | +18         | —           | —           | —           | —             | +23                 | -20           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 20   | **Flat subsets accum2.0x LR-lo**       | +9.67     | +25         | —           | —           | —           | —             | +6                  | -2            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 21   | **Flat subsets accum1.5x LR-lo**       | +10.25    | +16         | —           | +1          | —           | —             | +29                 | -5            | —                 | —             | —          | —                   | —           | —             | **OK** |
| 22   | **MS tune**                            | +12.00    | —           | —           | -2          | —           | —             | +26                 | —             | —                 | —             | —          | —                   | —           | —             | **OK** |
| 23   | **Flat subsets accum1.5x**             | +12.33    | +20         | —           | —           | —           | —             | +5                  | +12           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 24   | **Flat accum1.5x LR-lo guidance**      | +15.50    | +22         | —           | +9          | —           | —             | —                   | —             | —                 | —             | —          | —                   | —           | —             | **OK** |
| 25   | **Flat accum1.5x LR-lo no win-norm**   | +20.00    | +34         | —           | +6          | —           | —             | —                   | —             | —                 | —             | —          | —                   | —           | —             | **OK** |
| 26   | **Flat subsets EMA0.99 LB336/H96**     | +22.67    | +13         | —           | —           | —           | —             | +30                 | +25           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 27   | **Flat accum1.5x LR-lo MS tune**       | +23.50    | —           | —           | —           | —           | —             | +24                 | +23           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 28   | **AR LB336/H96 accum1.5x**             | +24.67    | +12         | —           | —           | —           | —             | +35                 | +27           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 29   | **Flat subsets guidance accum8x**      | +28.33    | +32         | —           | —           | —           | —             | +36                 | +17           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 30   | **Discrete**                           | +31.33    | +35         | —           | —           | —           | —             | +33                 | +26           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 31   | **Flat subsets EMA0.99 LB96/H720**     | +35.00    | +38         | —           | —           | —           | —             | +38                 | +29           | —                 | —             | —          | —                   | —           | —             | **OK** |
| 32   | **AR LB96/H720 accum1.5x**             | +37.00    | +37         | —           | —           | —           | —             | +37                 | —             | —                 | —             | —          | —                   | —           | —             | **OK** |
| —    | **MMPD (subset)**                      | —         | —           | —           | —           | —           | —             | —                   | —             | —                 | —             | —          | —                   | —           | —             | ref    |
| —    | **MMPD (MaskedAE)**                    | —         | —           | —           | —           | —           | —             | —                   | —             | —                 | —             | —          | —                   | —           | —             | ref    |
| —    | **MMPD**                               | —         | —           | —           | —           | —           | —             | —                   | —             | —                 | —             | —          | —                   | —           | —             | ref    |


### ETTh1

Baseline `sweep_baseline` rank: **14** / 32 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).


| Rank | Config                               | anchor_mse | anchor_mae | crps   | Δrank | Status         |
| ---- | ------------------------------------ | ---------- | ---------- | ------ | ----- | -------------- |
| 1    | **MMPD (subset)**                    | 0.3762     | 0.3936     | 0.2985 | —     | ref            |
| 2    | **MMPD (MaskedAE)**                  | 0.3831     | 0.4016     | 0.2983 | —     | ref            |
| 3    | **Flat subsets accum1.5x LR-hi**     | 0.3943     | 0.4005     | 0.2994 | -11   | **OK**         |
| 5    | **Flat subsets EMA0.99**             | 0.3974     | 0.4040     | 0.3021 | -9    | **OK**         |
| 7    | **Flat subsets accum2.0x LR-hi**     | 0.4004     | 0.4034     | 0.3000 | -7    | **OK**         |
| 8    | **Flat subsets accum4x**             | 0.4012     | 0.4053     | 0.3020 | -6    | **OK**         |
| 10   | **Flat subsets guidance accum1.5x**  | 0.4046     | 0.4107     | 0.3069 | -4    | **OK**         |
| 11   | **Flat subsets guidance accum2x**    | 0.4049     | 0.4141     | 0.3091 | -3    | **OK**         |
| 12   | **Flat subsets EMA0.90**             | 0.4052     | 0.4082     | 0.3057 | -2    | **OK**         |
| 15   | **Binary flat**                      | 0.4059     | 0.4085     | 0.3060 | +1    | **OK**         |
| 16   | **Flat subsets**                     | 0.4059     | 0.4085     | 0.3060 | +2    | **OK**         |
| 19   | **Flat subsets EMA0.95**             | 0.4061     | 0.4083     | 0.3061 | +5    | **OK**         |
| 20   | **Flat subsets accum1.25x**          | 0.4077     | 0.4116     | 0.3052 | +6    | **OK**         |
| 21   | **Flat subsets guidance accum4x**    | 0.4081     | 0.4169     | 0.3110 | +7    | **OK**         |
| 22   | **Flat subsets EMA0.98**             | 0.4084     | 0.4087     | 0.3066 | +8    | **OK**         |
| 24   | **Flat subsets accum2.0x**           | 0.4092     | 0.4105     | 0.3068 | +10   | **OK**         |
| 26   | **AR LB336/H96 accum1.5x**           | 0.4093     | 0.4158     | 0.3078 | +12   | **OK**         |
| 27   | **Flat subsets EMA0.99 LB336/H96**   | 0.4109     | 0.4088     | 0.2951 | +13   | **OK**         |
| 29   | **2d-guidance**                      | 0.4116     | 0.4161     | 0.3114 | +15   | **OK**         |
| 30   | **Flat subsets accum1.5x LR-lo**     | 0.4122     | 0.4112     | 0.3080 | +16   | **OK**         |
| 32   | **Flat subsets EMA0.999**            | 0.4140     | 0.4097     | 0.3087 | +18   | **OK**         |
| 34   | **Flat subsets accum1.5x**           | 0.4149     | 0.4118     | 0.3066 | +20   | **OK**         |
| 35   | **Flat subsets EMA0.995**            | 0.4158     | 0.4111     | 0.3090 | +21   | **OK**         |
| 36   | **Flat accum1.5x LR-lo guidance**    | 0.4160     | 0.4221     | 0.3149 | +22   | **OK**         |
| 39   | **Flat subsets accum2.0x LR-lo**     | 0.4186     | 0.4124     | 0.3104 | +25   | **OK**         |
| 40   | **MAE Discrete**                     | 0.4204     | 0.4116     | 0.7263 | +26   | **OK**         |
| 46   | **Flat subsets guidance accum8x**    | 0.4506     | 0.4376     | 0.3281 | +32   | **OK**         |
| 48   | **Flat accum1.5x LR-lo no win-norm** | 0.4978     | 0.4879     | 0.3536 | +34   | **OK**         |
| 49   | **Discrete**                         | 0.5468     | 0.4577     | 0.7199 | +35   | **OK**         |
| 51   | **AR LB96/H720 accum1.5x**           | 0.7242     | 0.5749     | 0.4608 | +37   | **OK**         |
| 52   | **Flat subsets EMA0.99 LB96/H720**   | 0.7247     | 0.5637     | 0.4868 | +38   | **OK**         |
| 53   | **MS tune**                          | —          | —          | —      | +39   | **incomplete** |


### ETTh2

Baseline `sweep_baseline` missing. Total configs: 27


| Rank | Config                              | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | ----------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Flat subsets guidance accum2x**   | 0.3011     | 0.3397     | 0.2685 | —     | **OK** |
| 2    | **Flat subsets guidance accum1.5x** | 0.3016     | 0.3395     | 0.2678 | —     | **OK** |
| 3    | **Flat subsets accum1.5x LR-hi**    | 0.3017     | 0.3394     | 0.2631 | —     | **OK** |
| 4    | **Flat subsets accum2.0x LR-hi**    | 0.3029     | 0.3423     | 0.2656 | —     | **OK** |
| 5    | **Flat subsets guidance accum4x**   | 0.3076     | 0.3538     | 0.2801 | —     | **OK** |
| 6    | **Flat subsets accum4x**            | 0.3104     | 0.3476     | 0.2680 | —     | **OK** |
| 7    | **Flat subsets EMA0.99**            | 0.3116     | 0.3500     | 0.2673 | —     | **OK** |
| 8    | **Flat subsets EMA0.95**            | 0.3121     | 0.3509     | 0.2686 | —     | **OK** |
| 9    | **Flat subsets EMA0.90**            | 0.3124     | 0.3513     | 0.2684 | —     | **OK** |
| 10   | **Flat subsets EMA0.98**            | 0.3125     | 0.3507     | 0.2690 | —     | **OK** |
| 11   | **Flat subsets EMA0.995**           | 0.3134     | 0.3523     | 0.2699 | —     | **OK** |
| 12   | **Flat subsets guidance accum8x**   | 0.3148     | 0.3619     | 0.2941 | —     | **OK** |
| 13   | **2d-guidance**                     | 0.3151     | 0.3488     | 0.2682 | —     | **OK** |
| 14   | **Flat subsets accum1.25x**         | 0.3169     | 0.3545     | 0.2754 | —     | **OK** |
| 15   | **Flat subsets accum1.5x LR-lo**    | 0.3183     | 0.3595     | 0.2739 | —     | **OK** |
| 16   | **MAE Discrete**                    | 0.3183     | 0.3497     | 0.8584 | —     | **OK** |
| 17   | **MMPD (subset)**                   | 0.3186     | 0.3614     | 0.2705 | —     | ref    |
| 18   | **Binary flat**                     | 0.3199     | 0.3546     | 0.2705 | —     | **OK** |
| 19   | **Flat subsets**                    | 0.3199     | 0.3546     | 0.2705 | —     | **OK** |
| 20   | **Flat subsets accum2.0x**          | 0.3200     | 0.3621     | 0.2780 | —     | **OK** |
| 21   | **Flat subsets EMA0.999**           | 0.3250     | 0.3647     | 0.2799 | —     | **OK** |
| 22   | **Flat subsets accum2.0x LR-lo**    | 0.3272     | 0.3674     | 0.2841 | —     | **OK** |
| 23   | **Flat subsets accum1.5x**          | 0.3275     | 0.3676     | 0.2872 | —     | **OK** |
| 24   | **AR LB336/H96 accum1.5x**          | 0.3373     | 0.3690     | 0.2855 | —     | **OK** |
| 25   | **MMPD (MaskedAE)**                 | 0.3383     | 0.3783     | 0.2916 | —     | ref    |
| 26   | **Discrete**                        | 0.3397     | 0.3582     | 0.6547 | —     | **OK** |
| 27   | **AR LB96/H720 accum1.5x**          | 0.4503     | 0.4575     | 0.3846 | —     | **OK** |


### ETTm1

Baseline `sweep_baseline` rank: **10** / 7 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).


| Rank | Config                               | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | ------------------------------------ | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **MMPD**                             | 0.4208     | 0.4122     | 0.3109 | —     | ref    |
| 2    | **MMPD (MaskedAE)**                  | 0.4338     | 0.4251     | 0.3195 | —     | ref    |
| 3    | **Flat subsets EMA0.99**             | 0.4514     | 0.4215     | 0.3273 | -7    | **OK** |
| 8    | **MS tune**                          | 0.4679     | 0.4246     | 0.3176 | -2    | **OK** |
| 11   | **Flat subsets accum1.5x LR-lo**     | 0.4691     | 0.4291     | 0.3371 | +1    | **OK** |
| 16   | **Flat accum1.5x LR-lo no win-norm** | 0.4828     | 0.4551     | 0.3481 | +6    | **OK** |
| 19   | **Flat accum1.5x LR-lo guidance**    | 0.4887     | 0.4376     | 0.3383 | +9    | **OK** |


### ETTm2

Baseline `sweep_baseline` missing. Total configs: 3


| Rank | Config                           | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | -------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Flat subsets EMA0.99**         | 0.1847     | 0.2593     | 0.2027 | —     | **OK** |
| 2    | **Flat subsets accum1.5x LR-lo** | 0.1864     | 0.2605     | 0.2032 | —     | **OK** |
| 3    | **MMPD (MaskedAE)**              | 0.2124     | 0.2963     | 0.2220 | —     | ref    |


### illness

Baseline `sweep_baseline` missing. Total configs: 3


| Rank | Config                           | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | -------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Flat subsets EMA0.99**         | 4.3519     | 1.5247     | 1.2278 | —     | **OK** |
| 2    | **Flat subsets accum1.5x LR-lo** | 4.3647     | 1.5311     | 1.2399 | —     | **OK** |
| 3    | **MMPD (MaskedAE)**              | 4.3888     | 1.5253     | 1.1415 | —     | ref    |


### exchange_rate

Baseline `sweep_baseline` rank: **15** / 32 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).


| Rank | Config                                 | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | -------------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **MMPD (subset)**                      | 0.0810     | 0.1987     | 0.1563 | —     | ref    |
| 4    | **MAE Discrete**                       | 0.0845     | 0.2039     | 0.4216 | -11   | **OK** |
| 5    | **MMPD (MaskedAE)**                    | 0.0849     | 0.2030     | 0.1616 | —     | ref    |
| 8    | **2d-guidance**                        | 0.0868     | 0.2059     | 0.1646 | -7    | **OK** |
| 9    | **Flat subsets EMA0.90**               | 0.0868     | 0.2071     | 0.1699 | -6    | **OK** |
| 10   | **Flat subsets guidance accum4x**      | 0.0873     | 0.2072     | 0.1690 | -5    | **OK** |
| 11   | **Flat subsets EMA0.995**              | 0.0875     | 0.2074     | 0.1667 | -4    | **OK** |
| 12   | **Flat subsets accum2.0x**             | 0.0875     | 0.2073     | 0.1687 | -3    | **OK** |
| 13   | **Flat subsets EMA0.95**               | 0.0875     | 0.2077     | 0.1689 | -2    | **OK** |
| 14   | **Flat subsets EMA0.98**               | 0.0878     | 0.2078     | 0.1684 | -1    | **OK** |
| 16   | **Binary flat**                        | 0.0880     | 0.2078     | 0.1660 | +1    | **OK** |
| 17   | **Flat subsets**                       | 0.0880     | 0.2078     | 0.1660 | +2    | **OK** |
| 19   | **Flat accum1.5x LR-lo no cross-attn** | 0.0882     | 0.2074     | 0.1663 | +4    | **OK** |
| 20   | **Flat subsets accum1.5x**             | 0.0882     | 0.2071     | 0.1657 | +5    | **OK** |
| 21   | **Flat subsets accum2.0x LR-lo**       | 0.0889     | 0.2084     | 0.1655 | +6    | **OK** |
| 22   | **Flat subsets accum4x**               | 0.0889     | 0.2080     | 0.1673 | +7    | **OK** |
| 26   | **Flat subsets EMA0.99**               | 0.0893     | 0.2086     | 0.1684 | +11   | **OK** |
| 27   | **Flat subsets accum1.25x**            | 0.0895     | 0.2090     | 0.1698 | +12   | **OK** |
| 32   | **Flat subsets accum2.0x LR-hi**       | 0.0900     | 0.2106     | 0.1690 | +17   | **OK** |
| 34   | **Flat subsets accum1.5x LR-hi**       | 0.0901     | 0.2096     | 0.1686 | +19   | **OK** |
| 35   | **Flat subsets guidance accum1.5x**    | 0.0904     | 0.2118     | 0.1716 | +20   | **OK** |
| 38   | **Flat subsets EMA0.999**              | 0.0909     | 0.2107     | 0.1703 | +23   | **OK** |
| 39   | **Flat accum1.5x LR-lo MS tune**       | 0.0909     | 0.2108     | 0.1711 | +24   | **OK** |
| 40   | **Flat subsets guidance accum2x**      | 0.0910     | 0.2111     | 0.1722 | +25   | **OK** |
| 41   | **MS tune**                            | 0.0910     | 0.2114     | 0.1676 | +26   | **OK** |
| 44   | **Flat subsets accum1.5x LR-lo**       | 0.0913     | 0.2109     | 0.1708 | +29   | **OK** |
| 45   | **Flat subsets EMA0.99 LB336/H96**     | 0.0922     | 0.2141     | 0.1692 | +30   | **OK** |
| 48   | **Discrete**                           | 0.0924     | 0.2132     | 0.3384 | +33   | **OK** |
| 50   | **AR LB336/H96 accum1.5x**             | 0.1035     | 0.2224     | 0.1867 | +35   | **OK** |
| 51   | **Flat subsets guidance accum8x**      | 0.1054     | 0.2328     | 0.1704 | +36   | **OK** |
| 52   | **AR LB96/H720 accum1.5x**             | 0.8445     | 0.6937     | 0.6506 | +37   | **OK** |
| 53   | **Flat subsets EMA0.99 LB96/H720**     | 0.9771     | 0.7582     | 0.7219 | +38   | **OK** |


### weather

Baseline `sweep_baseline` rank: **22** / 31 (lower anchor MSE is better). Δrank = config rank − baseline rank (negative = improvement).


| Rank | Config                                 | anchor_mse | anchor_mae | crps   | Δrank | Status         |
| ---- | -------------------------------------- | ---------- | ---------- | ------ | ----- | -------------- |
| 1    | **Flat subsets guidance accum2x**      | 0.0938     | 0.2172     | 0.1730 | -21   | **OK**         |
| 2    | **Flat subsets EMA0.999**              | 0.0945     | 0.2193     | 0.1759 | -20   | **OK**         |
| 3    | **MMPD (MaskedAE)**                    | 0.0945     | 0.2156     | 0.1693 | —     | ref            |
| 4    | **Flat subsets guidance accum1.5x**    | 0.0945     | 0.2176     | 0.1740 | -18   | **OK**         |
| 5    | **Flat subsets guidance accum4x**      | 0.0959     | 0.2195     | 0.1756 | -17   | **OK**         |
| 6    | **2d-guidance**                        | 0.0960     | 0.2196     | 0.1745 | -16   | **OK**         |
| 9    | **Flat subsets accum1.5x LR-hi**       | 0.0963     | 0.2207     | 0.1746 | -13   | **OK**         |
| 10   | **Flat subsets EMA0.995**              | 0.0964     | 0.2213     | 0.1752 | -12   | **OK**         |
| 11   | **Flat subsets EMA0.95**               | 0.0964     | 0.2217     | 0.1792 | -11   | **OK**         |
| 12   | **Flat subsets EMA0.98**               | 0.0966     | 0.2216     | 0.1760 | -10   | **OK**         |
| 13   | **Flat subsets EMA0.90**               | 0.0968     | 0.2222     | 0.1803 | -9    | **OK**         |
| 16   | **Flat subsets EMA0.99**               | 0.0971     | 0.2220     | 0.1758 | -6    | **OK**         |
| 17   | **Flat subsets accum1.5x LR-lo**       | 0.0973     | 0.2233     | 0.1775 | -5    | **OK**         |
| 18   | **Flat subsets accum4x**               | 0.0974     | 0.2225     | 0.1748 | -4    | **OK**         |
| 19   | **Flat subsets accum1.25x**            | 0.0975     | 0.2229     | 0.1777 | -3    | **OK**         |
| 20   | **Flat subsets accum2.0x LR-lo**       | 0.0976     | 0.2234     | 0.1765 | -2    | **OK**         |
| 21   | **Flat subsets accum2.0x LR-hi**       | 0.0978     | 0.2226     | 0.1760 | -1    | **OK**         |
| 23   | **Binary flat**                        | 0.0978     | 0.2224     | 0.1776 | +1    | **OK**         |
| 24   | **Flat subsets**                       | 0.0978     | 0.2224     | 0.1776 | +2    | **OK**         |
| 25   | **Flat subsets accum2.0x**             | 0.0979     | 0.2234     | 0.1858 | +3    | **OK**         |
| 26   | **MAE Discrete**                       | 0.0980     | 0.2195     | 0.4609 | +4    | **OK**         |
| 32   | **Flat accum1.5x LR-lo no cross-attn** | 0.0992     | 0.2246     | 0.1798 | +10   | **OK**         |
| 34   | **Flat subsets accum1.5x**             | 0.0994     | 0.2251     | 0.1759 | +12   | **OK**         |
| 39   | **Flat subsets guidance accum8x**      | 0.1007     | 0.2260     | 0.1879 | +17   | **OK**         |
| 45   | **Flat accum1.5x LR-lo MS tune**       | 0.1041     | 0.2306     | 0.1854 | +23   | **OK**         |
| 47   | **Flat subsets EMA0.99 LB336/H96**     | 0.1053     | 0.2302     | 0.1807 | +25   | **OK**         |
| 48   | **Discrete**                           | 0.1079     | 0.2280     | 0.3707 | +26   | **OK**         |
| 49   | **AR LB336/H96 accum1.5x**             | 0.1113     | 0.2312     | 0.1932 | +27   | **OK**         |
| 50   | **MMPD (subset)**                      | 0.1128     | 0.2323     | 0.1911 | —     | ref            |
| 51   | **Flat subsets EMA0.99 LB96/H720**     | 0.5848     | 0.5399     | 0.4771 | +29   | **OK**         |
| 52   | **MS tune**                            | —          | —          | —      | +30   | **incomplete** |


### electricity

Baseline `sweep_baseline` missing. Total configs: 28


| Rank | Config                               | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | ------------------------------------ | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **MMPD (subset)**                    | 0.1617     | 0.2088     | 0.1610 | —     | ref    |
| 2    | **AR LB336/H96 accum1.5x**           | 0.1638     | 0.2048     | 0.1534 | —     | **OK** |
| 3    | **Flat subsets accum1.5x LR-hi**     | 0.1643     | 0.1995     | 0.1504 | —     | **OK** |
| 4    | **MMPD (MaskedAE)**                  | 0.1645     | 0.2148     | 0.1624 | —     | ref    |
| 5    | **Flat subsets accum4x**             | 0.1651     | 0.2028     | 0.1530 | —     | **OK** |
| 6    | **Flat subsets accum2.0x LR-hi**     | 0.1657     | 0.2003     | 0.1503 | —     | **OK** |
| 7    | **Flat subsets guidance accum2x**    | 0.1671     | 0.2038     | 0.1537 | —     | **OK** |
| 8    | **2d-guidance**                      | 0.1675     | 0.2056     | 0.1547 | —     | **OK** |
| 9    | **Flat subsets guidance accum1.5x**  | 0.1678     | 0.2023     | 0.1537 | —     | **OK** |
| 10   | **Flat subsets EMA0.95**             | 0.1690     | 0.2088     | 0.1560 | —     | **OK** |
| 11   | **Flat subsets EMA0.98**             | 0.1692     | 0.2091     | 0.1562 | —     | **OK** |
| 12   | **Flat subsets EMA0.90**             | 0.1693     | 0.2086     | 0.1561 | —     | **OK** |
| 13   | **Flat subsets EMA0.995**            | 0.1706     | 0.2101     | 0.1566 | —     | **OK** |
| 14   | **Flat subsets EMA0.99**             | 0.1713     | 0.2111     | 0.1572 | —     | **OK** |
| 15   | **MAE Discrete**                     | 0.1714     | 0.2001     | 0.5281 | —     | **OK** |
| 16   | **Flat subsets**                     | 0.1735     | 0.2132     | 0.1602 | —     | **OK** |
| 17   | **Flat subsets accum1.25x**          | 0.1737     | 0.2140     | 0.1600 | —     | **OK** |
| 18   | **Flat subsets guidance accum4x**    | 0.1742     | 0.2123     | 0.1595 | —     | **OK** |
| 19   | **Flat subsets accum1.5x**           | 0.1820     | 0.2213     | 0.1629 | —     | **OK** |
| 20   | **Flat subsets accum1.5x LR-lo**     | 0.1824     | 0.2215     | 0.1607 | —     | **OK** |
| 21   | **Flat accum1.5x LR-lo guidance**    | 0.1837     | 0.2223     | 0.1639 | —     | **OK** |
| 22   | **Flat subsets accum2.0x**           | 0.1843     | 0.2245     | 0.1617 | —     | **OK** |
| 23   | **Flat subsets accum2.0x LR-lo**     | 0.1962     | 0.2347     | 0.1633 | —     | **OK** |
| 24   | **Flat subsets EMA0.999**            | 0.1974     | 0.2361     | 0.1619 | —     | **OK** |
| 25   | **Flat accum1.5x LR-lo no win-norm** | 0.2011     | 0.2477     | 0.1781 | —     | **OK** |
| 26   | **Flat subsets guidance accum8x**    | 0.2156     | 0.2554     | 0.1873 | —     | **OK** |
| 27   | **Binary flat**                      | 0.2598     | 0.2713     | 0.2080 | —     | **OK** |
| 28   | **Discrete**                         | 0.2972     | 0.2712     | 0.5994 | —     | **OK** |


### traffic

Baseline `sweep_baseline` missing. Total configs: 29


| Rank | Config                              | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | ----------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Discrete**                        | 0.3863     | 0.2382     | 1.2598 | —     | **OK** |
| 2    | **Binary flat**                     | 0.4184     | 0.2813     | 0.2121 | —     | **OK** |
| 3    | **MAE Discrete**                    | 0.4356     | 0.2597     | 1.0900 | —     | **OK** |
| 4    | **AR LB336/H96 accum1.5x**          | 0.4423     | 0.2763     | 0.2092 | —     | **OK** |
| 5    | **Flat subsets accum1.5x LR-hi**    | 0.4533     | 0.2753     | 0.2086 | —     | **OK** |
| 6    | **Flat subsets accum2.0x LR-hi**    | 0.4573     | 0.2800     | 0.2111 | —     | **OK** |
| 7    | **2d-guidance**                     | 0.4621     | 0.2814     | 0.2139 | —     | **OK** |
| 8    | **Flat subsets EMA0.99 LB336/H96**  | 0.4791     | 0.2912     | 0.2214 | —     | **OK** |
| 9    | **Flat subsets guidance accum1.5x** | 0.4845     | 0.2948     | 0.2244 | —     | **OK** |
| 10   | **Flat subsets guidance accum2x**   | 0.5007     | 0.3052     | 0.2312 | —     | **OK** |
| 11   | **Flat subsets accum4x**            | 0.5011     | 0.3054     | 0.2276 | —     | **OK** |
| 12   | **Flat subsets EMA0.98**            | 0.5152     | 0.3161     | 0.2367 | —     | **OK** |
| 13   | **MMPD (MaskedAE)**                 | 0.5165     | 0.3642     | 0.2377 | —     | ref    |
| 14   | **Flat subsets EMA0.95**            | 0.5167     | 0.3161     | 0.2376 | —     | **OK** |
| 15   | **Flat subsets EMA0.90**            | 0.5208     | 0.3168     | 0.2398 | —     | **OK** |
| 16   | **Flat subsets EMA0.995**           | 0.5219     | 0.3196     | 0.2397 | —     | **OK** |
| 17   | **MMPD (subset)**                   | 0.5225     | 0.3612     | 0.2515 | —     | ref    |
| 18   | **Flat subsets**                    | 0.5263     | 0.3252     | 0.2430 | —     | **OK** |
| 19   | **Flat subsets EMA0.99**            | 0.5296     | 0.3245     | 0.2436 | —     | **OK** |
| 20   | **Flat subsets accum1.25x**         | 0.5318     | 0.3318     | 0.2481 | —     | **OK** |
| 21   | **Flat subsets guidance accum4x**   | 0.5417     | 0.3336     | 0.2605 | —     | **OK** |
| 22   | **Flat subsets accum1.5x**          | 0.5456     | 0.3373     | 0.2556 | —     | **OK** |
| 23   | **Flat subsets accum1.5x LR-lo**    | 0.5790     | 0.3489     | 0.2585 | —     | **OK** |
| 24   | **Flat subsets accum2.0x**          | 0.5998     | 0.3526     | 0.2710 | —     | **OK** |
| 25   | **Flat subsets accum2.0x LR-lo**    | 0.6120     | 0.3635     | 0.2655 | —     | **OK** |
| 26   | **Flat subsets guidance accum8x**   | 0.7107     | 0.4152     | 0.3299 | —     | **OK** |
| 27   | **Flat subsets EMA0.999**           | 0.8891     | 0.4865     | 0.3351 | —     | **OK** |
| 28   | **AR LB96/H720 accum1.5x**          | 1.3981     | 0.7447     | 0.6410 | —     | **OK** |
| 29   | **Flat subsets EMA0.99 LB96/H720**  | 1.4590     | 0.7630     | 0.5796 | —     | **OK** |


### PeMS

Baseline `sweep_baseline` missing. Total configs: 3


| Rank | Config                           | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | -------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Flat subsets EMA0.99**         | 0.3160     | 0.3828     | 0.2859 | —     | **OK** |
| 2    | **Flat subsets accum1.5x LR-lo** | 0.3330     | 0.3913     | 0.2934 | —     | **OK** |
| 3    | **MMPD (MaskedAE)**              | 0.4138     | 0.4402     | 0.3341 | —     | ref    |


### solar_Alabama

Baseline `sweep_baseline` missing. Total configs: 26


| Rank | Config                              | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | ----------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Flat subsets accum1.5x**          | 0.2002     | 0.2313     | 0.1789 | —     | **OK** |
| 2    | **AR LB336/H96 accum1.5x**          | 0.2041     | 0.2229     | 0.1796 | —     | **OK** |
| 3    | **Flat subsets accum1.5x LR-hi**    | 0.2078     | 0.2283     | 0.1861 | —     | **OK** |
| 4    | **Flat subsets accum4x**            | 0.2098     | 0.2325     | 0.1878 | —     | **OK** |
| 5    | **Flat subsets EMA0.999**           | 0.2108     | 0.2369     | 0.1912 | —     | **OK** |
| 6    | **Flat subsets accum1.5x LR-lo**    | 0.2110     | 0.2379     | 0.1916 | —     | **OK** |
| 7    | **Flat subsets accum2.0x LR-hi**    | 0.2111     | 0.2321     | 0.1868 | —     | **OK** |
| 8    | **2d-guidance**                     | 0.2121     | 0.2313     | 0.1846 | —     | **OK** |
| 9    | **Flat subsets EMA0.99**            | 0.2123     | 0.2366     | 0.1890 | —     | **OK** |
| 10   | **Flat subsets EMA0.98**            | 0.2126     | 0.2371     | 0.1881 | —     | **OK** |
| 11   | **Flat subsets EMA0.995**           | 0.2129     | 0.2393     | 0.1932 | —     | **OK** |
| 12   | **Flat subsets guidance accum2x**   | 0.2132     | 0.2367     | 0.1879 | —     | **OK** |
| 13   | **Flat subsets EMA0.95**            | 0.2136     | 0.2379     | 0.1881 | —     | **OK** |
| 14   | **Flat subsets accum2.0x LR-lo**    | 0.2136     | 0.2398     | 0.1933 | —     | **OK** |
| 15   | **MMPD (MaskedAE)**                 | 0.2144     | 0.2439     | 0.1783 | —     | ref    |
| 16   | **Flat subsets guidance accum1.5x** | 0.2144     | 0.2347     | 0.1890 | —     | **OK** |
| 17   | **Flat subsets EMA0.90**            | 0.2147     | 0.2387     | 0.1884 | —     | **OK** |
| 18   | **Flat subsets accum2.0x**          | 0.2158     | 0.2394     | 0.1876 | —     | **OK** |
| 19   | **Flat subsets guidance accum4x**   | 0.2169     | 0.2415     | 0.1924 | —     | **OK** |
| 20   | **Binary flat**                     | 0.2170     | 0.2426     | 0.1945 | —     | **OK** |
| 21   | **Flat subsets**                    | 0.2170     | 0.2426     | 0.1945 | —     | **OK** |
| 22   | **Flat subsets accum1.25x**         | 0.2231     | 0.2439     | 0.1977 | —     | **OK** |
| 23   | **Flat subsets guidance accum8x**   | 0.2246     | 0.2498     | 0.2010 | —     | **OK** |
| 24   | **MMPD (subset)**                   | 0.2360     | 0.2690     | 0.2013 | —     | ref    |
| 25   | **MAE Discrete**                    | 0.2540     | 0.3142     | 2.4934 | —     | **OK** |
| 26   | **Discrete**                        | 0.2560     | 0.2472     | 1.5170 | —     | **OK** |


### dalia

Baseline `sweep_baseline` missing. Total configs: 3


| Rank | Config                           | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | -------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **Flat subsets EMA0.99**         | 0.8841     | 0.4958     | 0.3914 | —     | **OK** |
| 2    | **MMPD (MaskedAE)**              | 0.9006     | 0.5224     | 0.3849 | —     | ref    |
| 3    | **Flat subsets accum1.5x LR-lo** | 0.9264     | 0.5123     | 0.3999 | —     | **OK** |


### dynamic

Baseline `sweep_baseline` missing. Total configs: 3


| Rank | Config                           | anchor_mse | anchor_mae | crps   | Δrank | Status |
| ---- | -------------------------------- | ---------- | ---------- | ------ | ----- | ------ |
| 1    | **MMPD (MaskedAE)**              | 0.3664     | 0.2458     | 0.1527 | —     | ref    |
| 2    | **Flat subsets accum1.5x LR-lo** | 0.4206     | 0.1876     | 0.1547 | —     | **OK** |
| 3    | **Flat subsets EMA0.99**         | 0.4210     | 0.1830     | 0.1542 | —     | **OK** |


