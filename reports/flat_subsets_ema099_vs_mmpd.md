# Flat subsets EMA0.99 vs MMPD MaskAE

Head-to-head on the six **remaining** flat-subset datasets (Jun 16, 2026). Both use the same YAML `data_subset` policy (`binary_anchor_stationary_flat_subsets`).

- **EMA0.99**: `binary_anchor_stationary_flat_subsets_ema099`, `dpmpp` 20 steps / 20 samples (jobs `3967251`–`3967256`).
- **MMPD MaskAE**: `06-16-mmpd-maskae-grad-accum-150-lr-lo-subset`, UP2ME MaskAE backbone, grad_accum_150_lr_lo, Optuna tune (7 trials), 20 diffusion samples, full test (jobs `3968154`–`3968538`). MMPD indices come from subset-config (no binary ckpt).

Lower is better for MSE and CRPS.

| Dataset | subset_id | EMA0.99 mse | EMA0.99 crps | EMA job | MMPD mse | MMPD crps | MMPD job | Better MSE | Better CRPS |
|---|---|---|---|---|---|---|---|---|---|
| ETTm1 | ETTm1_4v_s3 | 0.4514 | 0.3273 | 3967251 | 0.4338 | 0.3195 | 3968154 | MMPD | MMPD |
| ETTm2 | ETTm2_7v_s4 | 0.1847 | 0.2027 | 3967252 | 0.2124 | 0.2220 | 3968155 | EMA0.99 | EMA0.99 |
| illness | illness | 4.3519 | 1.2278 | 3967253 | 4.3888 | 1.1415 | 3968156 | EMA0.99 | MMPD |
| PeMS | PeMS_7v_s1 | 0.3160 | 0.2859 | 3967254 | 0.4138 | 0.3341 | 3968537 | EMA0.99 | EMA0.99 |
| dalia | dalia_5v_s2 | 0.8841 | 0.3914 | 3967255 | 0.9006 | 0.3849 | 3968158 | EMA0.99 | MMPD |
| dynamic | dynamic_7v_s29 | 0.4210 | 0.1542 | 3967256 | 0.3664 | 0.1527 | 3968538 | MMPD | MMPD |

## Summary

- **MSE**: EMA0.99 4/6, MMPD 2/6
- **CRPS**: EMA0.99 2/6, MMPD 4/6
