# Binary flat vs MMPD — ETTh1-capped subset comparison

Apples-to-apples runs on matched variate subsets (`binary_anchor_stationary_flat_subsets`). Binary eval: `dpmpp`, 20 steps, 20 samples. MMPD: same subset indices from binary ckpt metadata, 20 samples, full test (`06-13-binary-mmpd-subset-compare`). Merge job `3951208` failed but all dataset partials are present.

| Dataset | subset_id | Flat subsets anchor_mse | Flat subsets crps | Flat subsets job | Flat subsets EMA0.99 anchor_mse | Flat subsets EMA0.99 crps | EMA job | MMPD (subset) anchor_mse | MMPD (subset) crps | MMPD job | Best CRPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | ETTh1 | 0.4059 | 0.3060 | 3951193 | 0.3974 | 0.3021 | 3951527 | 0.3762 | 0.2985 | 3951201 | **MMPD (subset)** |
| ETTh2 | ETTh2 | 0.3199 | 0.2705 | 3951194 | 0.3116 | 0.2673 | 3951528 | 0.3186 | 0.2705 | 3951202 | **Flat subsets EMA0.99** |
| exchange_rate | exchange_rate | 0.0880 | 0.1660 | 3951195 | 0.0893 | 0.1684 | 3951529 | 0.0810 | 0.1563 | 3951203 | **MMPD (subset)** |
| weather | weather_4v_s2 | 0.0978 | 0.1776 | 3951196 | 0.0971 | 0.1758 | 3951530 | 0.1128 | 0.1911 | 3951204 | **Flat subsets EMA0.99** |
| electricity | electricity_4v_s1 | 0.1735 | 0.1602 | 3951197 | 0.1713 | 0.1572 | 3951531 | 0.1617 | 0.1610 | 3951205 | **Flat subsets EMA0.99** |
| traffic | traffic_4v_s1 | 0.5263 | 0.2430 | 3951198 | 0.5296 | 0.2436 | 3951532 | 0.5225 | 0.2515 | 3951206 | **Flat subsets** |
| solar_Alabama | solar_Alabama_2v_s1 | 0.2170 | 0.1945 | 3951199 | 0.2123 | 0.1890 | 3951533 | 0.2360 | 0.2013 | 3951207 | **Flat subsets EMA0.99** |

## CRPS win count (7 datasets)

- **Flat subsets**: 1
- **Flat subsets EMA0.99**: 4
- **MMPD (subset)**: 2
