# Diffusion TSF — 7-Variate Pipeline Results

**Models with diffusion eval:** 18  
**Models with iTransformer baseline:** 18

## Per-Dataset Summary

| Dataset | N | Diff MSE (avg) | Diff MAE (avg) | iTrans MSE | iTrans MAE | ΔMSE | Winner |
|---------|---|----------------|----------------|------------|------------|------|--------|
| ETTh1 | 1 | 0.7623 | 0.6272 | 0.9490 | 0.7145 | -0.1867 | **Diffusion** |
| ETTh2 | 1 | 0.1686 | 0.2955 | 0.1916 | 0.3239 | -0.0230 | **Diffusion** |
| ETTm1 | 1 | 0.8955 | 0.6036 | 0.8176 | 0.6311 | +0.0779 | iTransformer |
| ETTm2 | 1 | 0.1687 | 0.2846 | 0.1773 | 0.2992 | -0.0086 | **Diffusion** |
| electricity | 5 | 0.5325 | 0.5193 | 1.0897 | 0.8432 | -0.5572 | **Diffusion** |
| exchange_rate | 1 | 0.2199 | 0.4060 | 0.1460 | 0.3262 | +0.0739 | iTransformer |
| traffic | 5 | 0.6243 | 0.5235 | 1.3201 | 0.8762 | -0.6958 | **Diffusion** |
| weather | 3 | 0.2463 | 0.2842 | 0.2987 | 0.3293 | -0.0524 | **Diffusion** |
| **OVERALL** | **18** | **0.4854** | **0.4602** | **0.8459** | **0.6600** | **-0.3605** | **Diffusion** |


## Detailed Per-Dataset Breakdown

### ETTh1 (1 subset)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| ETTh1 | 0.7623 | 0.6272 | 0.9490 | 0.7145 | -0.1867 | 0.554 | 0.0037 |

### ETTh2 (1 subset)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| ETTh2 | 0.1686 | 0.2955 | 0.1916 | 0.3239 | -0.0230 | 0.513 | 0.0014 |

### ETTm1 (1 subset)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| ETTm1 | 0.8955 | 0.6036 | 0.8176 | 0.6311 | +0.0779 | 0.551 | 0.0080 |

### ETTm2 (1 subset)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| ETTm2 | 0.1687 | 0.2846 | 0.1773 | 0.2992 | -0.0086 | 0.558 | 0.0059 |

### electricity (5 subsets)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| electricity-0 | 0.4229 | 0.5022 | 1.0277 | 0.8435 | -0.6048 | 0.666 | 0.0069 |
| electricity-1 | 0.5844 | 0.4694 | 1.5763 | 0.9496 | -0.9919 | 0.677 | 0.0049 |
| electricity-2 | 0.4678 | 0.5336 | 0.9055 | 0.7930 | -0.4377 | 0.662 | 0.0053 |
| electricity-3 | 0.9432 | 0.7467 | 1.0671 | 0.8429 | -0.1239 | 0.621 | 0.0035 |
| electricity-4 | 0.2440 | 0.3444 | 0.8717 | 0.7870 | -0.6277 | 0.685 | 0.0050 |

### exchange_rate (1 subset)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| exchange_rate-0 | 0.2199 | 0.4060 | 0.1460 | 0.3262 | +0.0739 | 0.501 | 0.0000 |

### traffic (5 subsets)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| traffic-0 | 0.4589 | 0.4324 | 1.2092 | 0.8514 | -0.7503 | 0.763 | 0.0041 |
| traffic-1 | 0.5002 | 0.4891 | 1.1085 | 0.8311 | -0.6083 | 0.722 | 0.0011 |
| traffic-2 | 0.6167 | 0.5665 | 1.2272 | 0.8930 | -0.6105 | 0.699 | 0.0011 |
| traffic-3 | 0.8960 | 0.6593 | 1.5294 | 0.9135 | -0.6334 | 0.644 | 0.0018 |
| traffic-4 | 0.6497 | 0.4702 | 1.5261 | 0.8919 | -0.8764 | 0.757 | 0.0042 |

### weather (3 subsets)

| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |
|--------|----------|----------|------------|------------|------|-----------|----------|
| weather-0 | 0.4194 | 0.3365 | 0.4910 | 0.3919 | -0.0716 | 0.517 | 0.0034 |
| weather-1 | 0.1855 | 0.2587 | 0.2352 | 0.3095 | -0.0497 | 0.527 | 0.0051 |
| weather-2 | 0.1341 | 0.2574 | 0.1699 | 0.2864 | -0.0357 | 0.539 | 0.0044 |

## Hyperparameter Summary

- **Learning rates:** min=2.05e-06, max=4.62e-05, median=1.59e-05
- **Batch sizes:** {4: 13, 16: 5}
