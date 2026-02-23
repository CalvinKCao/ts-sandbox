# Diffusion TSF — 7-Variate Pipeline Results

**Models with diffusion eval:** 80  
**Models with iTransformer baseline:** 0

## Per-Dataset Summary

| Dataset | N | Avg MSE (single) | Avg MAE (single) | Avg MSE (30-avg) | Avg MAE (30-avg) | Avg Trend Acc |
|---------|---|------------------|------------------|------------------|------------------|---------------|
| ETTh1 | 1 | 0.8872 | 0.6885 | 0.7623 | 0.6272 | 0.554 |
| ETTh2 | 1 | 0.1849 | 0.3117 | 0.1686 | 0.2955 | 0.513 |
| ETTm1 | 1 | 1.2621 | 0.7415 | 0.8955 | 0.6036 | 0.551 |
| ETTm2 | 1 | 0.2517 | 0.3518 | 0.1687 | 0.2846 | 0.558 |
| electricity | 45 | 2.5737 | 1.1279 | 1.0993 | 0.7660 | 0.643 |
| exchange_rate | 1 | 0.2414 | 0.3601 | 0.2199 | 0.4060 | 0.501 |
| traffic | 27 | 2.7555 | 1.0878 | 1.1335 | 0.6917 | 0.708 |
| weather | 3 | 0.4010 | 0.3723 | 0.2463 | 0.2842 | 0.527 |
| **OVERALL** | **80** | | | **1.0379** | **0.7027** | |

## Best & Worst Models (diffusion avg MSE)

### Top 5 Best
| Rank | Subset | Diff MSE |
|------|--------|----------|
| 1 | weather-2 | 0.1341 |
| 2 | ETTh2 | 0.1686 |
| 3 | ETTm2 | 0.1687 |
| 4 | weather-1 | 0.1855 |
| 5 | exchange_rate-0 | 0.2199 |

### Top 5 Worst
| Rank | Subset | Diff MSE |
|------|--------|----------|
| 1 | traffic-15 | 2.1336 |
| 2 | electricity-7 | 2.1849 |
| 3 | electricity-43 | 2.2734 |
| 4 | electricity-31 | 2.5932 |
| 5 | electricity-12 | 3.9662 |

## Detailed Per-Dataset Breakdown

### ETTh1 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTh1 | 0.8872 | 0.6885 | 0.7623 | 0.6272 | 0.554 | 0.0037 |

### ETTh2 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTh2 | 0.1849 | 0.3117 | 0.1686 | 0.2955 | 0.513 | 0.0014 |

### ETTm1 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTm1 | 1.2621 | 0.7415 | 0.8955 | 0.6036 | 0.551 | 0.0080 |

### ETTm2 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTm2 | 0.2517 | 0.3518 | 0.1687 | 0.2846 | 0.558 | 0.0059 |

### electricity (45 subsets)

- **Diffusion MSE:** min=0.2440, max=3.9662, mean=1.0993, std=0.7137
- **Diffusion MAE:** min=0.3444, max=1.6169, mean=0.7660, std=0.2501

### exchange_rate (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| exchange_rate-0 | 0.2414 | 0.3601 | 0.2199 | 0.4060 | 0.501 | 0.0000 |

### traffic (27 subsets)

- **Diffusion MSE:** min=0.4589, max=2.1336, mean=1.1335, std=0.4726
- **Diffusion MAE:** min=0.4324, max=1.0195, mean=0.6917, std=0.1674

### weather (3 subsets)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| weather-0 | 0.6574 | 0.4367 | 0.4194 | 0.3365 | 0.517 | 0.0034 |
| weather-1 | 0.2580 | 0.3239 | 0.1855 | 0.2587 | 0.527 | 0.0051 |
| weather-2 | 0.2877 | 0.3562 | 0.1341 | 0.2574 | 0.539 | 0.0044 |

## Hyperparameter Summary

- **Learning rates:** min=2.05e-06, max=4.62e-05, median=1.59e-05
- **Batch sizes:** {4: 13, 16: 41, 64: 18, 32: 8}
