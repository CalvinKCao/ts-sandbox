# Diffusion TSF — 7-Variate Pipeline Results

**Total models evaluated:** 80

## Per-Dataset Summary

| Dataset | Subsets | Avg MSE (single) | Avg MAE (single) | Avg MSE (30-avg) | Avg MAE (30-avg) | Avg Trend Acc |
|---------|---------|-------------------|-------------------|-------------------|-------------------|---------------|
| ETTh1 | 1 | 3.2932 | 1.2386 | 1.3533 | 0.7990 | 0.552 |
| ETTh2 | 1 | 0.4894 | 0.4518 | 0.2505 | 0.3582 | 0.580 |
| ETTm1 | 1 | 0.8011 | 0.5871 | 0.5199 | 0.4619 | 0.553 |
| ETTm2 | 1 | 0.3788 | 0.4150 | 0.1783 | 0.2993 | 0.554 |
| electricity | 45 | 2.7246 | 1.1714 | 1.1681 | 0.7953 | 0.639 |
| exchange_rate | 1 | 0.2952 | 0.3415 | 0.1394 | 0.2577 | 0.444 |
| traffic | 27 | 3.2800 | 1.2210 | 1.2955 | 0.7537 | 0.704 |
| weather | 3 | 0.4238 | 0.3633 | 0.2164 | 0.2803 | 0.547 |
| **OVERALL** | **80** | **2.7212** | **1.1226** | **1.1329** | **0.7395** | **0.651** |

## Best & Worst Models (by averaged MSE)

### Top 5 Best
| Rank | Subset | MSE |
|------|--------|-----|
| 1 | exchange_rate-0 | 0.1394 |
| 2 | weather-2 | 0.1653 |
| 3 | ETTm2 | 0.1783 |
| 4 | weather-1 | 0.2084 |
| 5 | ETTh2 | 0.2505 |

### Top 5 Worst
| Rank | Subset | MSE |
|------|--------|-----|
| 1 | traffic-3 | 2.2560 |
| 2 | electricity-43 | 2.2734 |
| 3 | electricity-0 | 2.5174 |
| 4 | electricity-31 | 2.5932 |
| 5 | electricity-12 | 3.9662 |

## Detailed Per-Dataset Breakdown

### ETTh1 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTh1 | 3.2932 | 1.2386 | 1.3533 | 0.7990 | 0.552 | 0.0128 |

### ETTh2 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTh2 | 0.4894 | 0.4518 | 0.2505 | 0.3582 | 0.580 | 0.0099 |

### ETTm1 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTm1 | 0.8011 | 0.5871 | 0.5199 | 0.4619 | 0.553 | 0.0123 |

### ETTm2 (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| ETTm2 | 0.3788 | 0.4150 | 0.1783 | 0.2993 | 0.554 | 0.0103 |

### electricity (45 subsets)

- **MSE:** min=0.4378, max=3.9662, mean=1.1681, std=0.7228
- **MAE:** min=0.4977, max=1.6169, mean=0.7953, std=0.2435

### exchange_rate (1 subset)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| exchange_rate-0 | 0.2952 | 0.3415 | 0.1394 | 0.2577 | 0.444 | 0.0011 |

### traffic (27 subsets)

- **MSE:** min=0.7425, max=2.2560, mean=1.2955, std=0.4877
- **MAE:** min=0.5412, max=1.0735, mean=0.7537, std=0.1711

### weather (3 subsets)

| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |
|--------|------------|------------|---------|---------|-----------|----------|
| weather-0 | 0.5853 | 0.3800 | 0.2756 | 0.2810 | 0.566 | 0.0062 |
| weather-1 | 0.3882 | 0.3596 | 0.2084 | 0.2852 | 0.539 | 0.0078 |
| weather-2 | 0.2979 | 0.3503 | 0.1653 | 0.2747 | 0.535 | 0.0060 |

## Hyperparameter Summary

- **Learning rates:** min=2.05e-06, max=4.62e-05, median=1.59e-05
- **Batch sizes:** {64: 27, 32: 10, 16: 43}
