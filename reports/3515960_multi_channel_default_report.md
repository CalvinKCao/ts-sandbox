# Experiment Comparison Report: Multi-Channel Default Configuration

This report compares the performance of the iTransformer baseline against the Diffusion model (Single and Averaged Ensemble) using the default multi-channel configuration on ETTm1, exchange_rate, and weather datasets.

## Summary of Results

The experiments evaluated the models on multivariate time-series datasets. In these specific runs, the iTransformer baseline significantly outperformed the Diffusion model across all metrics.

| Run ID | Dataset | Model | MSE | MAE | Improvement vs iTrans (MSE) |
|--------|---------|-------|-----|-----|-----------------------------|
| 3515960 | ETTm1 | iTransformer | 0.4841 | 0.4679 | - |
| 3515960 | ETTm1 | Diffusion (Single) | 2.5408 | 1.0986 | -424.84% |
| 3515960 | ETTm1 | Diffusion (Avg) | 2.5145 | 1.0925 | -419.41% |
| | | | | | |
| 3515961 | exchange_rate | iTransformer | 0.2033 | 0.3457 | - |
| 3515961 | exchange_rate | Diffusion (Single) | 1.4297 | 0.9780 | -603.25% |
| 3515961 | exchange_rate | Diffusion (Avg) | 1.3875 | 0.9669 | -582.49% |
| | | | | | |
| 3515962 | weather | iTransformer | 0.1967 | 0.2481 | - |
| 3515962 | weather | Diffusion (Single) | 0.5889 | 0.5099 | -199.39% |
| 3515962 | weather | Diffusion (Avg) | 0.5856 | 0.5069 | -197.71% |

## Observations

1. **iTransformer Superiority**: The iTransformer baseline consistently achieves much lower MSE and MAE compared to the Diffusion model in this multi-channel setup.
2. **Diffusion Performance**: The Diffusion model shows significantly higher error rates, suggesting that the current "default" hyperparameters or the multi-channel approach might need further tuning for these specific datasets.
3. **Ensemble Benefit**: The Averaged Ensemble of the Diffusion model consistently provides a slight improvement over the Single sample, though it remains far behind the baseline.
