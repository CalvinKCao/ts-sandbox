# Binary Anchor Diffusion for Time Series Forecasting

## Architecture

![Architectural diagram](diagram.png)

*Architectural diagram*

## Probabilistic forecasts

![Forecast comparison: our model vs iTransformer vs MMPD](viz_comparison.png)

Top: **our model** · Middle: **deterministic iTransformer** · Bottom: **baseline diffusion model from MMPD** (ICLR 2026).

Blue is ground truth; other colors are possible futures sampled from each model. Our model captures step functions and flatlines more faithfully than the baselines.

## Benchmark comparison (anchor eval)

Anchor-eval MSE/MAE (`n_samples=30`, `eval_sampler: anchor`) from [reports/3819108_binary_dual_scale_7v_etth1_cap_report.md](reports/3819108_binary_dual_scale_7v_etth1_cap_report.md) — the most recent report with anchor stats — restricted to datasets that also appear in the [iTransformer](https://github.com/thuml/iTransformer) long-horizon benchmark. **Bold** = best per row (lower is better).

| Dataset | Our MSE | Our MAE | iTransformer MSE | iTransformer MAE |
|---------|--------:|--------:|-----------------:|-----------------:|
| ECL | 0.2487 | **0.2604** | **0.178** | 0.270 |
| Exchange | **0.1041** | **0.2246** | 0.360 | 0.403 |
| Traffic | **0.3577** | **0.2603** | 0.428 | 0.282 |
| Weather | **0.1210** | **0.2334** | 0.258 | 0.278 |
| Solar-Energy | 0.2540 | **0.2507** | **0.233** | 0.262 |

Our runs use 7-variate subsets (`*_7v_s*`) for the large datasets; iTransformer numbers are from the published full-dataset benchmark (horizon 96).

### iTransformer paper benchmark (all methods, MSE / MAE)

| Dataset | iTransformer | RLinear | PatchTST | Crossformer | TiDE | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|---------|-------------:|--------:|---------:|------------:|-----:|---------:|--------:|-------:|----------:|-----------:|-----------:|
| ECL | 0.178 / 0.270 | 0.219 / 0.298 | 0.205 / 0.290 | 0.244 / 0.334 | 0.251 / 0.344 | 0.192 / 0.295 | 0.212 / 0.300 | 0.268 / 0.365 | 0.214 / 0.327 | 0.193 / 0.296 | 0.227 / 0.338 |
| ETT (Avg) | 0.383 / 0.399 | 0.380 / 0.392 | 0.381 / 0.397 | 0.685 / 0.578 | 0.482 / 0.470 | 0.391 / 0.404 | 0.442 / 0.444 | 0.689 / 0.597 | 0.408 / 0.428 | 0.471 / 0.464 | 0.465 / 0.459 |
| Exchange | 0.360 / 0.403 | 0.378 / 0.417 | 0.367 / 0.404 | 0.940 / 0.707 | 0.370 / 0.413 | 0.416 / 0.443 | 0.354 / 0.414 | 0.750 / 0.626 | 0.519 / 0.429 | 0.461 / 0.454 | 0.613 / 0.539 |
| Traffic | 0.428 / 0.282 | 0.626 / 0.378 | 0.481 / 0.304 | 0.550 / 0.304 | 0.760 / 0.473 | 0.620 / 0.336 | 0.625 / 0.383 | 0.804 / 0.509 | 0.610 / 0.376 | 0.624 / 0.340 | 0.628 / 0.379 |
| Weather | 0.258 / 0.278 | 0.272 / 0.291 | 0.259 / 0.281 | 0.259 / 0.315 | 0.271 / 0.320 | 0.259 / 0.287 | 0.265 / 0.317 | 0.292 / 0.363 | 0.309 / 0.360 | 0.288 / 0.314 | 0.338 / 0.382 |
| Solar-Energy | 0.233 / 0.262 | 0.369 / 0.356 | 0.270 / 0.307 | 0.641 / 0.639 | 0.347 / 0.417 | 0.301 / 0.319 | 0.330 / 0.401 | 0.282 / 0.375 | 0.291 / 0.381 | 0.261 / 0.381 | 0.885 / 0.711 |
| PEMS (Avg) | 0.119 / 0.218 | 0.514 / 0.482 | 0.217 / 0.305 | 0.220 / 0.304 | 0.375 / 0.440 | 0.148 / 0.246 | 0.320 / 0.394 | 0.121 / 0.222 | 0.224 / 0.327 | 0.151 / 0.249 | 0.614 / 0.575 |
