# Consolidated Experiment Comparison Report 
Dit one-variate at a time runs - best results with base
Results kinda collapsed into copying itrans exactly, perhaps due to the divergence from itrans penalty

<!-- 
Comparing iTransformer Baseline vs Diffusion (Avg Ensemble)

### Run: 05-11-3515961-multi-channel-default-exchange-rate *(Duration: 0h 11m 52s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| exchange_rate | 0.2033 | 0.3457 | 1.3875 | 0.9669 | -582.49% |

### Run: 05-11-3515962-multi-channel-default-weather *(Duration: 0h 15m 19s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| weather | 0.1967 | 0.2481 | 0.5856 | 0.5069 | -197.71% |

### Run: 05-11-3516894-attn-bottleneck-ETTh1 *(Duration: 2h 51m 49s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| attn-bottleneck-pen-0.2 | 0.7517 | 0.6177 | 0.7565 | 0.6161 | -0.64% |

### Run: 05-11-3516895-100pct-univariate-ETTh1 *(Duration: 1h 17m 2s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| 100pct-univariate-pen-0.2 | 0.7517 | 0.6177 | 0.7482 | 0.6151 | 0.47% |

### Run: 05-11-3516896-dit-ETTh1 *(Duration: 0h 14m 47s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen-0.2 | 0.7517 | 0.6177 | 0.7000 | 0.5956 | 6.88% |

### Run: 05-11-3516897-attn-bottleneck-ETTm1 *(Duration: 3h 31m 31s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| attn-bottleneck-pen-0.2 | 0.4841 | 0.4679 | 0.4610 | 0.4541 | 4.77% |

### Run: 05-11-3516898-100pct-univariate-ETTm1 *(Duration: 1h 34m 38s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| 100pct-univariate-pen-0.2 | 0.4841 | 0.4679 | 0.4635 | 0.4553 | 4.26% |

### Run: 05-11-3516899-dit-ETTm1 *(Duration: 0h 19m 34s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen-0.2 | 0.4841 | 0.4679 | 0.4712 | 0.4610 | 2.66% |

### Run: 05-11-3516900-attn-bottleneck-exchange-rate *(Duration: 3h 6m 53s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| attn-bottleneck-pen-0.2 | 0.2033 | 0.3457 | 0.2692 | 0.4038 | -32.42% |

### Run: 05-11-3516901-100pct-univariate-exchange-rate *(Duration: 1h 19m 55s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| 100pct-univariate-pen-0.2 | 0.2033 | 0.3457 | 0.2431 | 0.3870 | -19.58% |

### Run: 05-11-3516902-dit-exchange-rate *(Duration: 0h 15m 40s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen-0.2 | 0.2033 | 0.3457 | 0.1592 | 0.3081 | 21.69% |

### Run: 05-11-3516903-attn-bottleneck-weather *(Duration: 2h 38m 30s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| None (Failed/Incomplete) | N/A | N/A | N/A | N/A | N/A |

### Run: 05-11-3516904-100pct-univariate-weather *(Duration: 4h 8m 11s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| None (Failed/Incomplete) | N/A | N/A | N/A | N/A | N/A | -->

### Run: 05-11-3516905-dit-weather *(Duration: 0h 47m 46s)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen-0.2 | 0.1967 | 0.2481 | 0.2020 | 0.2489 | -2.69% |

### Run: 05-11-3522162-dit-ETTh1 *(Started: 05-11 22:27:50)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-ETTh1 | 0.7517 | 0.6177 | 0.7077 | 0.5994 | 5.85% |

### Run: 05-11-3522163-dit-ETTh2 *(Started: 05-11 22:27:50)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-ETTh2 | 0.1706 | 0.2909 | 0.1691 | 0.2896 | 0.88% |

### Run: 05-11-3522164-dit-ETTm1 *(Started: 05-11 22:27:54)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-ETTm1 | 0.4841 | 0.4679 | 0.4726 | 0.4619 | 2.38% |

### Run: 05-11-3522165-dit-ETTm2 *(Started: 05-11 22:27:59)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-ETTm2 | 0.1182 | 0.2420 | 0.1181 | 0.2417 | 0.08% |

### Run: 05-11-3522167-dit-exchange-rate *(Started: 05-11 22:28:07)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-exchange_rate | 0.2033 | 0.3457 | 0.1733 | 0.3193 | 14.76% |

### Run: 05-11-3522170-dit-h128-ETTh1 *(Started: 05-11 22:28:11)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-h128-ETTh1 | 0.7517 | 0.6177 | 0.7159 | 0.5998 | 4.76% |

### Run: 05-11-3522171-dit-h128-ETTh2 *(Started: 05-11 22:28:14)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-h128-ETTh2 | 0.1706 | 0.2909 | 0.1717 | 0.2917 | -0.64% |

### Run: 05-11-3522172-dit-h128-ETTm1 *(Started: 05-11 22:28:17)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-h128-ETTm1 | 0.4841 | 0.4679 | 0.4742 | 0.4618 | 2.05% |

### Run: 05-11-3522173-dit-h128-ETTm2 *(Started: 05-11 22:28:20)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-h128-ETTm2 | 0.1182 | 0.2420 | 0.1180 | 0.2413 | 0.17% |

### Run: 05-11-3522174-dit-h128-weather *(Started: 05-11 22:28:20)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-h128-weather | 0.1967 | 0.2481 | 0.1984 | 0.2479 | -0.86% |

### Run: 05-11-3522175-dit-h128-exchange-rate *(Started: 05-11 22:28:21)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-h128-exchange_rate | 0.2033 | 0.3457 | 0.1842 | 0.3318 | 9.39% |

### Run: 05-11-3522176-dit-pen0-ETTh1 *(Started: 05-11 22:28:20)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen0-ETTh1 | 0.7517 | 0.6177 | 0.6854 | 0.5908 | 8.82% |

### Run: 05-11-3522177-dit-pen0-ETTh2 *(Started: 05-11 22:28:23)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen0-ETTh2 | 0.1706 | 0.2909 | 0.1633 | 0.2849 | 4.28% |

### Run: 05-11-3522178-dit-pen0-ETTm1 *(Started: 05-11 22:28:26)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen0-ETTm1 | 0.4841 | 0.4679 | 0.4762 | 0.4656 | 1.63% |

### Run: 05-11-3522179-dit-pen0-ETTm2 *(Started: 05-11 22:28:35)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen0-ETTm2 | 0.1182 | 0.2420 | 0.1186 | 0.2420 | -0.34% |

### Run: 05-11-3522180-dit-pen0-weather *(Started: 05-11 22:28:39)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen0-weather | 0.1967 | 0.2481 | 0.2095 | 0.2520 | -6.51% |

### Run: 05-11-3522181-dit-pen0-exchange-rate *(Started: 05-11 22:28:39)*

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| dit-pen0-exchange_rate | 0.2033 | 0.3457 | 0.8965 | 0.7741 | -340.97% |
