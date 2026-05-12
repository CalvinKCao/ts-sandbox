# Experiment Comparison Report

100 pct runs are univariate only (no cross attn), showing that the multivariate info does give a tad bit help to my model. attn-bottleencks have 0.2 divergence penalty applied.

however results invalid bc classifier free guidance issue
### Run: 05-10-3501376-attn-bottleneck-ETTh1

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0629 | 0.7110 | -5.21% |
| ETTh2 | 0.1663 | 0.3038 | 0.1650 | 0.3016 | 0.82% |
| ETTm1 | 0.7397 | 0.5689 | 0.7486 | 0.5690 | -1.20% |
| ETTm2 | 0.1209 | 0.2488 | 0.1215 | 0.2484 | -0.49% |
### Run: 05-10-3501377-100pct-univariate-ETTh1

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0374 | 0.7024 | -2.69% |
| ETTh2 | 0.1663 | 0.3038 | 0.1676 | 0.3061 | -0.77% |
| ETTm1 | 0.7397 | 0.5689 | 0.7466 | 0.5667 | -0.94% |
| ETTm2 | 0.1209 | 0.2488 | 0.1213 | 0.2484 | -0.26% |
### Run: 05-10-3501378-attn-bottleneck-ETTh2

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0629 | 0.7110 | -5.21% |
| ETTh2 | 0.1663 | 0.3038 | 0.1650 | 0.3016 | 0.82% |
| ETTm1 | 0.7397 | 0.5689 | 0.7486 | 0.5690 | -1.20% |
| ETTm2 | 0.1209 | 0.2488 | 0.1215 | 0.2484 | -0.49% |
### Run: 05-10-3501379-100pct-univariate-ETTh2

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0374 | 0.7024 | -2.69% |
| ETTh2 | 0.1663 | 0.3038 | 0.1676 | 0.3061 | -0.77% |
| ETTm1 | 0.7397 | 0.5689 | 0.7466 | 0.5667 | -0.94% |
| ETTm2 | 0.1209 | 0.2488 | 0.1213 | 0.2484 | -0.26% |
### Run: 05-10-3501380-attn-bottleneck-ETTm1

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0629 | 0.7110 | -5.21% |
| ETTh2 | 0.1663 | 0.3038 | 0.1650 | 0.3016 | 0.82% |
| ETTm1 | 0.7397 | 0.5689 | 0.7486 | 0.5690 | -1.20% |
| ETTm2 | 0.1209 | 0.2488 | 0.1215 | 0.2484 | -0.49% |
### Run: 05-10-3501381-100pct-univariate-ETTm1

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0374 | 0.7024 | -2.69% |
| ETTh2 | 0.1663 | 0.3038 | 0.1676 | 0.3061 | -0.77% |
| ETTm1 | 0.7397 | 0.5689 | 0.7466 | 0.5667 | -0.94% |
| ETTm2 | 0.1209 | 0.2488 | 0.1213 | 0.2484 | -0.26% |
### Run: 05-10-3501382-attn-bottleneck-ETTm2

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0629 | 0.7110 | -5.21% |
| ETTh2 | 0.1663 | 0.3038 | 0.1650 | 0.3016 | 0.82% |
| ETTm1 | 0.7397 | 0.5689 | 0.7486 | 0.5690 | -1.20% |
| ETTm2 | 0.1209 | 0.2488 | 0.1215 | 0.2484 | -0.49% |
### Run: 05-10-3501383-100pct-univariate-ETTm2

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0374 | 0.7024 | -2.69% |
| ETTh2 | 0.1663 | 0.3038 | 0.1676 | 0.3061 | -0.77% |
| ETTm1 | 0.7397 | 0.5689 | 0.7466 | 0.5667 | -0.94% |
| ETTm2 | 0.1209 | 0.2488 | 0.1213 | 0.2484 | -0.26% |
### Run: 05-10-3501384-attn-bottleneck-weather

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0629 | 0.7110 | -5.21% |
| ETTh2 | 0.1663 | 0.3038 | 0.1650 | 0.3016 | 0.82% |
| ETTm1 | 0.7397 | 0.5689 | 0.7486 | 0.5690 | -1.20% |
| ETTm2 | 0.1209 | 0.2488 | 0.1215 | 0.2484 | -0.49% |
### Run: 05-10-3501385-100pct-univariate-weather

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0374 | 0.7024 | -2.69% |
| ETTh2 | 0.1663 | 0.3038 | 0.1676 | 0.3061 | -0.77% |
| ETTm1 | 0.7397 | 0.5689 | 0.7466 | 0.5667 | -0.94% |
| ETTm2 | 0.1209 | 0.2488 | 0.1213 | 0.2484 | -0.26% |
### Run: 05-10-3501386-attn-bottleneck-exchange-rate

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0629 | 0.7110 | -5.21% |
| ETTh2 | 0.1663 | 0.3038 | 0.1650 | 0.3016 | 0.82% |
| ETTm1 | 0.7397 | 0.5689 | 0.7486 | 0.5690 | -1.20% |
| ETTm2 | 0.1209 | 0.2488 | 0.1215 | 0.2484 | -0.49% |
### Run: 05-10-3501387-100pct-univariate-exchange-rate

| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |
|---------|------------|------------|---------------|---------------|-------------------|
| ETTh1 | 1.0102 | 0.6934 | 1.0374 | 0.7024 | -2.69% |
| ETTh2 | 0.1663 | 0.3038 | 0.1676 | 0.3061 | -0.77% |
| ETTm1 | 0.7397 | 0.5689 | 0.7466 | 0.5667 | -0.94% |
| ETTm2 | 0.1209 | 0.2488 | 0.1213 | 0.2484 | -0.26% |

## Missing or Incomplete Runs

- 05-10-3501409-electricity-electricity
