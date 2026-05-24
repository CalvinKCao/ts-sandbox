```python
import json

markdown_content = """# SimDiff: Reproduction Guide

This document extracts the core mathematical formulations, architectural choices, training objectives, and algorithms necessary to reproduce the **SimDiff** (Simpler Yet Better Diffusion Model for Time Series Point Forecasting) framework, excluding all extraneous sections (abstract, introductions, related works, ablations, etc.).

---

## 1. Problem Formulation
Given a sequence of multivariate time series observations $X = x_{-L+1}:x_0$ (where each $x_t$ is an $M$-dimensional vector and $L$ is the lookback window), the goal is to forecast $H$ future values $Y = x_1:x_H$.

---

## 2. Diffusion Process

### Forward Diffusion Process
The forward diffusion step for future window $Y$ at step $k$ (where $k \in \{1, \dots, K\}$) is given by:
$$Y_k = \sqrt{\overline{\\alpha}_k}Y_0 + \sqrt{1 - \overline{\\alpha}_k}\epsilon$$
where $\epsilon \sim \mathcal{N}(0, I)$ with the same size as $Y$.

### Backward Denoising Process
The backward denoising process reconstructs the future time series $Y$ using a denoising transformer backbone conditioned on past observations $c$:
$$p_\\theta(Y_{k-1}|Y_k, c) = \mathcal{N}(Y_{k-1}; \mu_\\theta(Y_k, k|c), \sigma_k^2 I)$$

The predicted mean $\mu_\\theta(Y_k, k|c)$ is parameterized as:
$$\mu_\\theta(Y_k, k|c) = \\frac{\sqrt{\\alpha_k}(1-\overline{\\alpha}_{k-1})}{1-\overline{\\alpha}_k}Y_k + \\frac{\sqrt{\overline{\\alpha}_{k-1}}\\beta_k}{1-\overline{\\alpha}_k}Y_\\theta(Y_k, k|c)$$

### Inference Process
During inference, generation starts from pure noise $\hat{Y}_K \sim \mathcal{N}(0, I)$. Each denoising step $k$ updates the state via:
$$\hat{Y}_{k-1} = \\frac{\sqrt{\\alpha_k}(1-\overline{\\alpha}_{k-1})}{1-\overline{\\alpha}_k}\hat{Y}_k + \\frac{\sqrt{\overline{\\alpha}_{k-1}}\\beta_k}{1-\overline{\\alpha}_k}Y_\\theta(\hat{Y}_k, k|c) + \sigma_k\epsilon$$
where $\epsilon \sim \mathcal{N}(0, I)$ when $k > 1$, and $\epsilon = 0$ when $k = 1$.

---

## 3. Training Objective
SimDiff uses a weighted Mean Absolute Error (MAE) loss as the denoising objective to allow the model to focus on periods with higher noise levels:
$$\mathcal{L}(\\theta) = \min_\\theta \mathbb{E}_{Y^0 \sim q(Y^0), \epsilon \sim \mathcal{N}(0,I), k} \left\| \\frac{Y^0 - Y_\\theta(Y^k, k|c)}{\sqrt{1 - \\alpha_{cumprod[k]}}} \\right\|_1$$

---

## 4. Key Architectural & Algorithmic Components

### A. Normalization Independence (N.I.)
Past and future segments are normalized independently to mitigate distribution drift. 

**Algorithm 1: N.I. in Training vs. Inference**
1. Compute $\mu_X, \sigma_X$ from past window $X$.
2. Normalize $X$: $X_{norm} = \gamma \cdot \\frac{X - \mu_X}{\sigma_X} + \\beta$ (where $\gamma, \\beta$ are learnable affine parameters).
3. **If Training:**
   * Compute $\mu_Y, \sigma_Y$ from future window $Y$.
   * Normalize $Y$: $Y_{norm} = \\frac{Y - \mu_Y}{\sigma_Y}$.
   * Corrupt $Y_{norm}$ with diffusion noise; optimize the training loss.
4. **If Inference:**
   * Sample standard Gaussian noise $\epsilon$.
   * Perform conditional DDPM denoising on $\epsilon$ conditioned on $X_{norm}$, producing $\hat{Y}_{norm}$.
   * De-normalize the prediction: $\hat{Y} = \sigma_X \cdot \\frac{\hat{Y}_{norm} - \\beta}{\gamma} + \mu_X$.

### B. Transformer Denoising Network
The denoiser $Y_\\theta$ is a unified Transformer with the following specific choices:
* **Patch-based Tokenization:** Time series are converted into overlapping patches/tokens via a dense MLP. Diffusion timesteps are processed into a time token and concatenated.
* **Rotary Position Embedding (RoPE):** Used to capture relative temporal order and dependencies in long-term forecasting.
* **Channel Independence:** Each channel/variate is processed separately, improving efficiency and data volume for distribution learning.
* **No Skip Connections:** Skip connections (like those in U-Net/U-ViT architectures) are explicitly removed to prevent noise amplification that distorts diffusion distributions.

### C. Median-of-Means (MoM) Ensemble
To transition from probabilistic diffusion sampling to stable, precise point estimation, SimDiff aggregates multiple sampled traces using the MoM estimator:
1. Divide a generated sample set of size $n$ into $K$ subsamples of size $B$.
2. Compute the mean of each subsample: $\hat{\mu}_1, \dots, \hat{\mu}_K$.
3. Take the median of these means.
4. Repeat this process $R$ times with shuffled data.
5. **Final Point Estimate:** Compute the average of the $R$ medians:
   $$\hat{\mu}_{MoM} = \\frac{1}{R} \sum_{r=1}^R \\text{median}(\hat{\mu}_1^{(r)}, \dots, \hat{\mu}_K^{(r)})$$

---

## 5. Datasets & Preprocessing (Appendix Details)

### Datasets
SimDiff was evaluated on 9 real-world datasets:
1.  **ETT (Electricity Transformer Temperature):** Includes ETTh1, ETTh2, ETTm1, ETTm2. Features: 7 (load, oil temperature, etc.).
2.  **Weather:** Local climatological data with 21 weather indicators.
3.  **Electricity:** Hourly electricity consumption of 321 clients.
4.  **Traffic:** Hourly road occupancy rates measured by 862 sensors in SF.
5.  **Exchange:** Daily exchange rates of 8 foreign countries from 1990 to 2016.
6.  **NorPool:** Hourly spot prices from the Nord Pool market (Europe). Features: 1.
7.  **Caiso:** Hourly load data from the California Independent System Operator. Features: 1.
8.  **Wind:** Wind power forecasting data.

### Data Splitting
* **ETT datasets:** 60% Train, 20% Validation, 20% Test.
* **Other datasets:** 70% Train, 10% Validation, 20% Test.

---

## 6. Implementation & Hyperparameter Details (Appendix Details)

* **Lookback Window ($L$):** Set to 96 for all datasets.
* **Prediction Horizons ($H$):** Evaluated at 96, 192, 336, and 720 steps.
* **Diffusion Timesteps ($K$):** 100 steps.
* **Sampling Strategy:** For MoM estimation, 50 probabilistic traces are sampled ($n=50$). The subsample size $B$ is set to 5.
* **Network Architecture (Transformer):**
    * Layers: 2
    * Attention Heads: 8
    * Model Dimension (d_model): 128
    * Dropout: 0.1
* **Training Details:**
    * Optimizer: Adam
    * Learning Rate: ranges between $1 \times 10^{-4}$ and $1 \times 10^{-3}$ (adjusted via early stopping within 10 epochs).
    * Batch Size: 32 (varies slightly depending on dataset size).
    * Loss: L1 Loss (Weighted MAE).

---

## 7. Main Results Table

**Table 2: Testing MSE in multivariate settings (Ranked)**
*(Select Baseline Diffusion Models vs. SimDiff)*

| Method | NorPool | Caiso | Traffic | Electricity | Exchange | Weather | ETTh1 | ETTm1 | Wind |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SimDiff (OURS)** | **0.534** | 0.106 | 0.383 | **0.145** | 0.299 | **0.015** | **0.394** | **0.322** | **0.880** |
| mr-Diff | 0.645 | 0.155 | 0.411 | 0.340 | 0.881 | 0.016 | 0.411 | 0.340 | 0.881 |
| TimeDiff | 0.474 | 0.193 | 0.407 | 0.336 | 0.896 | 0.296 | 0.407 | 0.336 | 0.896 |
| TimeGrad | 0.127 | 0.736 | 0.993 | 0.874 | 1.209 | 0.311 | 0.993 | 0.874 | 1.209 |
| TMDM | 0.665 | 0.267 | 0.535 | 0.436 | 0.901 | 0.018 | 0.535 | 0.436 | 0.901 |
| CSDI | 1.011 | 0.253 | - | - | 1.066 | 0.077 | 0.497 | 0.529 | 1.066 |
| SSSD | 0.872 | 0.195 | 0.642 | 0.255 | 1.188 | 0.061 | 0.726 | 0.464 | 1.188 |
| DLinear | 0.670 | 0.461 | 0.389 | 0.215 | 0.899 | 0.022 | 0.415 | 0.345 | 0.899 |
| PatchTST | 0.851 | 0.193 | 0.831 | 0.225 | 1.070 | 0.047 | 0.526 | 0.372 | 1.070 |
"""

with open("SimDiff_Reproduction_Guide.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("Markdown file generated successfully.")


```

```text
Markdown file generated successfully.


```

Here is the fully updated, detailed reproduction guide incorporating all the experimental settings, dataset specifics, and hyperparameter configurations extracted directly from the newly provided appendices.

---

# SimDiff: Complete Reproduction Guide

## 1. Problem Formulation

Given a sequence of multivariate time series observations $X = x_{-L+1}:x_0$, where each $x_t$ is an $M$-dimensional vector and $L$ is the lookback window, the goal is to forecast $H$ future values $Y = x_1:x_H$.

## 2. Diffusion and Denoising Process

**Forward Diffusion Process**
The forward diffusion step for the future window $Y$ at step $k$ (where $k=1, \dots, K$) is defined as:


$$Y_k = \sqrt{\overline{\alpha}_k}Y_0 + \sqrt{1 - \overline{\alpha}_k}\epsilon$$


where the noise matrix $\epsilon$ is sampled from $\mathcal{N}(0,I)$ with the same size as $Y$.

**Backward Denoising Process**
The backward denoising process reconstructs $Y$ through a unified denoising transformer backbone. Each step $k$ is formulated as:


$$p_\theta(Y_{k-1}|Y_k, c) = \mathcal{N}(Y_{k-1}; \mu_\theta(Y_k, k|c), \sigma_k^2 I)$$


The mean $\mu_\theta(Y_k, k|c)$ is parameterized as:


$$\mu_\theta(Y_k, k|c) = \frac{\sqrt{\alpha_k}(1-\overline{\alpha}_{k-1})}{1-\overline{\alpha}_k}Y_k + \frac{\sqrt{\overline{\alpha}_{k-1}}\beta_k}{1-\overline{\alpha}_k}Y_\theta(Y_k, k|c)$$

**Inference Process**
Initialization starts from pure noise $\hat{Y}_K \sim \mathcal{N}(0, I)$. For each step $k$, the update is:


$$\hat{Y}_{k-1} = \frac{\sqrt{\alpha_k}(1-\overline{\alpha}_{k-1})}{1-\overline{\alpha}_k}\hat{Y}_k + \frac{\sqrt{\overline{\alpha}_{k-1}}\beta_k}{1-\overline{\alpha}_{k}}Y_\theta(\hat{Y}_k, k|c) + \sigma_k\epsilon$$


where $\epsilon \sim \mathcal{N}(0, I)$ when $k > 1$, and $\epsilon = 0$ otherwise.

## 3. Training Objective

SimDiff uses a weighted Mean Absolute Error (MAE) loss rather than standard likelihood objectives to focus learning on periods with higher noise levels:


$$L(\theta) = \min_\theta \mathbb{E}_{Y^0, \epsilon, k} \left[ \frac{|Y^0 - Y_\theta(Y^k, k|c)|}{\sqrt{1 - \alpha_{cumprod}[k]}} \right]$$

## 4. Key Architectural Components

* **Normalization Independence (N.I.):** Past $X$ and future $Y$ are instance-normalized *independently* during training using their own statistics to prevent data leakage and handle distribution drift. A learnable affine layer $(\gamma, \beta)$ is applied to the past sequence to predict future scale shifts at test time.
* **Patch-based Tokenization:** Time series are converted into overlapping tokens via a dense MLP.
* **Time Token Embedding:** Diffusion timesteps are processed into a time token using a **straightforward linear embedding** (chosen over sinusoidal or activation functions) and concatenated with the data tokens.
* **Rotary Position Embedding (RoPE):** Crucial for ensuring the model maintains temporal context and focuses attention specifically on patterns associated with distribution drift.
* **Channel Independence & No Skip Connections:** Channels are processed independently. Skip connections and cross-channel attention are explicitly removed, as they introduce noisy, unrelated information that destabilizes diffusion convergence.

## 5. Datasets & Preprocessing (Appendix E)

The model dynamically selects history lengths $L \in \{96, 192, 336, 720, 1440\}$ using the validation set.

* **ETTh1:** 7 Dim, Hourly. History: 336. Prediction $H$: 168.
* **ETTm1:** 7 Dim, 15 min. History: 1440. Prediction $H$: 192.
* **Exchange:** 8 Dim, Daily. History: 96. Prediction $H$: 14.
* **Weather:** 21 Dim, 10 min. History: 1440. Prediction $H$: 672.
* **Electricity:** 321 Dim, Hourly. History: 720. Prediction $H$: 168.
* **Traffic:** 862 Dim, Hourly. History: 1440. Prediction $H$: 168.
* **Wind:** 7 Dim, 15 min. History: 1440. Prediction $H$: 192.
* **NorPool:** 18 Dim, Hourly. History: 1440. Prediction $H$: 720.
* **Caiso:** 10 Dim, Hourly. History: 1440. Prediction $H$: 720.

## 6. Hyperparameters & Implementation Details (Appendix E)

* **Optimizer:** Adam.
* **Learning Rate:** $1 \times 10^{-3}$.
* **Epochs:** Maximum of 100 epochs, with early stopping enabled.
* **Diffusion Steps (Training):** $K = 100$ steps (empirically found to provide optimal results).
* **Noise Schedule:** Cosine noise schedule with $\beta_t \in [0, 0.999]$ and an offset $s = 5$. This "weak first, strong later" progression was found highly effective for time series.
* **Inference Sampler:** DPM-Solver is used to speed up generation, reducing the required denoising steps during inference to **fewer than 5 steps** (usually 2 or 3 steps is sufficient).
* **Inference Skip Type:** Time Quadratic skips work best for lower-variable datasets (ETTh1, ETTm1), while Time Uniform skips work best for higher-variable datasets (Weather).

## 7. Median-of-Means (MoM) Configuration

To translate probabilistic distribution samples into a precise point estimate, SimDiff runs multiple inference traces and aggregates them:

1. **Total Inference Runs:** Set to **100** for numerical convenience (though improvements become trivial after ~30 runs).
2. **Groups ($K$):** The samples are divided into **3 to 5 groups** (more groups yielded no additional benefit).
3. **Shuffles ($R$):** The grouping and median-taking process is repeated **10 times** with shuffled data.
4. **Final Estimate:** The average of the $R$ medians.

## 8. Main Results Table (Testing MSE - Point Forecasting)

| Method | NorPool | Caiso | Traffic | Electricity | Exchange | Weather | ETTh1 | ETTm1 | Wind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SimDiff (OURS)** | **0.534** | 0.106 | 0.383 | **0.145** | 0.299 | **0.015** | **0.394** | **0.322** | **0.880** |
| mr-Diff | 0.645 | 0.155 | 0.411 | 0.340 | 0.881 | 0.016 | 0.411 | 0.340 | 0.881 |
| TimeDiff | 0.474 | 0.193 | 0.407 | 0.336 | 0.896 | 0.296 | 0.407 | 0.336 | 0.896 |
| TimeGrad | 0.127 | 0.736 | 0.993 | 0.874 | 1.209 | 0.311 | 0.993 | 0.874 | 1.209 |
| TMDM | 0.665 | 0.267 | 0.535 | 0.436 | 0.901 | 0.018 | 0.535 | 0.436 | 0.901 |
| CSDI | 1.011 | 0.253 | Out of Mem | Out of Mem | 1.066 | 0.077 | 0.497 | 0.529 | 1.066 |
| SSSD | 0.872 | 0.195 | 0.642 | 0.255 | 1.188 | 0.061 | 0.726 | 0.464 | 1.188 |
| DLinear | 0.670 | 0.461 | 0.389 | 0.215 | 0.899 | 0.022 | 0.415 | 0.345 | 0.899 |
| PatchTST | 0.851 | 0.193 | 0.831 | 0.225 | 1.070 | 0.047 | 0.526 | 0.372 | 1.070 |