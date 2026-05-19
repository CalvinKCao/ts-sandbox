# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
*Liu et al., ICLR 2024*

---

## 3. ITRANSFORMER ARCHITECTURE

### Problem Setup

Given historical observations **X** = {x₁, …, x_T} ∈ ℝ^(T×N) with T time steps and N variates, predict future S time steps **Y** = {x_{T+1}, …, x_{T+S}} ∈ ℝ^(S×N).

- **X_{t,:}** = simultaneously recorded time points at step t
- **X_{:,n}** = the whole time series of variate n

---

### 3.1 Structure Overview

iTransformer adopts an **encoder-only** architecture including embedding, projection, and Transformer blocks.

**Core inversion:** Instead of embedding multiple variates of the same timestamp as a temporal token, iTransformer embeds the **whole time series of each variate** independently as a variate token.

Forward pass:

```
h⁰ₙ = Embedding(X_{:,n})
H^{l+1} = TrmBlock(H^l),   l = 0, …, L−1
Ŷ_{:,n} = Projection(h^L_n)
```

Where **H** = {h₁, …, h_N} ∈ ℝ^(N×D) contains N embedded variate tokens of dimension D.

- **Embedding**: ℝ^T → ℝ^D (MLP)
- **Projection**: ℝ^D → ℝ^S (MLP)
- Position embedding is **not needed** — sequence order is implicitly stored in neuron permutations of the FFN.

---

### 3.2 Inverted Transformer Components

#### Layer Normalization

Applied to the series representation of each individual variate:

```
LayerNorm(H) = { (hₙ − Mean(hₙ)) / sqrt(Var(hₙ)) : n = 1, …, N }
```

Effect: normalizes each variate token to a Gaussian distribution, reducing discrepancies from inconsistent physical measurements across variates.

Contrast with vanilla Transformer: normalizing across the multivariate representation of a timestamp fuses heterogeneous variates and introduces interaction noise between noncausal/delayed processes.

#### Feed-Forward Network (FFN)

Applied identically to each variate token (series representation). By the universal approximation theorem, FFN can extract complex representations describing an entire time series — amplitude, periodicity, frequency spectra (neurons act as filters).

Key insight: the FFN acts as a combination of linear forecasters + Channel Independence. Temporal features extracted by MLPs are shared across distinct time series.

#### Self-Attention

Applied on the variate dimension. With extracted representations H = {h₀, …, h_N} ∈ ℝ^(N×D):

- Linear projections → Q, K, V ∈ ℝ^(N×dk)
- Score entry: A_{i,j} = (QK^T / √d_k)_{i,j} ∝ qᵢᵀkⱼ
- Score map **A** ∈ ℝ^(N×N) represents **multivariate correlations**

Since each variate token is normalized on its feature dimension, score entries reveal variate-wise correlation. Highly correlated variates receive higher weights for representation interaction.

#### Algorithm (Pseudo-code)

```
Input: X ∈ ℝ^(T×N)

X = X.transpose                    # X ∈ ℝ^(N×T)
H⁰ = MLP(X)                        # H⁰ ∈ ℝ^(N×D)

for l in {1, …, L}:
    H^{l-1} = LayerNorm(H^{l-1} + Self-Attn(H^{l-1}))   # variate attention
    H^l     = LayerNorm(H^{l-1} + Feed-Forward(H^{l-1})) # series representations

Ŷ = MLP(H^L)                       # Ŷ ∈ ℝ^(N×S)
Ŷ = Ŷ.transpose                    # Ŷ ∈ ℝ^(S×N)
return Ŷ
```

---

## 4. EXPERIMENTS

### Datasets

| Dataset       | Dim  | Freq   | Domain         | Prediction Lengths       |
|---------------|------|--------|----------------|--------------------------|
| ETTh1/ETTh2   | 7    | Hourly | Electricity    | {96, 192, 336, 720}      |
| ETTm1/ETTm2   | 7    | 15min  | Electricity    | {96, 192, 336, 720}      |
| Exchange      | 8    | Daily  | Economy        | {96, 192, 336, 720}      |
| Weather       | 21   | 10min  | Weather        | {96, 192, 336, 720}      |
| ECL           | 321  | Hourly | Electricity    | {96, 192, 336, 720}      |
| Traffic       | 862  | Hourly | Transportation | {96, 192, 336, 720}      |
| Solar-Energy  | 137  | 10min  | Energy         | {96, 192, 336, 720}      |
| PEMS03        | 358  | 5min   | Transportation | {12, 24, 48, 96}         |
| PEMS04        | 307  | 5min   | Transportation | {12, 24, 48, 96}         |
| PEMS07        | 883  | 5min   | Transportation | {12, 24, 48, 96}         |
| PEMS08        | 170  | 5min   | Transportation | {12, 24, 48, 96}         |
| Market (×6)   | 285–759 | 10min | Transaction  | {12, 24, 72, 144}        |

Lookback length T = 96 for all datasets except Market (T = 144).

### Implementation

- Framework: PyTorch, single NVIDIA P100 16GB GPU
- Optimizer: Adam, initial lr ∈ {1e-3, 5×10-4, 1e-4}, L2 loss
- Batch size: 32, epochs: 10
- Blocks L ∈ {2, 3, 4}, token dimension D ∈ {256, 512}

---

### 4.1 Main Forecasting Results

Results averaged over all prediction lengths. **Bold** = best, _underline_ = second best. Lower MSE/MAE is better. Lookback T = 96.

| Model             | ECL MSE | ECL MAE | ETT MSE | ETT MAE | Exchange MSE | Exchange MAE | Traffic MSE | Traffic MAE | Weather MSE | Weather MAE | Solar MSE | Solar MAE | PEMS MSE | PEMS MAE |
|-------------------|---------|---------|---------|---------|-------------|-------------|------------|------------|------------|------------|----------|----------|---------|---------|
| **iTransformer**  | **0.178** | **0.270** | 0.383 | 0.399 | 0.360 | 0.403 | **0.428** | **0.282** | 0.258 | 0.278 | **0.233** | **0.262** | **0.119** | **0.218** |
| RLinear           | 0.219   | 0.298   | **0.380** | **0.392** | 0.378 | 0.417 | 0.626 | 0.378 | 0.272 | 0.291 | 0.369 | 0.356 | 0.514 | 0.482 |
| PatchTST          | 0.205   | 0.290   | 0.381 | 0.397 | 0.367 | 0.404 | 0.481 | 0.304 | 0.259 | 0.281 | 0.270 | 0.307 | 0.217 | 0.305 |
| Crossformer       | 0.244   | 0.334   | 0.685 | 0.578 | 0.940 | 0.707 | 0.550 | 0.304 | 0.259 | 0.315 | 0.641 | 0.639 | 0.220 | 0.304 |
| TiDE              | 0.251   | 0.344   | 0.482 | 0.470 | 0.370 | 0.413 | 0.760 | 0.473 | 0.271 | 0.320 | 0.347 | 0.417 | 0.375 | 0.440 |
| TimesNet          | 0.192   | 0.295   | 0.391 | 0.404 | 0.416 | 0.443 | 0.620 | 0.336 | 0.259 | 0.287 | 0.301 | 0.319 | 0.148 | 0.246 |
| DLinear           | 0.212   | 0.300   | 0.442 | 0.444 | 0.354 | 0.414 | 0.625 | 0.383 | 0.265 | 0.317 | 0.330 | 0.401 | 0.320 | 0.394 |
| SCINet            | 0.268   | 0.365   | 0.689 | 0.597 | 0.750 | 0.626 | 0.804 | 0.509 | 0.292 | 0.363 | 0.282 | 0.375 | 0.121 | 0.222 |
| FEDformer         | 0.214   | 0.327   | 0.408 | 0.428 | 0.519 | 0.429 | 0.610 | 0.376 | 0.309 | 0.360 | 0.291 | 0.381 | 0.224 | 0.327 |
| Stationary        | 0.193   | 0.296   | 0.471 | 0.464 | 0.461 | 0.454 | 0.624 | 0.340 | 0.288 | 0.314 | 0.261 | 0.381 | 0.151 | 0.249 |
| Autoformer        | 0.227   | 0.338   | 0.465 | 0.459 | 0.613 | 0.539 | 0.628 | 0.379 | 0.338 | 0.382 | 0.885 | 0.711 | 0.614 | 0.575 |

---

### 4.2 Framework Generality (iTransformers)

The inverted framework applied to various Transformer variants. MSE reduction (Promotion) shown.

#### Average Performance Promotion by Dataset

| Model       | ECL Orig | ECL Inv | ECL Promo | Traffic Orig | Traffic Inv | Traffic Promo | Weather Orig | Weather Inv | Weather Promo |
|-------------|----------|---------|-----------|-------------|------------|--------------|-------------|------------|--------------|
| Transformer | 0.277    | 0.178   | **35.6%** | 0.665       | 0.428      | **35.6%**    | 0.657       | 0.258      | **60.2%**    |
| Reformer    | 0.338    | 0.208   | **38.4%** | 0.741       | 0.647      | **12.7%**    | 0.803       | 0.248      | **69.2%**    |
| Informer    | 0.311    | 0.216   | **30.5%** | 0.764       | 0.662      | **13.3%**    | 0.634       | 0.271      | **57.3%**    |
| Flowformer  | 0.267    | 0.210   | **21.3%** | 0.750       | 0.524      | **30.1%**    | 0.286       | 0.266      | **7.2%**     |
| Flashformer | 0.285    | 0.206   | **27.8%** | 0.658       | 0.492      | **25.2%**    | 0.659       | 0.262      | **60.2%**    |

**Overall average MSE promotion:** Transformer +38.9%, Reformer +36.1%, Informer +28.5%, Flowformer +16.8%, Flashformer +32.2%.

#### Full Promotion Results (all datasets, Transformer vs iTransformer)

| Dataset       | Transformer MSE | iTransformer MSE | Promotion |
|---------------|----------------|-----------------|-----------|
| ETT           | 2.750          | 0.383           | **86.1%** |
| ECL           | 0.277          | 0.178           | **35.6%** |
| PEMS          | 0.157          | 0.113           | **28.0%** |
| Solar-Energy  | 0.256          | 0.233           | **9.0%**  |
| Traffic       | 0.665          | 0.428           | **35.6%** |
| Weather       | 0.657          | 0.258           | **60.2%** |

---

### 4.3 Model Analysis

#### Ablation Study

Components on variate and temporal dimensions swapped/removed. Average results over all prediction lengths.

| Design           | Variate   | Temporal  | ECL MSE | ECL MAE | Traffic MSE | Traffic MAE | Weather MSE | Weather MAE | Solar MSE | Solar MAE |
|------------------|-----------|-----------|---------|---------|------------|------------|------------|------------|----------|----------|
| **iTransformer** | Attention | FFN       | **0.178** | **0.270** | **0.428** | **0.282** | 0.258 | 0.278 | **0.233** | **0.262** |
| Replace          | Attention | Attention | 0.193   | 0.293   | 0.913      | 0.500      | 0.255      | 0.280      | 0.261    | 0.291    |
| Replace          | FFN       | Attention | 0.202   | 0.300   | 0.863      | 0.499      | 0.258      | 0.283      | 0.285    | 0.317    |
| Replace          | FFN       | FFN       | 0.182   | 0.287   | 0.599      | 0.348      | **0.248**  | **0.274**  | 0.269    | 0.287    |
| w/o              | Attention | —         | 0.189   | 0.278   | 0.456      | 0.306      | 0.261      | 0.281      | 0.258    | 0.289    |
| w/o              | —         | FFN       | 0.193   | 0.276   | 0.461      | 0.294      | 0.265      | 0.283      | 0.261    | 0.283    |

Key finding: vanilla Transformer (FFN on variate, Attention on temporal) performs **worst** among all designs on high-dimensional datasets.

#### CKA Representation Analysis

Centered Kernel Alignment (CKA) similarity measured between output features of first and last block. Higher CKA = more similar representations = better performance for this low-level generative task.

Result: iTransformers cluster at **high CKA (0.85–1.0) and low MSE**, while vanilla Transformers cluster at **low CKA (0.6–0.8) and high MSE** — a clear division line separating the two families.

#### Multivariate Correlation Visualization (Solar-Energy)

- **Shallow attention layers**: learned score map ≈ correlation of raw lookback series
- **Deep attention layers**: learned score map gradually converges toward correlation of future series
- Interpretation: encoding the past → decoding for the future is conducted through series representations during layer stacking

#### Lookback Length Scaling

Evaluated with T ∈ {48, 96, 192, 336, 720}, fixed S = 96.

- Vanilla Transformers: performance does **not** consistently improve with longer lookback
- iTransformers: performance **consistently improves** with longer lookback on both ECL and Traffic

#### Efficient Training Strategy

Randomly subsample a fraction of variates per batch during training; predict all variates at inference.

| Sample Ratio | ECL MSE | Traffic MSE | Solar MSE | Memory (Traffic) |
|-------------|---------|------------|----------|-----------------|
| 100%        | ~0.178  | ~0.428     | ~0.233   | ~6.0 GB         |
| 80%         | ~0.178  | ~0.430     | ~0.235   | ~3.0 GB         |
| 60%         | ~0.179  | ~0.433     | ~0.237   | ~3.0 GB         |
| 40%         | ~0.180  | ~0.435     | ~0.240   | ~1.5 GB         |
| 20%         | ~0.183  | ~0.440     | ~0.245   | ~1.5 GB         |

Performance remains largely stable while memory footprint is cut significantly.

---

## APPENDIX B: Full Ablation Results

Full per-horizon ablation on iTransformer across all designs:

| Design           | Variate   | Temporal  | H   | ECL MSE | ECL MAE | Traffic MSE | Traffic MAE | Weather MSE | Weather MAE | Solar MSE | Solar MAE |
|------------------|-----------|-----------|-----|---------|---------|------------|------------|------------|------------|----------|----------|
| **iTransformer** | Attention | FFN       | 96  | 0.148   | 0.240   | 0.395      | 0.268      | 0.174      | 0.214      | 0.203    | 0.237    |
|                  |           |           | 192 | 0.162   | 0.253   | 0.417      | 0.276      | 0.221      | 0.254      | 0.233    | 0.261    |
|                  |           |           | 336 | 0.178   | 0.269   | 0.433      | 0.283      | 0.278      | 0.296      | 0.248    | 0.273    |
|                  |           |           | 720 | 0.225   | 0.317   | 0.467      | 0.302      | 0.358      | 0.349      | 0.249    | 0.275    |
|                  |           |           | Avg | **0.178** | **0.270** | **0.428** | **0.282** | 0.258 | 0.279 | **0.233** | **0.262** |
| Attn+Attn        | Attention | Attention | 96  | 0.161   | 0.263   | 1.021      | 0.581      | 0.168      | 0.213      | 0.227    | 0.270    |
|                  |           |           | 192 | 0.180   | 0.280   | 0.834      | 0.447      | 0.217      | 0.256      | 0.255    | 0.292    |
|                  |           |           | 336 | 0.194   | 0.296   | 0.906      | 0.493      | 0.277      | 0.299      | 0.279    | 0.301    |
|                  |           |           | 720 | 0.238   | 0.331   | 0.892      | 0.477      | 0.356      | 0.351      | 0.283    | 0.300    |
|                  |           |           | Avg | 0.193   | 0.293   | 0.913      | 0.500      | 0.255      | 0.280      | 0.261    | 0.291    |
| FFN+Attn         | FFN       | Attention | 96  | 0.169   | 0.270   | 0.907      | 0.540      | 0.176      | 0.221      | 0.247    | 0.299    |
|                  |           |           | 192 | 0.189   | 0.292   | 0.839      | 0.489      | 0.224      | 0.261      | 0.275    | 0.305    |
|                  |           |           | 336 | 0.204   | 0.304   | 0.248      | 0.364      | 0.279      | 0.301      | 0.317    | 0.337    |
|                  |           |           | 720 | 0.245   | 0.335   | 1.059      | 0.606      | 0.354      | 0.347      | 0.301    | 0.329    |
|                  |           |           | Avg | 0.202   | 0.300   | 0.863      | 0.499      | 0.258      | 0.283      | 0.285    | 0.317    |
| FFN+FFN          | FFN       | FFN       | 96  | 0.159   | 0.261   | 0.606      | 0.342      | 0.162      | 0.207      | 0.237    | 0.277    |
|                  |           |           | 192 | 0.171   | 0.271   | 0.559      | 0.342      | 0.211      | 0.252      | 0.273    | 0.293    |
|                  |           |           | 336 | 0.187   | 0.287   | 0.569      | 0.348      | 0.270      | 0.293      | 0.284    | 0.287    |
|                  |           |           | 720 | 0.211   | 0.307   | 0.664      | 0.359      | 0.349      | 0.345      | 0.284    | 0.289    |
|                  |           |           | Avg | 0.182   | 0.287   | 0.599      | 0.348      | **0.248**  | **0.274**  | 0.269    | 0.287    |
| w/o Attention    | Attn      | —         | Avg | 0.189   | 0.278   | 0.456      | 0.306      | 0.261      | 0.281      | 0.258    | 0.289    |
| w/o FFN          | —         | FFN       | Avg | 0.193   | 0.276   | 0.461      | 0.294      | 0.265      | 0.283      | 0.261    | 0.283    |

---

## APPENDIX C: Hyperparameter Sensitivity

Evaluated with T = 96, S = 96.

| Hyperparameter     | Observation |
|--------------------|-------------|
| Learning rate      | Most sensitive for large-variate datasets (ECL, Traffic). Optimal ~1e-4 for Traffic, ~5e-4 for ECL |
| Block number L     | Not monotonically better with more blocks; L=2–3 generally optimal |
| Hidden dimension D | Not monotonically better; D=256–512 generally optimal |

---

## APPENDIX D: Model Efficiency

Comparison at input-96-predict-96. Metrics: MSE, training time (ms/iter), memory footprint (GB).

### Traffic (862 variates)

| Model                    | MSE   | Memory  | Time (ms/iter) |
|--------------------------|-------|---------|---------------|
| Crossformer              | ~0.55 | 9.74 GB | 702           |
| PatchTST                 | ~0.48 | 8.58 GB | 635           |
| TiDE                     | ~0.76 | 2.72 GB | 130           |
| Transformer              | ~0.66 | 1.16 GB | 145           |
| iFlowformer              | ~0.52 | 1.66 GB | 91            |
| **iTransformer**         | ~0.43 | 7.50 GB | 265           |
| **iTransformer (Eff.)**  | ~0.44 | 1.28 GB | 91            |
| DLinear                  | ~0.63 | 0.91 GB | 60            |

### Weather (21 variates)

| Model                    | MSE   | Memory  | Time (ms/iter) |
|--------------------------|-------|---------|---------------|
| Crossformer              | ~0.26 | 1.18 GB | 110           |
| PatchTST                 | ~0.18 | 1.09 GB | 31            |
| TiDE                     | ~0.20 | 0.90 GB | 28            |
| Transformer              | ~0.39 | 1.09 GB | 85            |
| iFlowformer              | ~0.18 | 0.89 GB | 30            |
| **iTransformer**         | ~0.17 | 0.88 GB | 30            |
| **iTransformer (Eff.)**  | ~0.18 | 0.87 GB | 29            |
| DLinear                  | ~0.20 | 0.83 GB | 28            |

Note: iTransformer's O(N²) attention uses N = #variates as tokens vs N = #timesteps for vanilla Transformer. At 862 variates, this increases cost; efficient attention (linear complexity) resolves this.

---

## APPENDIX F: Full Results

### F.1 Full PEMS Results

Input length = 96. Avg = average over all prediction lengths {12, 24, 48, 96}.

**PEMS03 (358 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 12  | **0.071/0.174** | 0.126/0.236 | 0.099/0.216 | 0.090/0.203 | 0.178/0.305 | 0.085/0.192 | 0.122/0.243 | **0.066/0.172** | 0.126/0.251 | 0.081/0.188 | 0.272/0.385 |
| 24  | 0.093/0.201 | 0.246/0.334 | 0.142/0.259 | 0.121/0.240 | 0.257/0.371 | 0.118/0.223 | 0.201/0.317 | **0.085/0.198** | 0.149/0.275 | 0.105/0.214 | 0.334/0.440 |
| 48  | **0.125/0.236** | 0.551/0.529 | 0.211/0.319 | 0.202/0.317 | 0.379/0.463 | 0.155/0.260 | 0.333/0.425 | 0.127/0.238 | 0.227/0.348 | 0.154/0.257 | 1.032/0.782 |
| 96  | **0.164/0.275** | 1.057/0.787 | 0.269/0.370 | 0.262/0.367 | 0.490/0.539 | 0.228/0.317 | 0.457/0.515 | 0.178/0.287 | 0.348/0.434 | 0.247/0.336 | 1.031/0.796 |
| Avg | **0.113/0.221** | 0.495/0.472 | 0.180/0.291 | 0.169/0.281 | 0.326/0.419 | 0.147/0.248 | 0.278/0.375 | 0.114/0.224 | 0.213/0.327 | 0.147/0.249 | 0.667/0.601 |

**PEMS04 (307 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 12  | **0.078/0.183** | 0.138/0.252 | 0.105/0.224 | 0.098/0.218 | 0.219/0.340 | 0.087/0.195 | 0.148/0.272 | **0.073/0.177** | 0.138/0.262 | 0.088/0.196 | 0.424/0.491 |
| 24  | **0.095/0.205** | 0.258/0.348 | 0.153/0.275 | 0.131/0.256 | 0.292/0.398 | 0.103/0.215 | 0.224/0.340 | **0.084/0.193** | 0.177/0.293 | 0.104/0.216 | 0.459/0.509 |
| 48  | **0.120/0.233** | 0.572/0.544 | 0.229/0.339 | 0.205/0.326 | 0.409/0.478 | 0.136/0.250 | 0.355/0.437 | **0.099/0.211** | 0.270/0.368 | 0.137/0.251 | 0.646/0.610 |
| 96  | **0.150/0.262** | 1.137/0.820 | 0.291/0.389 | 0.402/0.457 | 0.492/0.532 | 0.190/0.303 | 0.452/0.504 | **0.114/0.227** | 0.341/0.427 | 0.186/0.297 | 0.912/0.748 |
| Avg | **0.111/0.221** | 0.526/0.491 | 0.195/0.307 | 0.209/0.314 | 0.353/0.437 | 0.129/0.241 | 0.295/0.388 | **0.092/0.202** | 0.231/0.337 | 0.127/0.240 | 0.610/0.590 |

**PEMS07 (883 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 12  | **0.067/0.165** | 0.118/0.235 | 0.095/0.207 | 0.094/0.200 | 0.173/0.304 | 0.082/0.181 | 0.115/0.242 | 0.068/0.171 | 0.109/0.225 | 0.083/0.185 | 0.199/0.336 |
| 24  | **0.088/0.190** | 0.242/0.341 | 0.150/0.262 | 0.139/0.247 | 0.271/0.383 | 0.101/0.204 | 0.210/0.329 | 0.119/0.225 | 0.125/0.244 | 0.102/0.207 | 0.323/0.420 |
| 48  | **0.110/0.215** | 0.562/0.541 | 0.253/0.340 | 0.311/0.369 | 0.446/0.495 | 0.134/0.238 | 0.398/0.458 | 0.149/0.237 | 0.165/0.288 | 0.136/0.240 | 0.390/0.470 |
| 96  | **0.139/0.245** | 1.096/0.795 | 0.346/0.404 | 0.396/0.442 | 0.628/0.577 | 0.181/0.279 | 0.594/0.553 | 0.141/0.234 | 0.262/0.376 | 0.187/0.287 | 0.554/0.578 |
| Avg | **0.101/0.204** | 0.504/0.478 | 0.211/0.303 | 0.235/0.315 | 0.380/0.440 | 0.124/0.225 | 0.329/0.395 | 0.119/0.234 | 0.165/0.283 | 0.127/0.230 | 0.367/0.451 |

**PEMS08 (170 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 12  | **0.079/0.182** | 0.133/0.247 | 0.168/0.232 | 0.165/0.214 | 0.227/0.343 | 0.112/0.212 | 0.154/0.276 | 0.087/0.184 | 0.173/0.273 | 0.109/0.207 | 0.436/0.485 |
| 24  | **0.115/0.219** | 0.249/0.343 | 0.224/0.281 | 0.215/0.260 | 0.318/0.409 | 0.141/0.238 | 0.248/0.353 | 0.122/0.221 | 0.210/0.301 | 0.140/0.236 | 0.467/0.502 |
| 48  | **0.186/0.235** | 0.569/0.544 | 0.321/0.354 | 0.315/0.355 | 0.497/0.510 | 0.198/0.283 | 0.440/0.470 | 0.189/0.270 | 0.320/0.394 | 0.211/0.294 | 0.966/0.733 |
| 96  | **0.221/0.267** | 1.166/0.814 | 0.408/0.417 | 0.377/0.397 | 0.721/0.592 | 0.320/0.351 | 0.674/0.565 | 0.236/0.300 | 0.442/0.465 | 0.345/0.367 | 1.385/0.915 |
| Avg | **0.150/0.226** | 0.529/0.487 | 0.280/0.321 | 0.268/0.307 | 0.441/0.464 | 0.193/0.271 | 0.379/0.416 | 0.158/0.244 | 0.286/0.358 | 0.201/0.276 | 0.814/0.659 |

1st place count (PEMS, MSE/MAE): iTransformer **13/13**, SCINet 7/7, all others 0.

---

### F.2 Full Long-Term Forecasting Results

Input length T = 96. Format: MSE/MAE.

**ETTm1**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | 0.334/0.368 | 0.355/0.376 | **0.329/0.367** | 0.404/0.426 | 0.364/0.387 | 0.338/0.375 | 0.345/0.372 | 0.418/0.438 | 0.379/0.419 | 0.386/0.398 | 0.505/0.475 |
| 192 | 0.377/0.391 | 0.391/0.392 | **0.367/0.385** | 0.450/0.451 | 0.398/0.404 | 0.374/0.387 | 0.380/0.389 | 0.439/0.450 | 0.426/0.441 | 0.459/0.444 | 0.553/0.496 |
| 336 | 0.426/0.420 | 0.424/0.415 | **0.399/0.410** | 0.532/0.515 | 0.428/0.425 | 0.410/0.411 | 0.413/0.413 | 0.490/0.485 | 0.445/0.459 | 0.495/0.464 | 0.621/0.537 |
| 720 | 0.491/0.459 | 0.487/0.450 | **0.454/0.439** | 0.666/0.589 | 0.487/0.461 | 0.478/0.450 | 0.474/0.453 | 0.595/0.550 | 0.543/0.490 | 0.585/0.516 | 0.671/0.561 |
| Avg | 0.407/0.410 | 0.414/0.407 | **0.387/0.400** | 0.513/0.496 | 0.419/0.419 | 0.400/0.406 | 0.403/0.407 | 0.485/0.481 | 0.448/0.452 | 0.481/0.456 | 0.588/0.517 |

**ETTm2**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | 0.180/0.264 | 0.182/0.265 | **0.175/0.259** | 0.287/0.366 | 0.207/0.305 | 0.187/0.267 | 0.193/0.292 | 0.286/0.377 | 0.203/0.287 | 0.192/0.274 | 0.255/0.339 |
| 192 | 0.250/0.309 | 0.246/0.304 | **0.241/0.302** | 0.414/0.492 | 0.290/0.364 | 0.249/0.309 | 0.284/0.362 | 0.399/0.445 | 0.269/0.328 | 0.280/0.339 | 0.281/0.340 |
| 336 | 0.311/0.348 | 0.307/0.342 | **0.305/0.343** | 0.597/0.542 | 0.377/0.422 | 0.321/0.351 | 0.369/0.427 | 0.637/0.591 | 0.325/0.366 | 0.334/0.361 | 0.339/0.372 |
| 720 | 0.412/0.407 | 0.407/0.398 | **0.402/0.400** | 1.730/1.042 | 0.558/0.524 | 0.408/0.403 | 0.554/0.522 | 0.960/0.735 | 0.421/0.415 | 0.417/0.413 | 0.433/0.432 |
| Avg | 0.288/0.332 | 0.286/0.327 | **0.281/0.326** | 0.757/0.610 | 0.358/0.404 | 0.291/0.333 | 0.350/0.401 | 0.571/0.537 | 0.305/0.349 | 0.306/0.347 | 0.327/0.371 |

**ETTh1**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | **0.386/0.405** | **0.386/0.395** | 0.414/0.419 | 0.423/0.448 | 0.479/0.464 | 0.384/0.402 | 0.386/0.400 | 0.654/0.599 | 0.376/0.419 | 0.513/0.491 | 0.449/0.459 |
| 192 | 0.441/0.436 | **0.437/0.424** | 0.460/0.445 | 0.471/0.474 | 0.525/0.492 | 0.436/0.429 | 0.437/0.432 | 0.719/0.631 | 0.420/0.448 | 0.534/0.504 | 0.500/0.482 |
| 336 | 0.487/0.458 | **0.479/0.446** | 0.501/0.466 | 0.570/0.546 | 0.565/0.515 | 0.491/0.469 | 0.481/0.459 | 0.778/0.659 | 0.459/0.465 | 0.588/0.535 | 0.521/0.496 |
| 720 | 0.503/0.491 | **0.481/0.470** | 0.500/0.488 | 0.653/0.621 | 0.594/0.558 | 0.521/0.500 | 0.519/0.516 | 0.836/0.699 | 0.506/0.507 | 0.643/0.616 | 0.514/0.512 |
| Avg | 0.454/0.447 | **0.446/0.434** | 0.469/0.454 | 0.529/0.522 | 0.541/0.507 | 0.458/0.450 | 0.456/0.452 | 0.747/0.647 | 0.440/0.460 | 0.570/0.537 | 0.496/0.487 |

**ETTh2**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | 0.297/0.349 | **0.288/0.338** | 0.302/0.348 | 0.745/0.584 | 0.400/0.440 | 0.340/0.374 | 0.333/0.387 | 0.707/0.621 | 0.358/0.397 | 0.476/0.458 | 0.346/0.388 |
| 192 | 0.380/0.400 | **0.374/0.390** | 0.388/0.400 | 0.877/0.656 | 0.528/0.509 | 0.402/0.414 | 0.477/0.476 | 0.860/0.689 | 0.429/0.439 | 0.512/0.493 | 0.456/0.452 |
| 336 | 0.428/0.432 | **0.415/0.426** | 0.426/0.433 | 1.043/0.731 | 0.643/0.571 | 0.452/0.452 | 0.594/0.541 | 1.000/0.744 | 0.496/0.487 | 0.552/0.551 | 0.482/0.486 |
| 720 | **0.427/0.445** | 0.420/0.440 | 0.431/0.446 | 1.104/0.763 | 0.874/0.679 | 0.462/0.468 | 0.831/0.657 | 1.249/0.838 | 0.463/0.474 | 0.562/0.560 | 0.515/0.511 |
| Avg | 0.383/0.407 | **0.374/0.398** | 0.387/0.407 | 0.942/0.684 | 0.611/0.550 | 0.414/0.427 | 0.559/0.515 | 0.954/0.723 | 0.437/0.449 | 0.526/0.516 | 0.450/0.459 |

**ECL (321 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | **0.148/0.240** | 0.201/0.281 | 0.181/0.270 | 0.219/0.314 | 0.237/0.329 | 0.168/0.272 | 0.197/0.282 | 0.247/0.345 | 0.193/0.308 | 0.169/0.273 | 0.201/0.317 |
| 192 | **0.162/0.253** | 0.201/0.283 | 0.188/0.274 | 0.231/0.322 | 0.236/0.330 | 0.184/0.289 | 0.196/0.285 | 0.257/0.355 | 0.201/0.315 | 0.182/0.286 | 0.222/0.334 |
| 336 | **0.178/0.269** | 0.215/0.298 | 0.204/0.293 | 0.246/0.337 | 0.249/0.344 | 0.198/0.300 | 0.209/0.301 | 0.269/0.369 | 0.214/0.329 | 0.200/0.304 | 0.231/0.338 |
| 720 | **0.225/0.317** | 0.257/0.331 | 0.246/0.324 | 0.280/0.363 | 0.284/0.373 | 0.220/0.320 | 0.245/0.333 | 0.299/0.390 | 0.246/0.355 | 0.222/0.321 | 0.254/0.361 |
| Avg | **0.178/0.270** | 0.219/0.298 | 0.205/0.290 | 0.244/0.334 | 0.251/0.344 | 0.192/0.295 | 0.212/0.300 | 0.268/0.365 | 0.214/0.327 | 0.193/0.296 | 0.227/0.338 |

**Traffic (862 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | **0.395/0.268** | 0.649/0.389 | 0.462/0.295 | 0.522/0.290 | 0.805/0.493 | 0.593/0.321 | 0.650/0.396 | 0.788/0.499 | 0.587/0.366 | 0.612/0.338 | 0.613/0.388 |
| 192 | **0.417/0.276** | 0.601/0.366 | 0.466/0.296 | 0.530/0.293 | 0.756/0.474 | 0.617/0.336 | 0.598/0.370 | 0.789/0.505 | 0.604/0.373 | 0.613/0.340 | 0.616/0.382 |
| 336 | **0.433/0.283** | 0.609/0.369 | 0.482/0.304 | 0.558/0.305 | 0.762/0.477 | 0.629/0.336 | 0.605/0.373 | 0.797/0.508 | 0.621/0.383 | 0.618/0.328 | 0.622/0.337 |
| 720 | **0.467/0.302** | 0.647/0.387 | 0.514/0.322 | 0.589/0.328 | 0.719/0.449 | 0.640/0.350 | 0.645/0.394 | 0.841/0.523 | 0.626/0.382 | 0.653/0.355 | 0.660/0.408 |
| Avg | **0.428/0.282** | 0.626/0.378 | 0.481/0.304 | 0.550/0.304 | 0.760/0.473 | 0.620/0.336 | 0.625/0.383 | 0.804/0.509 | 0.610/0.376 | 0.624/0.340 | 0.628/0.379 |

**Weather (21 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | 0.174/0.214 | 0.192/0.232 | 0.177/0.218 | **0.158/0.230** | 0.202/0.261 | 0.172/0.220 | 0.196/0.255 | 0.221/0.306 | 0.217/0.296 | 0.173/0.223 | 0.266/0.336 |
| 192 | **0.221/0.254** | 0.240/0.271 | 0.225/0.259 | 0.206/0.277 | 0.242/0.298 | 0.219/0.261 | 0.237/0.296 | 0.261/0.340 | 0.276/0.336 | 0.245/0.285 | 0.307/0.367 |
| 336 | 0.278/0.296 | 0.292/0.307 | 0.278/0.297 | **0.272/0.335** | 0.287/0.335 | 0.280/0.306 | 0.283/0.335 | 0.309/0.378 | 0.339/0.380 | 0.321/0.338 | 0.359/0.395 |
| 720 | **0.358/0.347** | 0.364/0.353 | 0.354/0.348 | 0.398/0.418 | 0.351/0.386 | 0.365/0.359 | 0.345/0.381 | 0.377/0.427 | 0.403/0.428 | 0.414/0.410 | 0.419/0.428 |
| Avg | **0.258/0.278** | 0.272/0.291 | 0.259/0.281 | 0.259/0.315 | 0.271/0.320 | 0.259/0.287 | 0.265/0.317 | 0.292/0.363 | 0.309/0.360 | 0.288/0.314 | 0.338/0.382 |

**Solar-Energy (137 variates)**

| H   | iTransformer | RLinear | PatchTST | Crossformer | TiDE  | TimesNet | DLinear | SCINet | FEDformer | Stationary | Autoformer |
|-----|-------------|---------|---------|------------|-------|---------|--------|-------|----------|-----------|-----------|
| 96  | **0.203/0.237** | 0.322/0.339 | 0.234/0.286 | 0.310/0.331 | 0.312/0.399 | 0.250/0.292 | 0.290/0.378 | 0.237/0.344 | 0.242/0.342 | 0.215/0.249 | 0.884/0.711 |
| 192 | **0.233/0.261** | 0.359/0.356 | 0.267/0.310 | 0.734/0.725 | 0.339/0.416 | 0.296/0.318 | 0.320/0.398 | 0.280/0.380 | 0.285/0.380 | 0.254/0.272 | 0.834/0.692 |
| 336 | **0.248/0.273** | 0.397/0.369 | 0.290/0.315 | 0.750/0.735 | 0.368/0.430 | 0.319/0.330 | 0.353/0.415 | 0.304/0.389 | 0.282/0.376 | 0.290/0.296 | 0.941/0.723 |
| 720 | **0.249/0.275** | 0.397/0.356 | 0.289/0.317 | 0.769/0.765 | 0.370/0.425 | 0.338/0.337 | 0.356/0.413 | 0.308/0.388 | 0.357/0.427 | 0.285/0.295 | 0.882/0.717 |
| Avg | **0.233/0.262** | 0.369/0.356 | 0.270/0.307 | 0.641/0.639 | 0.347/0.417 | 0.301/0.319 | 0.330/0.401 | 0.282/0.375 | 0.291/0.381 | 0.261/0.381 | 0.885/0.711 |

1st place count (long-term, MSE/MAE): iTransformer **16/22**, PatchTST 12/11, RLinear 6/12, Crossformer 3/0, others ≤4.

---

### F.3 Robustness (5 random seeds)

| Dataset | H   | MSE             | MAE             |
|---------|-----|-----------------|-----------------|
| ECL     | 96  | 0.148 ± 0.000   | 0.240 ± 0.000   |
|         | 192 | 0.162 ± 0.002   | 0.253 ± 0.002   |
|         | 336 | 0.178 ± 0.000   | 0.269 ± 0.001   |
|         | 720 | 0.225 ± 0.006   | 0.317 ± 0.007   |
| ETTh2   | 96  | 0.297 ± 0.002   | 0.349 ± 0.001   |
|         | 192 | 0.380 ± 0.001   | 0.400 ± 0.001   |
|         | 336 | 0.428 ± 0.002   | 0.432 ± 0.001   |
|         | 720 | 0.427 ± 0.004   | 0.445 ± 0.002   |
| Exchange| 96  | 0.088 ± 0.001   | 0.209 ± 0.001   |
|         | 192 | 0.181 ± 0.001   | 0.304 ± 0.001   |
|         | 336 | 0.334 ± 0.001   | 0.419 ± 0.001   |
|         | 720 | 0.829 ± 0.012   | 0.691 ± 0.005   |
| Solar   | 96  | 0.203 ± 0.002   | 0.237 ± 0.002   |
|         | 192 | 0.233 ± 0.002   | 0.261 ± 0.001   |
|         | 336 | 0.248 ± 0.000   | 0.273 ± 0.000   |
|         | 720 | 0.249 ± 0.001   | 0.275 ± 0.000   |
| Traffic | 96  | 0.395 ± 0.001   | 0.268 ± 0.001   |
|         | 192 | 0.417 ± 0.002   | 0.276 ± 0.001   |
|         | 336 | 0.433 ± 0.004   | 0.283 ± 0.000   |
|         | 720 | 0.467 ± 0.003   | 0.302 ± 0.000   |
| Weather | 96  | 0.174 ± 0.000   | 0.214 ± 0.000   |
|         | 192 | 0.221 ± 0.002   | 0.254 ± 0.001   |
|         | 336 | 0.278 ± 0.002   | 0.296 ± 0.001   |
|         | 720 | 0.358 ± 0.000   | 0.349 ± 0.000   |

---

## APPENDIX G: Discussions

### G.1 vs. Channel Independence (CI)

- CI (e.g., PatchTST) trains a shared backbone per-variate; generalizes well due to sample scarcity in current benchmarks
- CI is time-consuming at inference (predicts each variate sequentially) and cannot explicitly model multivariate correlations
- iTransformer: independent variate tokenization + attention for correlating → handles both independence and correlation explicitly, with parallel inference

### G.2 vs. Linear Forecasters

- Linear forecasters excel at temporal dependencies via dense time-point weighting
- iTransformer is particularly suited to **high-dimensional multivariate series** where correlations matter
- Under univariate scenarios, iTransformer degrades to a stackable linear forecaster (attention collapses)

### G.3 Tokenization Comparison

| Paradigm         | Token unit                    | Attention captures    | FFN captures          |
|------------------|-------------------------------|----------------------|-----------------------|
| Transformer      | Multivariate point at time t  | Temporal dependencies | Multivariate repr.    |
| PatchTST         | Patch of T/P consecutive steps (per variate) | Temporal (patch-level) | Series repr. |
| Crossformer      | Cross-dimension patches       | Cross-time + cross-variate | —               |
| **iTransformer** | **Entire series of one variate** | **Multivariate correlations** | **Series representations** |