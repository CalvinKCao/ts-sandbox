# MMPD: Multi-Mode Patch Diffusion Loss — Methods & Results Reference

> Paraphrased reproduction guide. All equations are as stated in the paper.
> Zhang et al., ICLR 2026. Code: https://github.com/Thinklab-SJTU/MMPD

---

## 1. Problem Framing: Loss as a Probabilistic Choice

Given past series $\mathbf{x} \in \mathbb{R}^T$, the goal is to model $p(\mathbf{y}|\mathbf{x})$ where $\mathbf{y} \in \mathbb{R}^\tau$ is the future horizon.

**Why MSE is limiting.** Minimizing MSE is equivalent to maximum likelihood under an independent Gaussian with fixed variance:

$$\max_\theta \mathbb{E}_{q(\mathbf{x},\mathbf{y})}\left[\log p_\theta(\mathbf{y}|\mathbf{x})\right] \equiv \min_\theta \mathbb{E}\left[\frac{\tau}{2\sigma^2} \text{MSE}(f_\theta(\mathbf{x}), \mathbf{y})\right]$$

The constant $\frac{\tau}{2\sigma^2}$ is absorbed by the optimizer step size, so MSE training implicitly enforces:
- **Single mode** — one Gaussian cannot represent multiple futures
- **Independent steps** — no cross-timestep correlation in uncertainty
- **Fixed variance** — uncertainty cannot grow/shrink over the horizon
- **Symmetry** — Gaussian is symmetric; real distributions (e.g. rainfall) often are not

MAE has the same structural problem (assumes a Laplace distribution).

**Decoupling backbone and projector.** The forecasting network is split as:

$$f_\theta(\mathbf{x}) = g_\phi(\mathbf{H}), \quad \mathbf{H} = h_\psi(\mathbf{x}), \quad \theta = \{\phi, \psi\}$$

- $h_\psi$: backbone — extracts latent representations, holds most parameters
- $g_\phi$: projector — maps representations to output space, lightweight

From the backbone's perspective, the projector is part of the loss, giving a **composite trainable loss**:

$$\min_{\phi,\psi} \text{Loss}_\phi(\mathbf{H}, \mathbf{y}), \quad \mathbf{H} = h_\psi(\mathbf{x})$$

MSE fits this framework as $\text{MSE}^\phi(\mathbf{H}, \mathbf{y}) = \frac{1}{\tau}\|\mathbf{y} - g_\phi(\mathbf{H})\|_2^2$.

---

## 2. Diffusion Model Preliminaries

Given training samples and conditions $(y_0, c) \sim q(y_0, c)$:

**Forward process** (adds noise over K steps):

$$q(\mathbf{y}^k | \mathbf{y}^{k-1}, c) = \mathcal{N}(\mathbf{y}^k; \sqrt{1-\beta_k}\mathbf{y}^{k-1},\ \beta_k \mathbf{I})$$

where $\{\beta_k \in (0,1)\}_{k=1}^K$ is the variance schedule.

**Reverse process** (learned denoising):

$$p_\phi(\mathbf{y}^{k-1}|\mathbf{y}^k, c) = \mathcal{N}(\mathbf{y}^{k-1};\ \mu_\phi(\mathbf{y}^k, c, k),\ \sigma_k^2 \mathbf{I})$$

**Training objective** (noise prediction):

$$\mathcal{L} = \mathbb{E}_{\mathbf{y}^0, c, k, \epsilon}\left\|\epsilon - \epsilon_\phi(\mathbf{y}^k, c, k)\right\|_2^2$$

where $\mathbf{y}^k = \sqrt{\bar{\alpha}_k}\mathbf{y}^0 + \sqrt{1-\bar{\alpha}_k}\epsilon$, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$, $\alpha_k = 1-\beta_k$, $\bar{\alpha}_k = \prod_{s=1}^k \alpha_s$.

---

## 3. MMPD Loss

### 3.1 Scope: Patch-Based Backbones

MMPD targets backbones that divide the input series into patches of length $P$, producing $T/P$ input tokens and $l = \tau/P$ future latent tokens $\mathbf{H} = \{h_j\}_{j=1}^l$, one per future patch. This covers most recent supervised and foundation TS models (PatchTST, Crossformer, SegRNN, MaskAE, MOMENT, MOIRAI, etc.).

The MMPD loss treats these future tokens as the **diffusion condition**, constructing:

$$\text{MMPD}^\phi(\mathbf{H}, \mathbf{y}) = \mathbb{E}_{\mathbf{y}^0, k, \epsilon}\left\|\epsilon - \epsilon_\phi(\mathbf{y}^k, \{h_j\}_{j=1}^l, k)\right\|_2^2$$

The denoising network $\epsilon_\phi$ must be lightweight (it is auxiliary to the backbone).

---

### 3.2 Patch Consistent MLP (Denoising Network)

**Problem with independent MLP.** A naive approach denoises each future patch $\mathbf{p}_j^k$ conditioned only on its corresponding token $h_j$. This models only the marginal $p(\mathbf{p}_j|\mathbf{x})$ rather than the joint $p(\mathbf{p}_1, \ldots, \mathbf{p}_l|\mathbf{x})$, producing samples with discontinuous jumps between patches at inference.

**Solution: Patch Consistent MLP.** Each patch's denoising also conditions on $r$ adjacent noisy patches on both sides:

$$\epsilon_\phi(\mathbf{y}^k, \{h_j\}_{j=1}^l, k) = [\hat{\epsilon}_1 \,\cdot\!\cdot\!\cdot\, \hat{\epsilon}_l]$$

$$\hat{\epsilon}_j = \text{AdaLN-MLP}(\mathbf{p}_j^k,\ \mathbf{c}_j^k)$$

$$\mathbf{c}_j^k = \text{token}_j + \text{step}^k + \text{prev}_j^k + \text{next}_j^k$$

where:

| Component | Formula | Role |
|-----------|---------|------|
| $\text{token}_j$ | $W^{(\text{token})} h_j$ | Backbone latent for patch $j$ |
| $\text{step}^k$ | $\text{Emb}^{(\text{step})}(k)$ | Diffusion timestep embedding |
| $\text{prev}_j^k$ | $W^{(\text{prev})}[\mathbf{p}_{j-r}^k \cdot\!\cdot\!\cdot \mathbf{p}_{j-1}^k]$ | Previous $r$ noisy patches |
| $\text{next}_j^k$ | $W^{(\text{next})}[\mathbf{p}_{j+1}^k \cdot\!\cdot\!\cdot \mathbf{p}_{j+r}^k]$ | Next $r$ noisy patches |

- $W^{(\text{token})} \in \mathbb{R}^{d \times d}$; $W^{(\text{prev})}, W^{(\text{next})} \in \mathbb{R}^{d \times rP}$
- Padding used when $j \leq r$ or $j > l - r$
- $r$ is a hyperparameter (default: $r=3$)
- The extra parameters from $W^{(\text{prev})}, W^{(\text{next})}$ are minimal

**AdaLN-MLP block** (from DiT, Peebles & Xie 2023). For each of $S$ blocks:

$$z^{(s)} = z^{(s-1)} + \alpha^{(s)}_\text{gate} \circ \text{MLP}\left(\text{AdaLN}(z^{(s-1)}, \gamma^{(s)}_\text{scale}, \beta^{(s)}_\text{shift})\right)$$

$$\text{AdaLN}(z, \gamma, \beta) = (1+\gamma) \circ \text{LayerNorm}(z) + \beta$$

$$\alpha^{(s)}_\text{gate} = W^{(s)}_\text{gate}\phi(c), \quad \gamma^{(s)}_\text{scale} = W^{(s)}_\text{scale}\phi(c), \quad \beta^{(s)}_\text{shift} = W^{(s)}_\text{shift}\phi(c)$$

Final output:

$$\hat{\epsilon} = W^{(\text{final})}\text{AdaLN}(z^{(S)}, \gamma^{(\text{out})}_\text{scale}, \beta^{(\text{out})}_\text{shift})$$

---

### 3.3 Integration with Deterministic Forecasting

At diffusion step $k^*$ where $\mathbf{y}^{k^*} = \mathbf{0}$ (noise exactly cancels the sample), the noise prediction target reduces to $\epsilon = -\frac{\sqrt{\bar{\alpha}_{k^*}}}{\sqrt{1-\bar{\alpha}_{k^*}}} \mathbf{y}^0$.

The joint training objective is:

$$\mathcal{L} = \lambda \left\|\epsilon - \epsilon_\phi(\mathbf{y}^k, \{h_j\}_{j=1}^l, k)\right\|_2^2 + (1-\lambda) \left\|\frac{\sqrt{\bar{\alpha}_{k^*}}}{\sqrt{1-\bar{\alpha}_{k^*}}}\mathbf{y}^0 + \epsilon_\phi(\mathbf{0}, \{h_j\}_{j=1}^l, k^*)\right\|_2^2$$

**Key settings:**
- $\lambda = 0.99$ by default (0.9 for SegRNN backbone)
- $k^*$ chosen so $\bar{\alpha}_{k^*} \approx 0.5$, making the deterministic coefficient $\approx 1$
- At inference, deterministic prediction = $-\frac{\sqrt{1-\bar{\alpha}_{k^*}}}{\sqrt{\bar{\alpha}_{k^*}}} \epsilon_\phi(\mathbf{0}, \{h_j\}_{j=1}^l, k^*)$ — **single MLP pass, no diffusion**
- This is not a separate module — it reuses $\epsilon_\phi$; the deterministic term is just a special case of the diffusion objective at the anchor

---

## 4. Multi-Mode Inference Algorithm

### 4.1 Motivation

Diffusion generates $N$ samples at inference, but these exhibit multi-mode structure that simple statistics (median, CI) cannot capture. The true distribution is modelled as:

$$q(\mathbf{y}^0|\mathbf{x}) = \sum_{m=1}^M w_m \delta(\mathbf{y}^0 - \mathbf{y}^*_m), \quad \sum_{m=1}^M w_m = 1$$

Forward diffusion transforms this into a GMM at each step $k$:

$$q(\mathbf{y}^k|\mathbf{x}) = \sum_{m=1}^M w_m \mathcal{N}(\mathbf{y}^k;\ \sqrt{\bar{\alpha}_k}\mathbf{y}^*_m,\ (1-\bar{\alpha}_k)\mathbf{I})$$

This motivates fitting a GMM **alongside** the reverse process (not as post-processing on final samples), using the two priors encoded in the equation above:
1. Mixture weights $w_m$ are constant across steps
2. Covariance at step $k$ is $(1-\bar{\alpha}_k)\mathbf{I}$ — known from the forward process

### 4.2 Variational GMM Priors

At each step $k$, the following prior distributions are set:

$$q(\mathbf{Y}^k | \mathbf{Z}^k, \boldsymbol{\mu}^k, \boldsymbol{\Lambda}^k) = \prod_{n=1}^N \prod_{m=1}^M \mathcal{N}(y_n^k;\ \mu_m^k,\ (\Lambda_m^k)^{-1}\mathbf{I})^{z_{nm}^k}$$

$$q(\mathbf{Z}^k | \mathbf{w}^k) = \prod_{n=1}^N \prod_{m=1}^M (w_m^k)^{z_{nm}^k}$$

**Mixture weight prior** (Dirichlet, constant across $k$):
$$q(\mathbf{w}^k) = \text{Dirichlet}(\mathbf{w}^k; \boldsymbol{\pi}), \quad \pi_m = \rho^{m-1}$$

$\rho < 1$ makes higher-indexed modes decay to zero, limiting active mode count.

**Variance prior** (Gamma, evolves with $k$):
$$q(\boldsymbol{\Lambda}^k) = \prod_{m=1}^M \text{Gamma}(\Lambda_m^k;\ u_m^k, v_m^k), \quad u_m^k = u, \quad v_m^k = u \cdot (1 - \bar{\alpha}_k)$$

This encodes the known covariance $(1-\bar{\alpha}_k)\mathbf{I}$ from forward diffusion: $\mathbb{E}[\Lambda_m^k]^{-1} = v_m^k / u_m^k = 1-\bar{\alpha}_k$.

**No prior on $\boldsymbol{\mu}^k$** — these are the unknown mode locations $\mathbf{y}^*_m$.

Hyperparameters: $\rho = 0.5$, $u = 100$ (defaults).

### 4.3 Algorithm

**Initialization:** At step $K$, initialize $\{\mu_m^K\}_{m=1}^M$ and posterior parameters $\{\tilde{u}_m^K, \tilde{v}_m^K, \tilde{\pi}_m^K\}_{m=1}^M$.

**At each step $k = K-1, \ldots, 0$:**

1. Generate samples $\{y_n^k\}_{n=1}^N$ via one reverse diffusion step from $\{y_n^{k+1}\}$

2. **E-step** — update assignment posterior $p(\mathbf{Z}^k)$:

$$\ln \gamma_{nm}^k = -\frac{1}{2}\left[\frac{\tilde{u}_m^{k+1}}{\tilde{v}_m^{k+1}}\|y_n^k - \mu_m^{k+1}\|_2^2 + \tau\ln(2\pi)\right] + \frac{\tau}{2}\left[\psi(\tilde{u}_m^{k+1}) - \ln(\tilde{v}_m^{k+1})\right] + \psi(\tilde{\pi}_m^{k+1}) - \psi\left(\sum_{s=1}^M \tilde{\pi}_s^{k+1}\right)$$

$$\tilde{\gamma}_{nm}^k = \frac{\gamma_{nm}^k}{\sum_{s=1}^M \gamma_{ns}^k}, \quad p(\mathbf{Z}^k) = \prod_{n,m} (\tilde{\gamma}_{nm}^k)^{z_{nm}^k}$$

where $\psi(\cdot)$ is the digamma function.

3. **M-step** — update mode means:

$$\mu_m^k = \frac{1}{\tilde{N}_m^k} \sum_{n=1}^N \tilde{\gamma}_{nm}^k y_n^k, \quad \tilde{N}_m^k = \sum_{n=1}^N \tilde{\gamma}_{nm}^k$$

4. **M-step** — update weight posterior:

$$p(\mathbf{w}^k) = \text{Dirichlet}(\mathbf{w}^k; \tilde{\boldsymbol{\pi}}^k), \quad \tilde{\pi}_m^k = \pi_m + \tilde{N}_m^k$$

5. **M-step** — update variance posterior:

$$p(\boldsymbol{\Lambda}^k) = \prod_m \text{Gamma}(\Lambda_m^k;\ \tilde{u}_m^k, \tilde{v}_m^k)$$

$$\tilde{u}_m^k = u_m^k + \frac{\tau}{2}\tilde{N}_m^k, \quad \tilde{v}_m^k = v_m^k + \frac{1}{2}\sum_{n=1}^N \tilde{\gamma}_{nm}^k \|y_n^k - \mu_m^k\|_2^2$$

**Initialization across steps:** Posteriors from step $k+1$ warm-start step $k$ (consecutive diffusion samples differ little, so this is stable).

**Final output** (at $k=0$):

$$\text{Mode}_m = \{y_n^0 \mid \arg\max_s \tilde{\gamma}_{ns}^0 = m\}$$

$$P(\text{Mode}_m) = |\text{Mode}_m| / N, \quad \text{then compute statistics of each mode}$$

**Inference settings:** $N=100$ samples, $M=10$ max modes (algorithm auto-selects active count), $K_\text{infer}=20$ resampled steps, 10 EM iterations per step.

---

## 5. Adapting Encoder-Only Backbones

Encoder-only backbones (e.g. original PatchTST) flatten encoder outputs and project directly to predictions — they produce no future latent tokens, so MMPD cannot be applied directly.

**Adaptation:** Append $l = \tau/P$ learnable tokens to the end of the input patch sequence. Feed the padded sequence through the same encoder network. The output tokens corresponding to the appended positions serve as future latent tokens $\{h_j\}$ for MMPD.

This converts an encoder-only model into a decoder-only model with no new architecture — same weights, same network.

**Benefit:** The projection layer in the original encoder-only design scales as $O(T \cdot \tau)$ with input/output lengths; the decoder-only adaptation makes it constant $O(\tau/P \cdot d)$.

---

## 6. Adapting Non-Patch-Based Backbones

Non-patch backbones (TSMixer, iTransformer) produce latent $\mathbf{H}$ not naturally structured as per-patch tokens.

**Adaptation:** Insert a single Transformer decoder layer between backbone and MMPD loss:
- $\mathbf{H}$ → keys and values
- $l$ learnable tokens (indicating future patches) → queries
- Decoder output → future latent tokens $\{h_j\}$ for MMPD

---

## 7. Computational Complexity

| Structure / Stage | MSE Loss | TS Diffusion Models | MMPD Loss |
|---|---|---|---|
| MLP Projector | $F_\text{MLP} = O\!\left(\frac{\tau}{P}[(2S+1)d^2 + Pd]\right)$ | N/A | $F_\text{PC-MLP} = O\!\left(\frac{\tau}{P}[(5S+3)d^2 + (2r+2)Pd]\right)$ |
| Training (fwd only) | $F_\text{bkb} + F_\text{MLP}$ | $F_\text{bkb}$ | $F_\text{bkb} + 2F_\text{PC-MLP}$ |
| Deterministic Infer | $F_\text{bkb} + F_\text{MLP}$ | N/A | $F_\text{bkb} + F_\text{PC-MLP}$ |
| Prob/Multi-Mode Infer | N/A | $NKF_\text{bkb}$ | $F_\text{bkb} + NKF_\text{PC-MLP}$ |

Since $F_\text{bkb} \gg F_\text{MLP}, F_\text{PC-MLP}$, training FLOPs of MMPD ≈ MSE. The critical advantage over standalone TS diffusion models: MMPD runs one heavy backbone pass then $K$ lightweight MLP passes; diffusion models run $K$ full backbone passes.

**Measured (WTH dataset, $T=336$, $\tau=192$, batch=32, $N=100$, $K=20$):**

| Stage | MSE Loss Memory | MSE Time | Diffusion-TS Memory | Diffusion-TS Time | MMPD Memory | MMPD Time |
|---|---|---|---|---|---|---|
| Training | 2.599 GB | 89.9 ms | 4.358 GB | 676.4 ms | 2.930 GB | 106.3 ms |
| Deterministic Infer | 0.031 GB | 2.3 ms | N/A | N/A | 0.034 GB | 3.1 ms |
| Prob/Multi-Mode Infer | N/A | N/A | 11.245 GB | 28,495.1 ms | 0.505 GB | 415.8 ms |

---

## 8. Experimental Setup

### Datasets

| Dataset | Channels | Timestamps | Split | $T$ | $\tau$ | Domain |
|---|---|---|---|---|---|---|
| ETTh1 | 7 | 14,400 | 0.6/0.2/0.2 | 336 | {96,192,336,720} | Electricity Transformer |
| ETTm1 | 7 | 57,600 | 0.6/0.2/0.2 | 336 | {96,192,336,720} | Electricity Transformer |
| ETTh2 | 7 | 14,400 | 0.6/0.2/0.2 | 336 | {96,192,336,720} | Electricity Transformer |
| ETTm2 | 7 | 57,600 | 0.6/0.2/0.2 | 336 | {96,192,336,720} | Electricity Transformer |
| WTH | 21 | 52,696 | 0.7/0.1/0.2 | 336 | {96,192,336,720} | Weather |
| ECL | 321 | 26,304 | 0.7/0.1/0.2 | 336 | {96,192,336,720} | Electricity Consumption |
| Traffic | 862 | 17,544 | 0.7/0.1/0.2 | 336 | {96,192,336,720} | Road Occupancy |
| Dynamic | 17 | 500,000 | 0.7/0.1/0.2 | 600 | {60,120,180,300} | Complex Dynamical System |

Dynamic uses first 10% of original 5M-timestamp dataset; test set uses non-overlapping windows (step=$\tau$). All others use overlapping windows (step=1).

### Metrics

**Top-K MSE / Top-K MAE** ($K=3$): Select top-$K$ modes by probability; report minimum MSE/MAE among them. Requires full diffusion + GMM inference.

**MSE**: Standard point forecast error. Uses deterministic anchor shortcut.

**CRPS**: Continuous Ranked Probability Score. Approximated as:

$$\text{CRPS}_t \approx \frac{1}{N}\sum_{i=1}^N |\tilde{y}_{i,t} - y_t| - \frac{1}{2N^2}\sum_{i=1}^N\sum_{j=1}^N |\tilde{y}_{i,t} - \tilde{y}_{j,t}|, \quad \text{CRPS} = \frac{1}{\tau}\sum_{t=1}^\tau \text{CRPS}_t$$

All metrics computed per channel then averaged across channels and test instances.

### Implementation

| Hyperparameter | Value |
|---|---|
| Patch size $P$ | 12 (24 for $\tau \in \{336,720\}$ or ECL/Traffic) |
| MLP hidden dim $d_\text{model}$ | 256 |
| Adjacent range $r$ | 3 |
| Noise schedule | Linear (1000 training steps) |
| Inference steps $K_\text{infer}$ | 20 |
| $\lambda$ (det/prob balance) | 0.99 (0.9 for SegRNN) |
| $\bar{\alpha}_{k^*}$ | ≈ 0.5 |
| Samples $N$ | 100 |
| Max modes $M$ | 10 |
| EM iterations per step | 10 |
| $\rho$ (Dirichlet decay) | 0.5 |
| $u$ (Gamma shape) | 100 |
| Optimizer | Adam, lr=1e-4 |
| Max epochs | 20 (early stop: 5 patience) |
| Normalization | Instance normalization |
| Hardware | NVIDIA RTX 3090 (24GB) |

---

## 9. Main Results

### 9.1 Loss Comparison (Decoder-only Transformer backbone, averaged over 4 horizons)

**Top-3 MSE** ↓ (lower=better diverse prediction):

| Loss | ETTh1 | ETTm1 | ETTh2 | ETTm2 | WTH | ECL | Traffic | Dynamic | Rank |
|---|---|---|---|---|---|---|---|---|---|
| MSE | 0.430 | 0.348 | 0.364 | 0.264 | 0.224 | 0.176 | 0.433 | 0.336 | 3.5 |
| MAE | 0.441 | 0.364 | 0.368 | 0.276 | 0.235 | 0.179 | 0.449 | 0.426 | 5.25 |
| Gaussian | 0.439 | 0.361 | 0.379 | 0.280 | 0.255 | 0.165 | 0.419 | 0.343 | 4.75 |
| Student-T | 0.430 | 0.352 | 0.375 | 0.286 | 0.241 | 0.165 | 0.416 | 0.390 | 4.25 |
| Mix | 0.425 | 0.289 | 0.343 | 0.245 | 0.209 | 0.147 | 0.412 | 0.322 | 1.875 |
| **MMPD** | **0.396** | **0.269** | **0.299** | **0.214** | **0.193** | **0.147** | **0.389** | **0.301** | **1** |

**Top-3 MAE** ↓:

| Loss | ETTh1 | ETTm1 | ETTh2 | ETTm2 | WTH | ECL | Traffic | Dynamic | Rank |
|---|---|---|---|---|---|---|---|---|---|
| MSE | 0.440 | 0.381 | 0.398 | 0.320 | 0.262 | 0.278 | 0.326 | 0.311 | 4.875 |
| MAE | 0.437 | 0.375 | 0.392 | 0.321 | 0.263 | 0.272 | 0.310 | 0.295 | 4.125 |
| Gaussian | 0.437 | 0.386 | 0.411 | 0.337 | 0.292 | 0.256 | 0.282 | 0.309 | 5.125 |
| Student-T | 0.428 | 0.371 | 0.398 | 0.333 | 0.263 | 0.250 | 0.261 | 0.292 | 3.375 |
| Mix | 0.426 | 0.338 | 0.387 | 0.308 | 0.242 | 0.240 | 0.261 | 0.246 | 2 |
| **MMPD** | **0.412** | **0.331** | **0.357** | **0.285** | **0.221** | **0.238** | **0.254** | **0.207** | **1** |

**MSE** ↓ (deterministic, single-pass):

| Loss | ETTh1 | ETTm1 | ETTh2 | ETTm2 | WTH | ECL | Traffic | Dynamic | Rank |
|---|---|---|---|---|---|---|---|---|---|
| MSE | 0.425 | 0.350 | 0.376 | 0.270 | 0.227 | 0.160 | 0.399 | 0.345 | 1.875 |
| MAE | 0.432 | 0.355 | 0.366 | 0.274 | 0.233 | 0.164 | 0.417 | 0.426 | 3.5 |
| Gaussian | 0.434 | 0.357 | 0.382 | 0.284 | 0.255 | 0.163 | 0.413 | 0.349 | 4 |
| Student-T | 0.426 | 0.349 | 0.372 | 0.283 | 0.241 | 0.164 | 0.418 | 0.392 | 3.5 |
| Mix | 0.446 | 0.358 | 0.390 | 0.285 | 0.259 | 0.167 | 0.426 | 0.482 | 6 |
| **MMPD** | **0.412** | **0.337** | **0.354** | **0.264** | **0.229** | **0.164** | **0.409** | **0.353** | **1.75** |

**CRPS** ↓ (probabilistic):

| Loss | ETTh1 | ETTm1 | ETTh2 | ETTm2 | WTH | ECL | Traffic | Dynamic | Rank |
|---|---|---|---|---|---|---|---|---|---|
| MSE | 0.337 | 0.307 | 0.308 | 0.247 | 0.218 | 0.270 | 0.343 | 0.257 | 4.25 |
| MAE | 0.346 | 0.313 | 0.299 | 0.247 | 0.220 | 0.288 | 0.362 | 0.275 | 4.625 |
| Gaussian | 0.317 | 0.282 | 0.315 | 0.256 | 0.228 | 0.190 | 0.217 | 0.233 | 4.375 |
| Student-T | 0.310 | 0.271 | 0.300 | 0.250 | 0.201 | 0.187 | 0.204 | 0.224 | 2.25 |
| Mix | 0.316 | 0.269 | 0.310 | 0.247 | 0.209 | Inf | 0.205 | 0.224 | 3 |
| **MMPD** | **0.318** | **0.270** | **0.301** | **0.243** | **0.199** | **0.191** | **0.202** | **0.203** | **2** |

---

### 9.2 Backbone Generality (Top-3 MSE, averaged over 4 horizons, #1st = wins across 8 datasets)

| Backbone | Loss | ETTh1 | ETTm1 | ETTh2 | ETTm2 | WTH | ECL | Traffic | Dynamic | #1st |
|---|---|---|---|---|---|---|---|---|---|---|
| Crossformer | MSE | 0.443 | 0.378 | 0.372 | 0.266 | 0.223 | 0.184 | 0.451 | 0.331 | 0 |
| Crossformer | Mix | 0.433 | 0.330 | 0.359 | 0.236 | 0.200 | 0.160 | 0.424 | 0.307 | 0 |
| Crossformer | **MMPD** | **0.381** | **0.310** | **0.315** | **0.228** | **0.197** | **0.152** | **0.404** | **0.295** | **8** |
| SegRNN | MSE | 0.440 | 0.385 | 0.365 | 0.273 | 0.222 | 0.183 | 0.464 | 0.333 | 0 |
| SegRNN | Mix | 0.435 | 0.335 | 0.341 | 0.248 | 0.214 | 0.155 | 0.439 | 0.330 | 0 |
| SegRNN | **MMPD** | **0.402** | **0.321** | **0.321** | **0.233** | **0.201** | **0.150** | **0.418** | **0.295** | **8** |
| MaskAE | MSE | 0.438 | 0.354 | 0.355 | 0.280 | 0.226 | 0.178 | 0.436 | 0.339 | 0 |
| MaskAE | Mix | 0.415 | 0.314 | 0.340 | 0.253 | 0.203 | 0.150 | 0.416 | 0.321 | 0 |
| MaskAE | **MMPD** | **0.399** | **0.280** | **0.311** | **0.247** | **0.197** | **0.144** | **0.387** | **0.296** | **8** |

---

### 9.3 Comparison vs. Standalone TS Diffusion Models (ETTh1, $T=96$, $\tau=24$)

| Method | Top-3 MSE | Top-3 MAE | MSE | CRPS | Inference Time (s) |
|---|---|---|---|---|---|
| CSDI | 0.225 | 0.304 | 0.339 | 0.265 | 1.014 |
| TSDiff | 0.275 | 0.336 | 0.345 | 0.292 | 0.419 |
| MG-TSD | 0.287 | 0.331 | 0.340 | 0.306 | 0.217 |
| Diffusion-TS | 0.282 | 0.324 | 0.351 | 0.294 | 0.520 |
| D³U | 0.244 | 0.344 | 0.338 | 0.285 | 0.453 |
| **Decoder+MMPD** | **0.186** | **0.280** | **0.298** | **0.254** | **0.075** |

---

## 10. Ablation Results

### Patch Consistent MLP: Effect of Adjacent Range $r$

Tested on Dynamic ($\tau=180$). At $r=0$ (independent MLP), Top-3 MSE exceeds MSE loss — independent MLP fails entirely at diverse prediction. Even $r=1$ gives large gains; $r>1$ gives smaller marginal improvements.

### Balancing Weight $\lambda$

At $\lambda=1.0$ (diffusion only): deterministic MSE is poor. At $\lambda=0.999$: MSE drops sharply, other metrics unharmed. Interestingly, decreasing $\lambda$ from 1.0 to 0.9 also improves multi-mode and probabilistic metrics, suggesting a collaborative training effect. MSE stabilizes for $\lambda \in [0.9, 0.999]$.

### Multi-Mode Inference Ablation (Top-3 MSE / Top-3 MAE on Dynamic)

| Method | Top-3 MSE | Top-3 MAE |
|---|---|---|
| Random assignment | 0.365 | 0.250 |
| Post-KMeans | 0.324 | 0.217 |
| Post-Spectral | 0.328 | 0.220 |
| Post-GMM (no evolving priors) | 0.310 | 0.212 |
| **MMPD (evolving variational GMM)** | **0.301** | **0.207** |

Post-GMM applies the same GMM formulation but only to final samples $\{y_n^0\}$ without evolving priors. MMPD outperforms it because distributing EM across diffusion steps provides better initialization and automatic mode count selection.

### Noise Schedule (ETTh1, $T=336$, $\tau=96$)

| Schedule | Top-3 MSE | Top-3 MAE | MSE | CRPS |
|---|---|---|---|---|
| Quadratic | 0.314 | 0.362 | 0.379 | 0.286 |
| Cosine | 0.317 | 0.364 | 0.382 | 0.289 |
| Linear (default) | 0.329 | 0.371 | 0.375 | 0.289 |

Advanced schedules (Quadratic, Cosine) improve Top-3 metrics; TS-specific schedules are a future direction.

### Training Steps $K_\text{train}$

More training steps consistently improve all metrics. Default $K_\text{train}=1000$ balances quality and cost. The improvement comes at no inference cost (inference steps are resampled independently to $K_\text{infer}=20$).

### Maximum Modes $M$

Results are nearly identical for $M \in \{5, 10, 15, 20, 25\}$ because the variational Dirichlet prior automatically deactivates redundant modes. Example: with $M=10$, only ~3 modes are typically activated after inference.

---

## 11. Non-Patch and Multi-Task Extensions

### Non-Patch Backbones (ETTh1/ETTm1/WTH, $\tau=192$)

| Backbone | Loss | Top-3 MSE (avg) | Top-3 MAE (avg) | MSE (avg) | CRPS (avg) |
|---|---|---|---|---|---|
| TSMixer | MSE | 0.270 | 0.322 | 0.282 | 0.260 |
| TSMixer | MMPD | **0.238** | **0.289** | **0.277** | **0.231** |
| iTransformer | MSE | 0.278 | 0.325 | 0.292 | 0.263 |
| iTransformer | MMPD | **0.257** | **0.300** | **0.297** | **0.249** |

### Multi-Task (UNITS backbone, 20 tasks): Winning Counts

| Metric | MSE Loss wins | MMPD wins |
|---|---|---|
| MSE | 5/20 | 15/20 |
| MAE | 1/20 | 19/20 |
| Top-3 MSE | 2/20 | 18/20 |
| Top-3 MAE | 1/20 | 19/20 |
| CRPS | 1/20 | 19/20 |

### Few-Shot (5% data, prompt tuning): Winning Counts

| Metric | MSE Loss wins | MMPD wins |
|---|---|---|
| Top-3 MSE | 0/5 | 5/5 |
| Top-3 MAE | 0/5 | 5/5 |
| CRPS | 0/5 | 5/5 |

### Zero-Shot: Winning Counts

| Metric | MSE Loss wins | MMPD wins |
|---|---|---|
| Top-3 MSE | 0/3 | 3/3 |
| Top-3 MAE | 0/3 | 3/3 |
| CRPS | 0/3 | 3/3 |
