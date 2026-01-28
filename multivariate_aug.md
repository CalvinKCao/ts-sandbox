# What the augmentation does — short summary

1. For each training sample (a target series `y` of length `T+h`), sample up to `k` covariate series `x_i` drawn either from your corpus or from a **synthetic covariate generator**.
2. For each covariate `x_i`, sample an **impact function** `f_i` from a simple, constrained function family (piecewise linear with sparse linear-lag terms and optional bias + noise). The impact function maps `(y, x_i)` → a time series of the same output length (`T+h`) but is nonzero only on a selected set of time steps `S(x,y)` (domain selection via quantile-thresholding).
3. Build an **augmented target** `y_aug = y + sum_i f_i(y, x_i)` (elementwise, for t=1..T+h). Return training sample `(y_aug, {x_i})`. This forces covariates to be predictive in local context so the model can learn in-context covariate→target relationships. (Algorithm 1) 

---

# Algorithmic details (ready-to-implement)

### Algorithm 1 — Informative Covariate Augmentation (implementation pseudocode)

```python
def informative_covariate_augmentation(y, T, h, corpus, synth_generator, F_sampler, hyperparams):
    # y: array length T+h (target full context + horizon)
    # corpus: collection of candidate covariate series (each length >= T+h)
    # synth_generator: function generate_synthetic_covariate(T) -> array length T+h
    # F_sampler: function sample_impact_function(hyperparams) -> impact_function object
    # hyperparams contains p, kmax, etc.

    # 1. sample k
    kappa = geom_sample(p=hyperparams['p'])              # Geometric draw
    k = min(kappa, hyperparams['kmax'])

    covariates = []
    impact_functions = []

    for i in range(k):
        # 2. sample covariate either from corpus or synthetic generator
        if random_choice_from_corpus_or_synth():
            x_i = sample_from_corpus(corpus)  # random pick and clip/pad to length T+h
        else:
            x_i = synth_generator(T+h, hyperparams)
        covariates.append(x_i)

        # 3. sample impact function f_i from function space F (Algorithm 2)
        f_i = sample_impact_function(hyperparams)  # returns callable f_i(y, x_i) -> array (T+h)
        impact_functions.append(f_i)

    # 4. apply impacts to create augmented target
    y_aug = y.copy()
    for f_i, x_i in zip(impact_functions, covariates):
        y_aug += f_i(x=x_i, y=y)   # elementwise, length T+h

    return y_aug, covariates
```

Notes: the paper specifies sampling `kappa ∼ Geom(p)` and `k = min(kappa, kmax)`. This is line-for-line with Algorithm 1. 

---

### Algorithm 2 — Sample Impact Function (detailed)

The impact function `f` has the form (for each time t)

```
f_t(x,y) = { a0 + <a, x_{t-l : t}> + ε_t   if t ∈ S(x,y)
            { 0                                else
```

Implementation steps (pseudocode):

```python
def sample_impact_function(hyperparams):
    # hyperparams: contains pFO, pPW, plagcount, plagpos, l (max lag), s_epsilon, etc.
    # returns a callable f(x,y) -> array length T+h

    # initialize coefficients a[0..l] = 0
    a = np.zeros(hyperparams['l'] + 1)  # include a0 if you want separate bias

    if random_uniform() > hyperparams['pFO']:
        # sample number of active lags clag ~ Geom(plagcount)
        clag = geom_sample(p=hyperparams['plagcount'])
        # sample lag positions (recency-biased) L = {λ1,...,λ_clag} with λi ~ Geom(plagpos)
        L = [geom_sample(p=hyperparams['plagpos']) for _ in range(clag)]
        # sample coefficients for those lags A ~ N(0,1)
        A = [normal_sample(0,1) for _ in range(clag)]
        # set a[λ_j] = A_j
        for lag, coeff in zip(L, A):
            if lag <= hyperparams['l']:
                a[lag] = coeff

        # optionally sample piecewise parameters
        if random_uniform() > hyperparams['pPW']:
            a0 = normal_sample(0,1)    # bias
            z = random_choice(['y','x'])            # which series to test for domain S selection
            relation = random_choice(['>','<'])
            q = random_uniform()                     # quantile threshold
            # S(x,y) will be defined as indices where z_t ⋄ z_q (see below)
            piecewise = True
        else:
            # fallback simpler default
            a0 = 0.0
            z='y'; relation='>'; q=0.0

    else:
        # no impact (all zeros)
        return zero_impact_function()

    # define the S(x,y) domain selector (see next)
    def f_callable(x, y):
        # compute S(x,y) according to domain selection procedure (quantile threshold)
        S = compute_active_set(x, y, z, relation, q)
        # compute convolution-like dot <a, x_{t-l : t}> (treat missing lags as 0)
        # for t in 1..T+h: ft = (a0 + dot(a, x_{t-l:t}) + eps_t) if t in S else 0
        # noise eps_t ~ N(0, var = var_of_impact * s_epsilon)
        return f_array
    return f_callable
```

**Important notes from the paper:**

* The paper **samples clag from a geometric distribution** (`plagcount`) and lag positions from a geometric (`plagpos`) — this biases toward **sparse** and **recent** lags (recency + simplicity). 
* Coefficients for selected lags are sampled as `N(0,1)` (Algorithm 2). The paper also mentions using a *zero-inflated Gaussian* for sparse coefficients; the presented algorithm uses geometric draws for sparsity. 
* Noise `ε_t` is Gaussian; its variance is the variance of the impact scaled by `s_ε`. (See Table 2.) 

**Domain selection S(x,y):**

1. Choose variable `z ∼ {x, y}` uniformly.
2. Sample quantile level `q ∼ U(0,1)`. Compute empirical quantile value `z_q` on the sequence.
3. Sample inequality `⋄ ∈ {>, <}` uniformly.
4. `S(x,y) := { t | z_t ⋄ z_q }`. That is, only times where `z_t` is above/below that quantile are “active” and receive non-zero impact. This yields a piecewise (sparse-in-time) effect. 

---

### Algorithm 3 — Synthetic Covariates Generation (detailed)

Goal: produce realistic covariate signals that contain **events** (step or bell/gaussian bumps), **trends** and **changepoints**.

Pseudocode:

```python
def generate_synthetic_covariate(T, hyperparams):
    # 1) sample event count ce ~ Uniform(1, cmax_e)
    ce = random_int(1, hyperparams['cmax_e'])

    # 2) sample event positions Pe = {p0,...,p_{ce}} uniformly in [0, T)
    Pe = [random_int(0, T-1) for _ in range(ce)]

    # 3) sample type uniformly from {'step','gauss'}
    type = random_choice(['step','gauss'])

    # 4) for each event, sample event parameters and build xe (length T)
    xe = np.zeros(T)
    for pos in Pe:
        if type == 'gauss':
            alpha = sample_amplitude()        # e.g. N(0, sigma_a) or other
            sigma = sample_width()            # e.g. uniform range
            xe += alpha * gaussian_kernel(center=pos, sigma=sigma, length=T)
        else: # 'step'
            amplitude = sample_amplitude()
            # the step alternates at each event position; implement as cumulative step
            apply_step_at_positions(pe_list, amplitude)  # alternate sign optionally

    # 5) sample change-point count ccp ~ U(0, cmax_cp)
    ccp = random_int(0, hyperparams['cmax_cp'])
    pi = [random_int(0, T-1) for _ in range(ccp)]

    # 6) sample change-point amplitudes ai ~ N(0, sigma_cp), and construct a piecewise trend
    a_list = [normal(0, hyperparams['sigma_cp']) for _ in range(ccp + 3)] # includes endpoints
    # 7) build xtrend by interpolating points: {0,a0}, {pi[0], a1}, ..., {T, a_{end}}
    xtrend = interp_points([(0,a_list[0])] + list(zip(pi, a_list[1:-1])) + [(T-1, a_list[-1])])

    # return xe + xtrend
    return xe + xtrend
```

**Paper specifics:**

* `ce ∼ U(1, cmax_e)`. In Table 2, `cmax_e = 20`.
* For gaussian events: sample amplitude `α_i` and width `σ_i`, event contribution is `α_i * G(pos, σ_i)` (Gaussian kernel).
* For step events: amplitude is sampled; step alternates between each event position; events are summed with zeros.
* `ccp ∼ U(0, cmax_cp)`, `cmax_cp = 8`, and change-point amplitudes `ai ∼ N(0, σ_cp)` with `σ_cp = 2`. Then they construct a trend by ordering and interpolating points `({0,a0}, {π0,a1},..., {T, accp+2})`. Finally return `xe + xtrend`. 

---

### Hyperparameter defaults (explicit — copy these into config)

From Table 2 in the Appendix (use these as the starting defaults):

```
p      = 0.25     # geometric p for covariate count
pFO    = 0.2      # probability threshold used in Algorithm 2 (see code: impact exists if U > pFO)
pPW    = 0.15     # piecewise probability threshold (U > pPW leads to sampling piecewise domain)
kmax   = 10       # maximum number of covariates sampled
plagcount = 0.85  # geometric param for lag count (favors small clag)
plagpos   = 0.15  # geometric param for lag positions (favors recent lags)
l       = 500     # maximum allowed lag
s_epsilon = 0.02  # noise scale for impact epsilon_t
cmax_e  = 20      # max synthetic covariate events
cmax_cp = 8       # max synthetic covariate change-points
sigma_cp= 2       # stddev for change-point amplitudes
```

(Notation: `Geom` = geometric distribution; `U` = uniform; `N` = Gaussian.) 

---

# Implementation/integration tips — practical checklist for your coding assistant

1. **Vectorize for speed**: generate many covariates in batch (shape `(batch, T+h)`), and sample impact functions in vectorized loops. Use NumPy for gaussian bumps and linear interpolation for trends.
2. **Length and alignment**: COSMIC expects covariates of length `T+h` (past + forecast horizon). Ensure synthetic generator returns exactly that length. When sampling from corpus, clip/pad.
3. **Past-only vs past+future covariates**: the augmentation can handle both. If you want past-only covariates, zero out future slice `T+1 : T+h` in the covariate before returning. COSMIC trains with both types present. 
4. **Normalization**: the model applies per-series z-score normalization before patching. Implement normalization in the pipeline (compute mean/std per series) and store the scale if you need to unscale later. 
5. **Sparsity & recency bias**: use geometric distributions as specified; they intentionally bias to recent and sparse lags — preserve these choices for behavior matching the paper. 
6. **Noise scaling**: set `ε_t ~ N(0, var_of_impact * s_ε)` where `s_ε = 0.02` by default. This ensures the injected impact has realistic variance. 
7. **Determinism for debugging**: add RNG `seed` parameter to all sampling functions.
8. **API suggestions** (function signatures):

```python
generate_synthetic_covariate(T_plus_h, rng, cfg) -> np.array (T_plus_h,)
sample_impact_function(l_max, rng, cfg) -> callable f(x,y)-> np.array (T_plus_h,)
informative_covariate_augmentation(y, corpus, cfg, rng) -> (y_aug, [x_i])
```

9. **Unit tests**: write tests verifying (a) S(x,y) selection yields non-empty active sets sometimes, (b) `y_aug` differs from `y` when an impact exists, (c) synthetic covariates contain both trend and event components.
10. **Scaling to multivariate**: to generate *multivariate* covariates (many channels), call `generate_synthetic_covariate` repeatedly (or vectorize) and optionally correlate channels by adding a shared GP or scaled shared event signals. COSMIC’s augmentation treats each covariate separately and then adds their impacts — so the augmentation is natively multivariate (variable number `k`). 
