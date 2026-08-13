Below is an **execution-grade reproducibility guide** for the ICLR 2024 paper *“iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.”* I separate **what the paper explicitly specifies** from **what the current official codebase does**, because there are several real discrepancies between the PDF and the present GitHub scripts. The architecture itself is the encoder-only inverted Transformer described in the paper. 

## 1. Freeze the reproduction target first

* **Paper:** ICLR 2024, arXiv v4, iTransformer.
* **Official implementation:** `thuml/iTransformer`; the maintainers state that the experiment scripts are under `./scripts/`, including main forecasting, performance boosting, unseen-variate generalization, increasing-lookback, and efficient-attention experiments. ([GitHub][1])
* **Before running anything, record:**

  * Git commit SHA.
  * Python version.
  * CUDA/cuDNN versions.
  * GPU model.
  * `pip freeze`.
  * Exact shell script/CLI.
  * Seed.
* **Hardware reported in the paper:** one NVIDIA P100 16 GB GPU. 
* **The paper does not specify:** Python version, CUDA version, cuDNN version, exact PyTorch version, or a repository commit SHA.
* **Current repo dependency pins** are:

  * `pandas==1.5.3`
  * `scikit-learn==1.2.2`
  * `numpy==1.23.5`
  * `matplotlib==3.7.0`
  * `torch==2.0.0`
  * `reformer-pytorch==1.4.4` ([GitHub][2])
* Therefore, for a new reproduction, I would treat the current `requirements.txt` plus a pinned Git SHA as the environment specification, while noting that **those package versions are repo-derived, not reported in the paper**.

## 2. Core iTransformer architecture

* **Task definition:** input (X\in\mathbb{R}^{T\times N}), where:

  * `T` = lookback length.
  * `N` = number of variates/channels.
  * forecast (S) future steps for all `N` variates.
* **Invert the conventional Transformer layout:**

  * Conventional Transformer: time steps are tokens.
  * iTransformer: **entire histories of individual variates are tokens**.
  * So input is transposed from `T × N` to `N × T`. 
* **Token embedding:**

  * Each variate history (X_{:,n}\in\mathbb{R}^{T}) is embedded independently into a `D`-dimensional variate token.
  * Paper denotes this as `Embedding: R^T → R^D`.
  * No conventional positional embedding is needed because temporal ordering is encoded through the ordered neurons/weights operating on the history dimension. 
  * Current code implements `DataEmbedding_inverted` as:

    * permute `[B,T,N] → [B,N,T]`;
    * `Linear(T,D)`;
    * dropout.
    * No positional embedding is added. ([GitHub][3])
* **Encoder only; no forecasting decoder.**

  * Stack `L` inverted Transformer blocks.
  * Each block contains:

    * multi-head self-attention **across variate tokens**;
    * residual connection + LayerNorm;
    * shared FFN applied independently to each variate representation;
    * residual connection + LayerNorm. 
* **Attention semantics:**

  * token axis has length `N`, not `T`;
  * attention therefore produces an `N × N` multivariate-correlation map.
* **FFN semantics:**

  * operates independently on the representation of every variate;
  * its weights are shared across variates.
* **Output projection:**

  * `D → S` independently for every variate;
  * transpose resulting `N × S` back to `S × N`. 
* **Current-code architectural defaults when a script does not override them:**

  * `n_heads = 8`
  * `d_model = 512`
  * `e_layers = 2`
  * `d_ff = 2048`
  * attention `factor = 1`
  * dropout `= 0.1`
  * activation `= GELU`
  * `embed = timeF`
  * `output_attention = False`
  * `d_layers = 1`, although the main iTransformer forecaster is encoder-only.
  * `label_len = 48`, explicitly marked by the code as no longer needed for inverted Transformers. ([GitHub][4])
* **Important:** the supplied experiment scripts usually override `d_ff` so that it equals `d_model`; do not blindly use the parser default `d_ff=2048`.

## 3. Normalization

* There are **two distinct normalization operations** to keep separate:

  * **Dataset scaling:** each channel is standardized using statistics fitted on the training split.
  * **Model-side normalization / denormalization:** current code exposes `--use_norm`, defaulting to enabled. ([GitHub][4])
* LayerNorm inside the Transformer is additionally applied to each variate representation; this is a core architectural element in the paper.
* For exact reproduction, **always log `use_norm` explicitly rather than relying on its default**, particularly for PEMS. This flag has caused reproducibility differences in the public repository; users have documented significant PEMS sensitivity to it. ([GitHub][5])

## 4. Main optimization/training settings reported in the paper

* **Framework:** PyTorch.
* **Optimizer:** Adam.
* **Objective:** L2 loss, i.e. MSE forecasting loss.
* **Initial learning rate candidates:**

  * `1e-3`
  * `5e-4`
  * `1e-4`
* **Batch size:** `32`.
* **Maximum training epochs:** `10`.
* **Number of inverted Transformer blocks searched:** `L ∈ {2,3,4}`.
* **Token/representation dimension searched:** `D ∈ {256,512}`. 
* Do **not** interpret “L2 loss” as Adam weight decay:

  * current code uses `torch.optim.Adam(model.parameters(), lr=...)` with no explicit `weight_decay`;
  * criterion is `nn.MSELoss()`. ([GitHub][6])
* The PDF does **not explicitly specify the rule used to choose among the LR/L/D configurations**—for example, it does not say “choose the best validation MSE after X trials.” Do not invent that detail.

## 5. Current-code training behavior that needs to be reproduced too

* **Current hard-coded seed:** `2023` for:

  * Python `random`
  * NumPy
  * PyTorch. ([GitHub][4])
* Current code does **not** additionally set a full suite of deterministic CUDA/cuDNN flags, so bit-for-bit reproducibility across GPUs/software stacks should not be expected.
* **Default maximum epochs:** 10.
* **Default early-stopping patience:** 3 epochs.
* **Validation metric used for checkpointing:** MSE loss.
* Best validation checkpoint is saved and restored.
* **Default LR schedule:** `type1`:

  * epoch 1: base LR
  * epoch 2: `0.5 × LR`
  * epoch 3: `0.25 × LR`
  * epoch 4: `0.125 × LR`
  * etc. ([GitHub][7])
* **AMP:** disabled unless `--use_amp` is passed.
* **Iterations:** default `itr=1`.
* **Workers:** default `num_workers=10`.
* **Training/validation DataLoader:**

  * uses requested batch size;
  * `shuffle=True`;
  * `drop_last=True`.
* **Test loader:**

  * `batch_size=1`;
  * `shuffle=False`;
  * `drop_last=True`. ([GitHub][8])

## 6. Hyperparameter-sensitivity experiment: do not confuse it with the paper's main tuning range

* Appendix C performs a **broader diagnostic sweep** than Appendix A's stated main configurations.
* Hyperparameter-sensitivity values shown in Figure 9:

  * learning rate:

    * `1e-3`
    * `5e-4`
    * `3e-4`
    * `1e-4`
  * encoder blocks:

    * `1`
    * `2`
    * `3`
    * `4`
  * hidden/token dimension:

    * `64`
    * `256`
    * `512`
    * `1024`
* Experiment setting:

  * lookback `T=96`
  * prediction length `S=96`
  * datasets shown: ETT, ECL, Traffic, Weather.
* Main finding reported by the authors:

  * LR becomes especially sensitive for high-dimensional ECL and Traffic;
  * larger `L` or `D` is not uniformly better. 
* Therefore:

  * **main paper search space:** LR `{1e-3,5e-4,1e-4}`, L `{2,3,4}`, D `{256,512}`;
  * **sensitivity-study space:** broader values above.

## 7. Data and split specification

* All splits are **chronological**, not random, following the TimesNet protocol. 
* **ETTh1 / ETTh2**

  * 7 variates.
  * hourly.
  * `(train,val,test) = (8545,2881,2881)`.
  * horizons `{96,192,336,720}`.
* **ETTm1 / ETTm2**

  * 7 variates.
  * 15-minute.
  * `(34465,11521,11521)`.
  * horizons `{96,192,336,720}`.
* **Exchange**

  * 8 variates.
  * daily.
  * `(5120,665,1422)`.
  * horizons `{96,192,336,720}`.
* **Weather**

  * 21 variates.
  * 10-minute.
  * `(36792,5271,10540)`.
  * horizons `{96,192,336,720}`.
* **ECL**

  * 321 variates.
  * hourly.
  * `(18317,2633,5261)`.
  * horizons `{96,192,336,720}`.
* **Traffic**

  * 862 variates.
  * hourly.
  * `(12185,1757,3509)`.
  * horizons `{96,192,336,720}`.
* **Solar-Energy**

  * 137 variates.
  * 10-minute.
  * `(36601,5161,10417)`.
  * horizons `{96,192,336,720}`.
* **PEMS03**

  * 358 variates.
  * 5-minute.
  * `(15617,5135,5135)`.
* **PEMS04**

  * 307 variates.
  * `(10172,3375,3375)`.
* **PEMS07**

  * 883 variates.
  * `(16911,5622,5622)`.
* **PEMS08**

  * 170 variates.
  * `(10690,3548,3548)`. 
* **Lookback for the main public benchmarks:** `T=96`.
* **Market datasets:** lookback `T=144`, horizons `{12,24,72,144}`. 

## 8. Critical PEMS inconsistency in the PDF

* The prose in Appendix A says PEMS prediction lengths are:

  * `{12,24,36,48}`. 
* But **Table 4** lists:

  * `{12,24,48,96}`. 
* More importantly, **the actual full results in Table 9 are reported for `12,24,48,96`**, with input length 96. 
* Therefore, to reproduce the numbers in the published result tables, use:

  * `seq_len = 96`
  * `pred_len ∈ {12,24,48,96}`
* Treat `{12,24,36,48}` as a textual error/inconsistency rather than silently mixing protocols.

## 9. Current code's preprocessing details

* **ETT:** code uses the conventional fixed 12-month train / 4-month validation / 4-month test boundaries, with lookback context extending backward into the preceding split. ([GitHub][9])
* **Generic CSV datasets** such as ECL, Traffic, Weather, Exchange:

  * 70% train
  * 10% validation
  * 20% test
  * chronologically ordered.
  * validation/test windows include preceding `seq_len` observations as context.
  * `StandardScaler` is fit **only on training data**. ([GitHub][9])
* **Solar:**

  * 70/10/20 chronological split.
  * scaler fit on training split only. ([GitHub][9])
* **PEMS:**

  * 60% train
  * 20% validation
  * 20% test.
  * loads `data['data'][:, :, 0]`, i.e. the first feature in the `.npz`.
  * training statistics are used for standardization.
  * missing values are forward-filled, then backward-filled. ([GitHub][9])
* These details matter enough that I would **not replace the loaders with your own preprocessing pipeline** if the goal is paper reproduction.

## 10. Current official main-script configurations for the principal long-term datasets

These are useful for reproducing the **current official repository**, but they reveal some drift from the PDF's Appendix A search-space statement.

* **ETTh1**

  * `e_layers=2`.
  * horizons 96 and 192:

    * `d_model=256`
    * `d_ff=256`.
  * horizons 336 and 720:

    * `d_model=512`
    * `d_ff=512`.
  * other optimizer settings fall back to current defaults unless specified. ([GitHub][10])
* **ETTh2**

  * `e_layers=2`
  * `d_model=128`
  * `d_ff=128`
  * all four horizons. ([GitHub][11])
* **ETTm1**

  * `e_layers=2`
  * `d_model=128`
  * `d_ff=128`
  * all four horizons. ([GitHub][12])
* **ETTm2**

  * `e_layers=2`
  * `d_model=128`
  * `d_ff=128`
  * all four horizons. ([GitHub][13])
* **Exchange**

  * `e_layers=2`
  * `d_model=d_ff=128`
  * current script has the unusual special case `train_epochs=1` for horizon 336, while the other horizons use the default 10. ([GitHub][14])
* **Weather**

  * `e_layers=3`
  * `d_model=512`
  * `d_ff=512`
  * current default batch `32`
  * current default LR `1e-4`. ([GitHub][15])
* **Solar-Energy**

  * `e_layers=2`
  * `d_model=512`
  * `d_ff=512`
  * LR `5e-4`
  * default batch `32`. ([GitHub][16])
* **ECL**

  * `e_layers=3`
  * `d_model=512`
  * `d_ff=512`
  * batch `16`
  * LR `5e-4`. ([GitHub][17])
* **Traffic**

  * `e_layers=4`
  * `d_model=512`
  * `d_ff=512`
  * batch `16`
  * LR `1e-3`. ([GitHub][18])

### Why that matters

* The PDF says:

  * **uniform batch size = 32**
  * `D ∈ {256,512}`. 
* Current scripts contain:

  * batch 16 for ECL/Traffic;
  * `D=128` on several low-dimensional datasets.
* So **“paper Appendix configuration” and “current GitHub main script configuration” are not identical**.
* Do not combine them into a fictional single configuration. For publication-quality reproduction, record which definition you are testing.

## 11. PEMS practical reproduction warning

* The actual published PEMS table uses horizons `12/24/48/96`. 
* There is a documented reproducibility issue in the official repository where users report that the supplied/current configuration, particularly for longer PEMS horizons, can deviate materially from the paper; `use_norm` is one identified sensitivity. One example reproduction for PEMS03/96 uses:

  * `seq_len=96`
  * `pred_len=96`
  * `e_layers=4`
  * `d_model=d_ff=512`
  * `n_heads=8`
  * dropout `0.1`
  * batch `32`
  * LR `1e-3`
  * 10 epochs
  * patience `3`
  * `use_norm=0`. ([GitHub][5])
* That GitHub issue is **community reproduction evidence, not an authoritative replacement for the paper**. I would mark PEMS as “known version-sensitive” in any reproduction README.

## 12. Main forecasting experiment protocol

* Use `features=M`: multivariate input → multivariate output.
* Public long-horizon datasets:

  * `seq_len=96`.
  * independently train one model for each:

    * 96
    * 192
    * 336
    * 720.
* PEMS:

  * `seq_len=96`;
  * train separately for 12, 24, 48, 96.
* Compute:

  * MSE
  * MAE.
* Lower is better.
* Table 1 reports averages across the four forecasting horizons; full horizon-specific results appear in Tables 9–10. 
* Never average predictions from the four horizons—**train/evaluate four separate horizon models, then average their reported metrics**.

## 13. Baseline reproduction

* Paper compares against ten established forecasters:

  * **Transformer-based**

    * Autoformer
    * FEDformer
    * Stationary / Non-stationary Transformer
    * Crossformer
    * PatchTST
  * **Linear**

    * DLinear
    * TiDE
    * RLinear
  * **TCN/other deep forecasting**

    * SCINet
    * TimesNet. 
* Paper says reproduced baseline implementations were built from the **TimesNet benchmark**, following configurations from each model's original paper or official code. 
* **Missing from the PDF:** a complete per-baseline, per-dataset hyperparameter matrix.
* Therefore exact baseline reproduction requires:

  * the relevant TimesNet/code snapshot;
  * each baseline's supplied scripts/configuration.
* Do not “fairly retune” all baselines yourself and then call that a reproduction of Table 1; that would be a new benchmark.

## 14. Generality / “inverted Transformer variants” experiment

* Apply the exact inversion idea to:

  * vanilla Transformer → iTransformer
  * Reformer → iReformer
  * Informer → iInformer
  * Flowformer → iFlowformer
  * FlashAttention-equipped Transformer → iFlashformer/Flashformer.
* Architectural principle stays the same:

  * whole variates become tokens;
  * chosen attention mechanism operates across variates;
  * FFN operates on each series representation.
* Evaluate original vs inverted counterparts.
* Paper reports this primarily on ECL, Traffic, and Weather, with complete variant results in Appendix Table 8. 
* Current repo exposes model choices `[iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]`. ([GitHub][4])

## 15. Unseen-variate / zero-shot generalization protocol

* For each dataset:

  * split variates into **five folders/folds**.
  * For a given fold, train using **only 20% of the variates**.
  * At inference, give the trained iTransformer **all variates**.
  * **No fine-tuning** on the unseen variates.
  * repeat over all five folders.
  * average results over the five folders to remove partition randomness. 
* Comparison strategy:

  * CI-Transformer:

    * Channel Independence;
    * shared backbone;
    * predicts channels independently.
  * iTransformer:

    * dynamically accepts different numbers of variate tokens at train vs inference.
* Figure 18 explicitly says results are averaged over all five folds. 
* Current repo provides the corresponding `scripts/variate_generalization/...` workflow. ([GitHub][1])

## 16. Increasing-lookback experiment

* Fix prediction horizon:

  * `S=96`.
* Sweep input/lookback:

  * `T ∈ {48,96,192,336,720}`.
* Datasets shown:

  * ECL
  * Traffic.
* Compare:

  * Transformer vs iTransformer
  * Informer vs iInformer
  * Flowformer vs iFlowformer.
* Goal: demonstrate that inverted models increasingly benefit from longer input windows while temporal-token Transformers may not.
* This exact `T` sweep and fixed `S=96` are specified in Figure 6. 
* Current repo has a dedicated `scripts/increasing_lookback/...` experiment. ([GitHub][1])

## 17. Efficient-training / partial-variate experiment

* Problem: self-attention has quadratic complexity in the number of variate tokens.
* During **each training batch**, randomly select only a subset of variates.
* Train on those selected channels.
* At inference, feed all channels.
* Sweep sampled-variate ratios:

  * 100%
  * 80%
  * 60%
  * 40%
  * 20%.
* Figure 8 evaluates:

  * forecasting MSE
  * GPU memory footprint.
* Datasets shown:

  * ECL
  * Traffic
  * Solar-Energy.
* Claim being tested: partial-variate training preserves prediction performance while substantially reducing memory. 
* For the efficiency comparison in Figure 10:

  * input = 96
  * output = 96
  * representative datasets:

    * Weather: 21 variates
    * Traffic: 862 variates
  * same batch size is used for compared models;
  * efficient iTransformer uses 20% variates;
  * linear-complexity Flowformer attention is also examined. 
* **Paper does not give a sufficiently detailed timing methodology** such as number of warm-up iterations, CUDA synchronization details, or repeated timing statistics, so exact `ms/iter` reproduction should be considered hardware/runtime dependent.

## 18. Architecture ablation protocol

Reproduce all six component arrangements while leaving the rest of the training setup fixed:

* **Full iTransformer**

  * Variate dimension: Attention
  * Temporal/series representation: FFN
* **Replace 1**

  * Variate: Attention
  * Temporal: Attention
* **Replace 2 / vanilla-like orientation**

  * Variate: FFN
  * Temporal: Attention
* **Replace 3**

  * Variate: FFN
  * Temporal: FFN
* **Remove FFN**

  * Variate: Attention
  * Temporal: none
* **Remove attention**

  * Variate: none
  * Temporal: FFN
* Evaluate:

  * ECL
  * Traffic
  * Weather
  * Solar-Energy.
* Run all four normal forecasting horizons and report individual and average MSE/MAE.
* Full values are in Appendix Table 6. 

## 19. Representation-analysis / CKA experiment

* Extract representations from:

  * first Transformer block;
  * last Transformer block.
* Calculate **Centered Kernel Alignment (CKA)** similarity between them.
* Compare standard Transformer variants against their inverted counterparts.
* Higher first↔last CKA is interpreted by the authors as retaining representations better suited for the low-level forecasting task.
* **Reproduction gap:** the PDF does not provide every implementation-level choice required for exact CKA replication, such as the exact sample subset/batch indices. Therefore reproduce the methodological trend, but do not assume exact plotted coordinates are guaranteed from the paper alone.

## 20. Attention-map / correlation visualization

* Dataset: Solar-Energy.
* Calculate pairwise Pearson correlations for:

  * historical/lookback data;
  * true future data.
* Extract **pre-Softmax attention score maps** from:

  * first inverted-attention layer;
  * final inverted-attention layer.
* Compare:

  * early attention ≈ lookback correlations;
  * deep attention ≈ future-series correlations.
* Appendix says the showcased Solar examples are **randomly chosen**. 
* Therefore exact visual cases cannot be reconstructed from the PDF unless the code or saved sample indices provides them.

## 21. Prediction visualization experiment

* Input 96 → predict 96.
* Representative datasets:

  * Traffic
  * ECL
  * Weather
  * PEMS.
* Compare forecasts from:

  * iTransformer
  * PatchTST
  * DLinear
  * Crossformer
  * Autoformer
  * Transformer. 
* Again, the exact plotted test-window indices are not specified, so the figures are not fully reconstructable from the PDF alone.

## 22. Random-seed robustness experiment

* Paper reports a separate robustness table obtained using **five different random seeds**. 
* Important reproducibility gap:

  * the PDF does **not identify the five seed values**.
* Current repo hardcodes seed `2023`, so:

  * seed 2023 can reproduce the standard current run;
  * the exact five-run Table 5 seed set cannot be recovered from the paper alone. ([GitHub][4])
* If re-running the robustness analysis today, explicitly choose and publish five seeds rather than claiming they are the authors' original five.

## 23. Market / Alipay results

* Six Market subsets:

  * Merchant: 285 variables
  * Wealth: 485
  * Finance: 405
  * Terminal: 307
  * Payment: 759
  * Customer: 395.
* Each has:

  * `(7045,1429,1429)` train/val/test points;
  * 10-minute sampling;
  * lookback 144;
  * horizons `{12,24,72,144}`. 
* The dataset is described as Alipay online-transaction server-load data collected by the authors. 
* The PDF does not provide a standalone public-data specification sufficient to reconstruct that dataset from scratch.
* Consequently, **Table 11 is only fully reproducible if your code/data package includes the same Market data**; otherwise this portion should be labeled unavailable rather than substituted with another dataset.

## 24. Known paper ↔ current-repository mismatches you should put in the reproduction README

* **Batch size**

  * Paper: uniformly 32.
  * Current ECL/Traffic scripts: 16.  ([GitHub][17])
* **Hidden dimension**

  * Paper Appendix A: `{256,512}`.
  * Current scripts include `128` for ETTh2/ETTm1/ETTm2/Exchange. ([GitHub][11])
* **Epoch count**

  * Paper: fixed 10.
  * Current Exchange-336 script: 1 epoch. ([GitHub][14])
* **PEMS horizons**

  * prose: `12/24/36/48`;
  * tables/results: `12/24/48/96`.
* **PEMS normalization**

  * current repository/version behavior around `use_norm` is known to affect reproduction materially. ([GitHub][5])
* **Seeds**

  * paper reports five-seed statistics without giving seed IDs;
  * current repo fixes seed 2023.
* These mismatches mean that there is no defensible way to say “the paper and current main branch specify one single exact experiment configuration.” They don't.

## 25. Minimum run manifest to save for *every* experiment

* Dataset filename + checksum.
* Dataset loader type (`ETTh1`, `ETTm1`, `custom`, `Solar`, `PEMS`, etc.).
* Dataset split sizes.
* `features=M`.
* `seq_len`.
* `pred_len`.
* `enc_in` / number of variates.
* `d_model`.
* `d_ff`.
* `n_heads`.
* `e_layers`.
* attention implementation.
* dropout.
* activation.
* `factor`.
* `use_norm`.
* batch size.
* optimizer.
* starting LR.
* LR schedule.
* max epochs.
* early-stopping patience.
* loss.
* AMP setting.
* seed.
* Git SHA.
* dependency versions.
* GPU/CUDA stack.
* best validation epoch.
* test MSE.
* test MAE.
* wall-clock train time if reporting efficiency.
* peak GPU memory if reporting efficiency.

## 26. Recommended order for a clean reproduction

* **First:** reproduce one easy sanity case such as Weather `96→96`.
* **Second:** reproduce all four Weather horizons.
* **Third:** ECL and Traffic, because these exercise large `N`.
* **Fourth:** ETT/Exchange/Solar.
* **Fifth:** PEMS, while explicitly documenting the horizon and `use_norm` ambiguity.
* **Sixth:** inverted-attention variants.
* **Seventh:** increasing-lookback experiment.
* **Eighth:** five-fold unseen-variate experiment.
* **Ninth:** efficient partial-variate experiment.
* **Tenth:** ablations and CKA/attention visualizations.
* **Finally:** Market only if the original private dataset is actually available.

The most important point for someone using the codebase is to **treat the shell script plus Git SHA as the executable configuration**, while treating the PDF as the experimental specification. Where they conflict, report both rather than silently choosing one. The official repository explicitly provides scripts for the major experiment families, so those scripts should be the starting point for any current-code reproduction. ([GitHub][1])

[1]: https://github.com/thuml/iTransformer?utm_source=chatgpt.com "GitHub - thuml/iTransformer: Official implementation for \"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting\" (ICLR 2024 Spotlight) · GitHub"
[2]: https://github.com/thuml/iTransformer/blob/main/requirements.txt "iTransformer/requirements.txt at main · thuml/iTransformer · GitHub"
[3]: https://github.com/thuml/iTransformer/blob/main/layers/Embed.py "iTransformer/layers/Embed.py at main · thuml/iTransformer · GitHub"
[4]: https://github.com/thuml/iTransformer/blob/main/run.py "iTransformer/run.py at main · thuml/iTransformer · GitHub"
[5]: https://github.com/thuml/iTransformer/issues/93?utm_source=chatgpt.com "能否提供一下PEMS所有数据集的96步长预测结果 · Issue #93 · thuml/iTransformer"
[6]: https://github.com/thuml/iTransformer/blob/main/experiments/exp_long_term_forecasting.py "iTransformer/experiments/exp_long_term_forecasting.py at main · thuml/iTransformer · GitHub"
[7]: https://github.com/thuml/iTransformer/blob/main/utils/tools.py "iTransformer/utils/tools.py at main · thuml/iTransformer · GitHub"
[8]: https://github.com/thuml/iTransformer/blob/main/data_provider/data_factory.py "iTransformer/data_provider/data_factory.py at main · thuml/iTransformer · GitHub"
[9]: https://github.com/thuml/iTransformer/blob/main/data_provider/data_loader.py "iTransformer/data_provider/data_loader.py at main · thuml/iTransformer · GitHub"
[10]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh "iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh at main · thuml/iTransformer · GitHub"
[11]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh "iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh at main · thuml/iTransformer · GitHub"
[12]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh "iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh at main · thuml/iTransformer · GitHub"
[13]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh "iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh at main · thuml/iTransformer · GitHub"
[14]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/Exchange/iTransformer.sh "iTransformer/scripts/multivariate_forecasting/Exchange/iTransformer.sh at main · thuml/iTransformer · GitHub"
[15]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/Weather/iTransformer.sh "iTransformer/scripts/multivariate_forecasting/Weather/iTransformer.sh at main · thuml/iTransformer · GitHub"
[16]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh "iTransformer/scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh at main · thuml/iTransformer · GitHub"
[17]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/ECL/iTransformer.sh "iTransformer/scripts/multivariate_forecasting/ECL/iTransformer.sh at main · thuml/iTransformer · GitHub"
[18]: https://github.com/thuml/iTransformer/blob/main/scripts/multivariate_forecasting/Traffic/iTransformer.sh "iTransformer/scripts/multivariate_forecasting/Traffic/iTransformer.sh at main · thuml/iTransformer · GitHub"
