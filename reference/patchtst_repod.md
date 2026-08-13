Below is a **paper-faithful reproducibility checklist for PatchTST**, built from the uploaded ICLR 2023 paper and cross-checked against the authors’ official repository. I’m separating **[Paper]** from **[Official code]** because there are a few meaningful differences between the published description and the current `main` branch. The paper is *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* by Nie et al.  The authors’ GitHub explicitly describes itself as the official implementation and separates supervised and self-supervised code. ([GitHub][1])

* **0. Reproducibility policy — decide this before running anything**

  * Treat the **paper as the authority for the experiment definition** and the **authors’ scripts as the authority for implementation details omitted from the paper**.
  * Pin the exact Git commit before running: save `git rev-parse HEAD` in every result directory. The current official repo has separate `PatchTST_supervised` and `PatchTST_self_supervised` directories. ([GitHub][1])
  * Do **not** silently “modernize” the code. The official supervised requirements pin `torch==1.11.0`, while NumPy, pandas, matplotlib and scikit-learn are listed without versions.  For closest reproducibility, use PyTorch 1.11.0 and save the resolved versions of all other packages with `pip freeze`.
  * The paper does **not** report the CUDA, cuDNN, Python, CPU, or GPU version used for the main benchmark. Do not invent these. It only explicitly mentions an **NVIDIA A40 48 GB** in the context of ablations that ran out of memory. 
  * Save for every run: Git SHA, Python version, `pip freeze`, PyTorch/CUDA/cuDNN versions, GPU model, complete CLI, random seed, dataset checksum, train/val/test sizes, number of trainable parameters, selected epoch, best validation loss, MSE and MAE.

* **1. Core PatchTST architecture**

  * **Input:** multivariate history (x\in\mathbb{R}^{M\times L}), where `M` is number of variables/channels and `L` is look-back length.
  * **Channel independence:** split the multivariate input into `M` univariate series. Every channel is passed through the same Transformer weights, but its forward pass and attention map are independent. This is one of the two defining PatchTST ideas. 
  * **Efficient implementation:** for batch size `B`, patch to `B × M × P × N`, then reshape to `(B·M) × P × N` so a standard Transformer can process all channels. 
  * **Patching:** patch length `P`, stride `S`. For the supervised implementation, the last value is repeated/padded by one stride at the end and the paper gives
    [
    N=\left\lfloor\frac{L-P}{S}\right\rfloor+2.
    ]
    Thus `L=336,P=16,S=8 → N=42`, and `L=512,P=16,S=8 → N=64`. 
  * Each patch is linearly projected `P → D`, then receives a **learnable additive positional embedding**.
  * Backbone is an **encoder-only Transformer**; there is no forecasting decoder in PatchTST. The official training loop calls `model(batch_x)` directly whenever the model name contains `TST`; therefore `label_len`, decoder inputs and timestamp embeddings are not used by PatchTST itself. 
  * **Supervised head:** flatten the encoded patch representations from one channel and map them through a linear forecasting head to all `T` future values for that channel.
  * **Normalization:** normalize each input series instance and reverse the normalization on the prediction. The paper describes instance normalization; the released code uses RevIN. Keep it enabled for published-style runs.
  * **Loss:** MSE over predicted future values, averaged over channels. Evaluation reports both **MSE and MAE**.
  * **Transformer normalization:** BatchNorm rather than LayerNorm. The released supervised model constructor has `norm='BatchNorm'`. 
  * **Current supervised-code implementation details not fully spelled out in the paper:** `attn_dropout=0`, GELU, residual attention enabled (`res_attention=True`), post-norm (`pre_norm=False`), zero-initialized learnable positional encoding, flatten head, end padding. 
  * **Default RevIN/head flags in current supervised CLI:** `revin=1`, `affine=0`, `subtract_last=0` meaning subtract the instance mean rather than the final point, `individual=0`, `decomposition=0`, `padding_patch='end'`, `head_dropout=0`. ([GitHub][2])

* **2. Published Transformer size**

  * **Default/larger datasets:** `e_layers=3`, `n_heads=16`, `d_model=128`, FFN `128 → 256 → 128`, GELU. 
  * **Small datasets ILI, ETTh1, ETTh2:** `e_layers=3`, `n_heads=4`, `d_model=16`, `d_ff=128`. 
  * **Paper dropout:** encoder dropout `0.2` for all experiments. 
  * **Important code/paper difference:** the released small-dataset scripts use dropout/fc-dropout `0.3` for ETTh1, ETTh2 and ILI rather than the paper Appendix's blanket statement of `0.2`. If the goal is to reproduce the released numbers with the provided codebase, use the script values; if performing a literal paper-description recreation, record this discrepancy instead of hiding it.

* **3. Datasets and task dimensions**

  * Weather: `M=21`, `52,696` time points.
  * Traffic: `M=862`, `17,544` time points.
  * Electricity: `M=321`, `26,304` time points.
  * ILI: `M=7`, `966` time points.
  * ETTh1: `M=7`, `17,420`; ETTh2: `M=7`, `17,420`.
  * ETTm1: `M=7`, `69,680`; ETTm2: `M=7`, `69,680`. 
  * Weather contains German meteorological variables; Traffic is San Francisco freeway occupancy; Electricity is hourly demand for 321 customers; ILI is weekly influenza-like illness data; ETT consists of transformer-temperature data at hourly and 15-minute resolution. 
  * The authors intentionally omit Exchange-rate from the main eight-dataset benchmark. 
  * **Forecast horizons:** ILI `T ∈ {24,36,48,60}`; every other dataset `T ∈ {96,192,336,720}`. 

* **4. Exact chronological data splitting/scaling used by the official supervised code**

  * **ETTh1/ETTh2:** train = first 12 months; validation = next 4 months; test = next 4 months. Validation and test regions begin `seq_len` points before their nominal boundary so each target has historical context. The scaler is fit **only on the training portion**, then applied to the whole series. 
  * **ETTm1/ETTm2:** identical 12/4/4-month scheme at four samples per hour. StandardScaler is again fit only on train. 
  * **Custom datasets** used for Weather, Traffic, Electricity and ILI: chronological `70% train / 10% validation / 20% test`; val/test receive the preceding `seq_len` history; StandardScaler is fit only on the training section. 
  * Use `features='M'` for the multivariate experiments.
  * **Critical implementation quirk:** the released supervised dataloader uses `drop_last=True` for **train, validation and test**; train/validation are shuffled while test is not.  This means the final incomplete test batch is excluded. If your aim is exact released-number reproduction, preserve this behavior. If your aim is methodologically cleaner evaluation, set `drop_last=False` for test—but call that a corrected re-evaluation, because it can change the reported results. This behavior has also been explicitly discussed in the repository’s issue tracker. ([GitHub][3])

* **5. Main supervised PatchTST variants**

  * **PatchTST/42:** nominal main setting `L=336`, `P=16`, `S=8`, yielding 42 patches under end-padding. 
  * **PatchTST/64:** `L=512`, `P=16`, `S=8`, yielding 64 patches. 
  * The official README says the supplied supervised scripts default to PatchTST/42. ([GitHub][1])
  * To obtain `/64` from a `/42` script for the ordinary datasets, change `seq_len 336 → 512` while retaining `patch_len=16,stride=8`; leave the remaining dataset-specific architecture/training settings unchanged unless the corresponding experiment specifies otherwise.
  * **ILI is special:** the official ILI script uses `L=104`, `P=24`, `S=2`, which also yields 42 patches under the paper's end-padding formula: `floor((104-24)/2)+2 = 42`. Therefore `/42` denotes the number of patch tokens, not necessarily a universal `L=336`. The current ILI script also uses the smaller Transformer. ([GitHub][4])

* **6. Dataset-by-dataset supervised configurations — use these when reproducing with the official code**

  * **Weather:** `seq_len=336`; horizons `96/192/336/720`; `enc_in=21`; layers `3`; heads `16`; `d_model=128`; `d_ff=256`; encoder dropout `0.2`; `fc_dropout=0.2`; `head_dropout=0`; `P=16`; `S=8`; `epochs=100`; early-stop patience `20`; batch `128`; LR `1e-4`; seed `2021`; `itr=1`. It does not override `lradj`, so it inherits CLI default `type3`. 
  * **Electricity:** same main architecture; `enc_in=321`; epochs `100`; patience `10`; batch `32`; LR `1e-4`; scheduler `TST`/OneCycle, `pct_start=0.2`; seed `2021`; `itr=1`. 
  * **Traffic:** same main architecture; `enc_in=862`; epochs `100`; patience `10`; batch `24`; LR `1e-4`; scheduler `TST`/OneCycle, `pct_start=0.2`; seed `2021`; `itr=1`. 
  * **ETTm1:** `L=336`; `enc_in=7`; layers `3`; heads `16`; `d_model=128`; `d_ff=256`; dropout/fc-dropout `0.2`; head-dropout `0`; `P=16,S=8`; epochs `100`; patience `20`; batch `128`; LR `1e-4`; `lradj=TST`, `pct_start=0.4`; seed `2021`.
  * **ETTm2:** same as ETTm1.
  * **ETTh1:** `L=336`; `enc_in=7`; layers `3`; heads `4`; `d_model=16`; `d_ff=128`; released script dropout/fc-dropout `0.3`; `head_dropout=0`; `P=16,S=8`; epochs `100`; batch `128`; LR `1e-4`. With no script override, patience remains `100` and LR schedule is `type3`.
  * **ETTh2:** same released settings as ETTh1.
  * **ILI:** `L=104`; horizons `24/36/48/60`; `enc_in=7`; layers `3`; heads `4`; `d_model=16`; `d_ff=128`; dropout/fc-dropout `0.3`; head-dropout `0`; `P=24`; `S=2`; epochs `100`; batch `16`; LR **`2.5e-3`**; constant LR; seed `2021`; `itr=1`. ([GitHub][4])
  * For all of those supervised runs, AMP defaults to **off**, multi-GPU defaults to off, and the current CLI defaults to `num_workers=10`. ([GitHub][2])

* **7. Supervised optimization and model selection**

  * Optimizer in the official implementation is **Adam** over all model parameters. 
  * Training criterion is `torch.nn.MSELoss()`. 
  * Maximum epochs are generally `100`, with early stopping on **validation loss**. After training, the code reloads the checkpoint with the lowest validation loss before final test evaluation. 
  * `delta=0` in EarlyStopping; any strict validation improvement resets the patience counter. 
  * For scripts with `lradj='TST'`, a PyTorch **OneCycleLR** is created using `max_lr=learning_rate`, `epochs=train_epochs`, `steps_per_epoch=len(train_loader)` and the dataset-specific `pct_start`; it is stepped every minibatch. 
  * For `lradj='type3'`, LR remains at the requested LR for epochs 1–2 and thereafter follows approximately `base_lr × 0.9^(epoch-3)`. `constant` keeps LR fixed. 
  * Training code computes train, validation **and test** loss each epoch, but the checkpoint decision is made from validation loss only. 
  * Final reported metrics are calculated after loading the best checkpoint. 

* **8. Randomness / seeds**

  * The paper states that its reported main and appendix results use **random seed 2021**. 
  * The supervised entrypoint seeds Python `random`, NumPy and `torch.manual_seed` with that seed. ([GitHub][2])
  * The code does not visibly set all strict-determinism knobs such as deterministic CuDNN algorithms or explicit CUDA-all-device seeding in the entrypoint, so bitwise reproducibility across GPU/CUDA stacks should not be assumed.
  * Robustness experiment: supervised PatchTST is retrained using seeds `{2019,2020,2021,2022,2023}`; reported mean/std are in Table 14. For self-supervised robustness they pretrain once and fine-tune five times with different minibatch selections. 
  * Therefore compare a reproduction against the **seed-level variability in Table 14**, not only the final three decimal digits of one published run. 

* **9. Baselines and tuning protocol — essential if reproducing the comparison, not just PatchTST**

  * Main baselines: FEDformer, Autoformer, Informer, Pyraformer, LogTrans and DLinear. 
  * Default look-back used in source baseline results: Transformer models `L=96`; DLinear `L=336`. 
  * To make the Transformer baselines stronger, the authors rerun **FEDformer, Autoformer and Informer** at `L ∈ {24,48,96,192,336,720}` separately for every dataset × horizon task and select the best of the six. 
  * ILI uses a separate look-back grid: `{24,36,48,60,104,144}`; the ordinary Transformer default is 36 and DLinear default is 104. 
  * When recreating Table 3, do **not** compare PatchTST only to a single `L=96` FEDformer/Autoformer/Informer run; that would understate the baselines relative to the paper's protocol.
  * The paper does not specify a complete optimizer/batch/LR tuning search for every baseline in the text. Use the baseline implementations/scripts shipped with the authors’ codebase and archive their exact CLI rather than guessing omitted settings.
  * Also document whether “best look-back” is selected using validation performance or by post-hoc table score; the paper says the best result among the look-backs is used but does not state the selection mechanism as precisely as would be desirable. Do not silently assume one.

* **10. Self-supervised / masked-patch pretraining — paper protocol**

  * Context length **512**.
  * Patch length **12**.
  * Non-overlapping patches, therefore stride **12**.
  * This produces **42 patches**, with the unused remainder outside the complete patch sequence.
  * Mask **40%** of patch tokens.
  * Patch indices are chosen uniformly at random.
  * Replace masked patches with zeros.
  * Use the same Transformer encoder, but remove the supervised flatten/forecasting head and attach a `D × P` linear reconstruction head.
  * Optimize MSE **only for reconstruction of masked content** as defined by the masked-patch training setup. 
  * Pretrain for **100 epochs**. 
  * **Linear probing:** freeze the backbone and train only the forecasting head for **20 epochs**. 
  * **End-to-end fine-tuning:** first train the head for **10 epochs**, then train the full network for **20 epochs**. 
  * For same-dataset representation experiments, pretrain on the relevant dataset and perform downstream forecasting.
  * For the transfer experiment in Tables 5/13, pretrain on **Electricity**, then transfer to other datasets. 
  * The ETTh1 representation comparison also considers representation transferred from **Traffic** versus self-supervised training on ETTh1.

* **11. Self-supervised current-code settings — where the repository and paper diverge**

  * Current pretraining entrypoint defaults to: context `512`, target `96`, batch `64`, workers `0`, standard scaling, `P=12`, `S=12`, RevIN on, layers `3`, heads `16`, `d_model=128`, **`d_ff=512`**, Transformer dropout `0.2`, head dropout `0.2`, mask ratio `0.4`. 
  * It builds the pretraining model with **ReLU**, shared channel embedding, `head_type='pretrain'`, and `res_attention=False`. 
  * **Major mismatch:** the current script's default `n_epochs_pretrain=10`, whereas the paper explicitly states **100 pretraining epochs**. To reproduce the paper, override it to 100.  
  * Current pretraining code first runs an **LR finder**, then trains with `fit_one_cycle` at the LR returned by the finder. The CLI's `1e-4` is therefore a starting LR-finder setting, not necessarily the actual peak LR used for the final pretraining run. **Log the `suggested_lr` for every reproduction.** 
  * Current fine-tuning defaults: context `512`, batch `64`, `P=S=12`, RevIN, layers `3`, heads `16`, `d_model=128`, **`d_ff=256`**, dropout `0.2`, head dropout `0.2`, 20 fine-tune epochs and initial LR `1e-4`. 
  * Current fine-tune model also uses ReLU and `res_attention=False`. 
  * End-to-end code calls `fine_tune(..., n_epochs=20, freeze_epochs=10)`, matching the paper's 10-epoch frozen-head phase followed by 20 epochs of full fine-tuning. Linear probing uses 20 epochs. 
  * Fine-tuning also invokes an LR finder before the actual run. 
  * **Another important mismatch:** current pretraining uses `d_ff=512` while current downstream fine-tuning defaults to `d_ff=256`, and the paper's general model-parameter appendix says default `F=256`.   Treat this as version drift/implementation ambiguity. For a paper-faithful recreation I would explicitly set and record `d_ff` rather than relying on defaults.
  * The README describes the self-supervised pretraining script as training “PatchTST/64,” but the current script with `context=512,P=S=12` computes **42 complete patches**. The paper also states 42 patches for the representation-learning configuration. ([GitHub][1])  Use the numerical parameters, not the README label.

* **12. Patch-length ablation**

  * Fix `L=336`.
  * Prediction horizon `T=96`.
  * Set stride equal to patch length so patches are non-overlapping.
  * The prose says the tested lengths include `[4,8,16,24,32,40]`, while the Figure 4 caption shows `[2,4,8,12,16,24,32,40]`. This is an internal paper discrepancy; if reproducing Figure 4 exactly, use the caption's full eight-point grid.  
  * Authors conclude performance is relatively insensitive and `P≈8–16` is generally good. 

* **13. Look-back-window ablation**

  * Supervised PatchTST/42.
  * Vary ordinary dataset `L ∈ {24,48,96,192,336,720}`.
  * For ILI, the corresponding lengths are `{24,36,48,60,104,144}`.
  * Table 9 contains full MSE/MAE results for all horizons. 
  * Figure 2 emphasizes Weather, Traffic and Electricity and horizons `96` and `720`.

* **14. Patching × channel-independence ablation**

  * **P+CI:** complete PatchTST.
  * **CI only:** remove patching by setting `P=S=1`, retaining channel independence.
  * **P only:** retain patching but channel-mix by reshaping `B×M×P×N → B×(M·P)×N` instead of `(B·M)×P×N`.
  * **Original:** no patching and no channel independence; equivalent to original TST architecture.  
  * For Electricity and Traffic ablation runs, maximum epochs were reduced **from 100 to 20** because no-patching variants were extremely expensive. 
  * Some variants exhaust an NVIDIA A40 48 GB even at batch size 1; preserve OOM as an experimental outcome rather than reducing model size unless you label it as a modified experiment. 

* **15. Instance-normalization ablation**

  * Evaluate both PatchTST/64 and PatchTST/42 **with and without instance normalization**.
  * Otherwise retain the corresponding supervised configuration.
  * Published Table 11 shows normalization generally gives a modest improvement, while the main gains remain attributable to patching/channel independence. 

* **16. Transformer-size robustness ablation**

  * Run all six `(layers,D)` combinations:
    `(3,128)`, `(3,256)`, `(4,128)`, `(4,256)`, `(5,128)`, `(5,256)`.
  * Always use `F=2D`.
  * Base experiment is supervised PatchTST/42 forecasting 96 steps.
  * For Traffic and Electricity, cap maximum epochs at `50` for this ablation.  

* **17. Channel-independence analysis experiments**

  * Attention-map visualization uses supervised **PatchTST/64 on Electricity**.
  * Average attention matrices over **all heads and all encoder layers** before plotting. 
  * CI-vs-channel-mixing data-efficiency/overfitting experiment uses **Weather**, base `PatchTST/42`, horizon `96`, and five seeds `{2019,…,2023}`.
  * Left plot varies the fraction of training data; right plot uses all training data and displays the first 20 epochs. 
  * Best model selection is based on validation data. 

* **18. Minimum commands/workflow for a codebase owner**

  * Enter the supervised directory, install its requirements, place the Autoformer-format CSV files under `./dataset`, and run the dataset scripts under `scripts/PatchTST`. The authors show Weather as `sh ./scripts/PatchTST/weather.sh`. ([GitHub][1])
  * Reproduce `/42` first. Do all four horizons independently; each horizon is a separately trained model.
  * Then produce `/64` by changing the historical window as required and keeping a distinct output/checkpoint directory so `/42` checkpoints cannot be accidentally reused.
  * After each run verify that `result.txt`, the checkpoint and exact CLI correspond to the intended dataset/horizon.
  * For self-supervised reproduction, run pretraining **once per desired source dataset** with `context=512`, `P=S=12`, `mask_ratio=.4`, and critically override `n_epochs_pretrain=100`; then use the resulting checkpoint for either 20-epoch linear probing or the 10+20 end-to-end protocol. The official repo exposes separate pretrain and fine-tune entrypoints. ([GitHub][1])

* **19. Things that must be frozen before calling a reproduction successful**

  * Same raw CSV files and column order as the authors’ benchmark.
  * Same chronological splits.
  * Same train-only fitted StandardScaler.
  * Same `features=M`.
  * Same look-back and forecast horizon.
  * Same patch length, stride and end-padding behavior.
  * Same channel-independent reshape.
  * Same RevIN behavior.
  * Same Transformer layer/head/model/FFN sizes.
  * Same encoder/fc/head dropout values.
  * Same optimizer and LR schedule.
  * Same batch size.
  * Same max epochs and early-stop patience.
  * Same random seed.
  * Same `drop_last` behavior.
  * Same test checkpoint selected from validation loss.
  * Same MSE/MAE implementation.
  * Same code commit and environment.
  * For self-supervised runs: same mask RNG, mask ratio, number of pretraining epochs, LR-finder result, pretraining checkpoint and frozen/unfrozen epoch schedule.

* **20. Reproducibility gaps / traps to state explicitly in your own `REPRODUCE.md`**

  * **Paper vs current code dropout:** Appendix says encoder dropout `0.2` for all experiments, released small-dataset scripts use `0.3`.
  * **Paper vs current self-supervised epochs:** paper = `100`; current pretrain default = `10`.  
  * **Self-supervised FFN:** paper/default supervised architecture = `256`; current pretrain script = `512`; current fine-tune = `256`.  
  * **README self-supervised `/64` label conflicts with actual 512/12/12 = 42-patch configuration.**
  * **ILI is not a generic L=336/P16/S8 experiment:** released code uses `L104/P24/S2`. ([GitHub][4])
  * **Test `drop_last=True`:** exact code reproduction excludes the final partial test batch. 
  * Package versions other than PyTorch, complete system software and primary-training GPU are not pinned by the paper/repository; record your own rather than claiming they are author settings.
  * The paper gives a fixed seed but does not provide enough information for guaranteed bitwise GPU determinism.
  * Consequently, the strongest reproducibility claim should be **“same protocol and statistically matching published metrics”**, not necessarily identical floating-point output.

The biggest practical rule is: **do not run everything from parser defaults**. The per-dataset shell scripts override architecture, dropout, batch size, LR schedule and—in ILI’s case—even the patch construction. Likewise, do not run the current self-supervised script unchanged and call it the ICLR experiment, because its default is 10 pretraining epochs rather than the paper's 100.

[1]: https://github.com/yuqinie98/patchtst "GitHub - yuqinie98/PatchTST: An offical implementation of PatchTST: \"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.\" (ICLR 2023) https://arxiv.org/abs/2211.14730 · GitHub"
[2]: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/run_longExp.py?utm_source=chatgpt.com "PatchTST/PatchTST_supervised/run_longExp.py at main"
[3]: https://github.com/yuqinie98/PatchTST/issues/7?utm_source=chatgpt.com "Why use drop_last=True in test (and val) dataloader? #7"
[4]: https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/illness.sh "raw.githubusercontent.com"
