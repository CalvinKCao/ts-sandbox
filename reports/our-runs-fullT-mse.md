# Our runs — MSE vs PatchTST / iTransformer / MMPD

## Intended (same-row)

One L per dataset, shared by binary / iTransformer / MMPD / PatchTST. Horizons = the rows already in this doc. Eval = 100% test (not 50-window, not unique-seg 50% keep-index, not T=20k prefix). Dedicated-H, not clip-from-max-H.

| Dataset           | L   | Horizons             | Split                              | Notes                    |
| ----------------- | --- | -------------------- | ---------------------------------- | ------------------------ |
| ETT (h1/h2/m1/m2) | 336 | 96 / 192 / 336 / 720 | official ETT 12/4/4; 100% test     | dedicated-H              |
| weather           | 336 | 96 / 192 / 336 / 720 | 70/10/20 full-T; 100% test         | dedicated-H              |
| electricity       | 336 | 96 / 192 / 336 / 720 | 70/10/20 full-T; 100% test         | dedicated-H              |
| traffic           | 336 | 96 / 192 / 336 / 720 | 70/10/20 full-T; 100% test         | dedicated-H              |
| solar             | 336 | 96 / 192 / 336 / 720 | 70/10/20 full-T; 100% test         | dedicated-H; allv        |
| exchange          | 336 | 96 / 192 / 336 / 720 | 70/10/20; 100% test                | dedicated-H              |
| illness           | 104 | 24 / 36 / 48 / 60    | 70/10/20; 100% test                |                          |
| PeMS              | 96  | 12 / 24 / 48 / 96    | official 60/20/20; 100% test       | **PeMS is PeMS04**       |
| dynamic           | 600 | 60 / 120 / 180 / 300 | 70/10/20 full series; 100% test    | stride 1                 |

Ambiguous elsewhere: illness leftover L=148 scratch vs PatchTST ILI L=104; solar 7v T=30k vs allv full-T; PeMS is PeMS04, not PeMS-Bay. PatchTST cells are row-L `test_100pct` except solar (still L=512; no L=336 100%) and traffic H336 (still L=512; job 5608062 wrote no `test_100pct`).

File-of-record numbers from Narval `$SCRATCH/ts-sandbox-fullT` (11 Sep fullT iTransformer / MMPD) plus Killarney PatchTST L=336 `test_100pct` (solar and traffic H336 still L=512) and completed binary 100% evals. **Not paper tables.** Bold is lowest MSE among Binary-anchor, PatchTST, iTransformer, and MMPD-anchor. Binary-prob is shown but not ranked.

Tags: our runs only · test 100% · MSE ↓ · pulled 14 Sep 2026 · elec n=10 16 Sep 2026 · ETTh1 itrans 16 Sep 2026 · weather n=10 17 Sep 2026 · illness PWB+linear 19 Sep 2026 · illness n=10 19 Sep 2026 · solar binary 100% 19 Sep 2026 · PeMS PatchTST L=96 100% 19 Sep 2026 · PatchTST L-match 23 Sep 2026

Canvas: [our-runs-fullT-mse.canvas.tsx](/home/cao/.cursor/projects/home-cao-ts-sandbox/canvases/our-runs-fullT-mse.canvas.tsx)

## Lookback, horizon, and split

One row per dataset. L, horizon recipe, and the train/eval split are not matched across columns. These MSE cells are test-set windows (or a cap on that test set). They are not the texture val∪test 50% keep-index pack. Prefix T=20k / solar 7v T=30k are not in these cells.

| Dataset     | Horizons in table | Binary L / split | iTransformer L / split | MMPD L / split | PatchTST L / split | Notes |
| ----------- | ----------------- | ---------------- | ---------------------- | -------------- | ------------------ | ----- |
| ETTh1       | 96 / 192 / 336 / 720 | L=336; official ETT 12/4/4; train full (cap 15153 is a no-op); eval seeded cap **n=1838** dedicated-H | L=336; official ETT; `test_100pct` dedicated-H | L=336; official ETT; `anchor_mse_test_100pct` dedicated-H | L=336; official ETT; `test_100pct` dedicated-H | Binary is not 100% test (1838 / 2785 at H96, 1838 / 2161 at H720). |
| ETTh2       | 96 / 192 / 336 / 720 | L=336; official ETT full-T; eval 100% n=2161, clip-from-720 | L=336; official ETT; `test_100pct` dedicated-H | L=336; official ETT; `test_100pct` dedicated-H | L=336; official ETT; `test_100pct` dedicated-H | Binary shorter H are prefixes of H=720. |
| ETTm1       | 96 / 192 / 336 / 720 | L=336; official ETTm 12/4/4 full-T; eval 100% n=10801, clip-from-720 | L=336; official ETTm; `test_100pct` dedicated-H | L=336; official ETTm; `test_100pct` dedicated-H | L=336; official ETTm; `test_100pct` dedicated-H | Same clip-vs-dedicated-H mismatch as ETTh2. |
| ETTm2       | 96 / 192 / 336 / 720 | L=336; official ETTm full-T; eval 100% n=10801, clip-from-720 | L=336; official ETTm; `test_100pct` dedicated-H | L=336; official ETTm; `test_100pct` dedicated-H | L=336; official ETTm; `test_100pct` dedicated-H | Same as ETTm1. |
| weather     | 96 / 192 / 336 / 720 | L=336; 70/10/20 full-T; HP on 50% train then 100% refit; eval 100% n=9820, clip-from-720 | L=336; 70/10/20; `test_100pct` dedicated-H | L=336; 70/10/20; `test_100pct` dedicated-H | L=336; 70/10/20; `test_100pct` dedicated-H | frac50 is train HP only, not a 50% test / val∪test keep-index. |
| electricity | 96 / 192 / 336 / 720 | L=336; 70/10/20 full-T; eval 100% n=4541, clip-from-720 | L=336; 70/10/20; `test_100pct` dedicated-H | L=336; 70/10/20; `test_100pct` dedicated-H | L=336; 70/10/20; `test_100pct` dedicated-H | Binary clips from H=720. |
| traffic     | 96 / 192 / 336 / 720 | L=336; 70/10/20 full-T; eval 100% n=2789, clip-from-720 | L=336; 70/10/20; `test_100pct` dedicated-H | L=336; 70/10/20; `test_100pct` dedicated-H | L=336 H96/H192/H720 `test_100pct`; H336 cell is still L=512 | H336 job 5608062 finished at `eval_test_fraction=0.1` and wrote no `test_100pct`. Avg includes that L=512 cell. |
| solar       | 96 / 192 / 336 / 720 | L=336; 70/10/20 full-T (`solar_Alabama`, not T=30k); eval 100% n=9792, clip-from-720 | L=336; 70/10/20; `test_100pct` dedicated-H | L=336; 70/10/20; `test_100pct` dedicated-H | L=512; 70/10/20; `test_100pct` dedicated-H | Not 7v T=30k. No Killarney L=336 `test_100pct`. Cells marked L=512. |
| exchange    | 96 / 192 / 336 / 720 | L=336; 70/10/20 full-pool train; with-pretrain dedicated-H; eval from train leaves (cap 939 / full 1422 at H96; H720 n=798 is already 100%) | L=336 PWB+linear; 70/10/20; `test_100pct` dedicated-H | L=336 PWB+linear; 70/10/20; root `anchor_mse` (no `anchor_mse_test_100pct`) | L=336 PWB+linear (not L=512); 70/10/20; `test_100pct` dedicated-H | PatchTST matches iTrans L here. Binary is RealTS pretrain, not PWB+linear. Footnote cites train jobs, not the H96 fulltest leaf. |
| illness     | 24 / 36 / 48 / 60 | L=104; 70/10/20; unique-seg eval 100% n=170/158/146/134; PWB+linear dedicated-H | L=104 PWB+linear; 70/10/20; `test_100pct` dedicated-H | L=104 PWB+linear; 70/10/20; **half-test** n=85/79/73/67 | L=104 PWB+linear; 70/10/20; `test_100pct` dedicated-H | MMPD is half the test set. All four columns L=104 (not L=148 scratch). |
| PeMS        | 12 / 24 / 48 / 96 | L=96 native H=96; official 60/20/20; eval 100% n=3304; H12/24/48 clip from H=96 | L=96; official 60/20/20; `test_100pct` dedicated-H | L=96; official 60/20/20; `test_100pct` dedicated-H | L=96; official 60/20/20; `test_100pct` dedicated-H | Same L=96 as binary / iTrans / MMPD. Binary shorter H are still clips from H=96. Old L=512 `test_100pct` (jobs 5572811/13/14) and the 50-window leftover are not in cells. |
| dynamic     | 60 / 120 / 180 / 300 | L=600 native H=300; 70/10/20 full series (not T=20k); eval stride **300** n=333 | — (no cells on this grid) | L=600; 70/10/20; `test_100pct` stride 1 dedicated-H 60/120/180/300 | — (no cells on this grid) | Binary vs MMPD eval lattice differs (s300 vs s1). Binary H 60/120/180 never run. |

## Protocol (not matched across columns)

PatchTST matches the row lookback: Killarney L=336 dedicated-H `test_100pct` on ETT, weather, electricity, and traffic H96/H192/H720; exchange L=336 PWB+linear; illness L=104 PWB+linear; PeMS Vast L=96 `test_100pct` (same lookback as binary / iTrans / MMPD; overwrites the Killarney L=512 jobs 5572811/13/14 and the 11 Sep L=336 /42 column). Traffic H336 is still the L=512 cell (job 5608062, `eval_test_fraction=0.1`, no `test_100pct`). Solar is still L=512 `test_100pct` (no L=336 100% on Killarney). iTransformer / MMPD are still 11 Sep fullT L=336 except PeMS L=96, dynamic L=600. Exchange PatchTST, iTransformer, and MMPD-anchor cells are Killarney PWB+linear synth pretrain then real finetune (L=336), not 11 Sep from-scratch. Illness PatchTST / iTransformer / MMPD-anchor are rented-L40S PWB+linear L=104 (not L=148 scratch / not Killarney 5555984–87 scratch / not 11 Sep from-scratch); MMPD-anchor is half-test (n=85/79/73/67 of 170/158/146/134). Binary exchange is still cap1x2x with-pretrain, dedicated H, n_samp=10 (jobs 5057766 / 2952251–53). Binary illness is Killarney U=217 PWB+linear: H24/36/48 dedicated **5557476–78** aMSE 2.249 / 2.666 / 2.251, pMSE 2.320 / 2.520 / 2.134 (n_samp=10); H60 **5548243** aMSE 2.058 and **5557547** pMSE 2.056. Binary electricity-anchor is prior clip-from-720 100% eval-anchor (n=1, n_win=4541, Narval 2831625–2831641); binary-prob is Killarney n=10 efficient-pack 100% (jobs 5471563–5471572, n_win=4541). Binary traffic is prior clip-from-720 100% eval-anchor (n=2789). Binary ETTh1 is Killarney dedicated-H iTrans-encoder eval-anchor (jobs 5474030/31/33/34, n=1838, configs `…nopretrain_itrans_hz{96,192,336,720}`); n=10 pMSE was not run. Binary ETTh2 / ETTm1 / ETTm2 are older n_samp=10 100% clip-from-max-H. Binary dynamic is L=600 native H=300 (n=333, n_samp=10); H 60/120/180 binary were not evaluated. Binary weather is Killarney p32 fullT H720 frac50-refit 100% stride-1 n=9820 (jobs **5489026–45**, 20/20 CD, n_samp=10, prefixes H96/192/336); old 10% / old-protocol leftover: anchor Narval **2954294** (n=982, n_samp=1) 0.149/0.194/0.249/0.322 and prob **2650285** (n=9820) 0.152/0.194/0.244/0.321. Binary PeMS H96 is native L=96 (eval-anchor 2954248 on ckpt 2673014, n=3304, n_samp=1, aMSE 0.4038; table 0.404). Binary-prob 0.398 is older n_samp=10. H12 / 24 / 48 binary never trained. Solar binary is Killarney of96 100% on ckpt **2650286** (n=9792, n_samp=10; leftover shards **5440840–99 / 5440921–26**). Do not read a column win as the same recipe.


| Column                         | Train                                             | Eval                          | Notes                                                                                                                             |
| ------------------------------ | ------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| PatchTST                       | Killarney L=336 uncapped (illness L=104 PWB+linear on rented L40S; PeMS Vast L=96; solar still L=512; traffic H336 still L=512) | `test_100pct` MSE             | Dedicated H, L matched to the row. Traffic H336 100% did not land (job 5608062, 10% only); that cell is still L=512. Solar cells are still L=512 `test_100pct` (no L=336 100%). PeMS is Vast L=96 `test_100pct` (fair run 19 Sep). Old Killarney PeMS L=512 jobs 5572811/13/14 and failed H24 5572812 are not in cells. Exchange is `hz*_lb336_pwb_linear`. Illness is PWB+linear L=104, not L=148 scratch. |
| iTransformer                   | 11 Sep fullT uncapped                             | `test_100pct` MSE             | Dedicated H. L=336 except illness L=104, PeMS L=96. Exchange is Killarney PWB+linear; illness is rented-L40S PWB+linear, not 11 Sep.                   |
| MMPD-anchor                    | 11 Sep fullT uncapped                             | `anchor_mse_test_100pct`      | Same H/L grid except dynamic (L=600, H in 60/120/180/300). Exchange is PWB+linear `anchor_mse` (no `anchor_mse_test_100pct`). Illness is rented-L40S PWB+linear L=104, half-test n=85/79/73/67. |
| Binary exchange / illness      | Exchange: cap1x2x **with-pretrain**, dedicated H, n_samp=10. Illness: U=217 PWB+linear, dedicated H | staged anchor / prob MSE      | Exchange H96 job 5057766; H192/336/720 jobs 2952251–53. Illness H24/36/48 **5557476–78** n=170/158/146 n_samp=10; H60 aMSE **5548243** n_win=134 n_samp=0; pMSE **5557547** n_samp=10. |
| Binary electricity / traffic   | prior uncapped fullT, clip-from-720               | 100% eval-anchor + elec n=10  | Elec aMSE n=1 Narval 2831625–2831641; elec pMSE/CRPS n=10 Killarney 5471563–72 (n_win=4541). Traffic n=2789, Killarney 5356912–5356928 |
| Binary ETTh1                   | 5057773 HPs except `guidance_type: itransformer`, dedicated H, nopre | eval-anchor n=1838            | Jobs 5474030/31/33/34; configs `…nopretrain_itrans_hz{96,192,336,720}`. iTrans encoder, not decoder. pMSE not run.               |
| Binary ETTh2 / m1 / m2         | older n_samp=10 100% clip-from-max-H              | staged MSE                    | Solar binary is Killarney of96 **100%** n=9792 n_samp=10 (ckpt **2650286**; not of10 n≈3988; not 7v T=30k aMSE 1.6559)               |
| Binary dynamic                 | L=600 native H=300, n_samp=10 100%                | staged MSE                    | n=333. H 60/120/180 never evaluated. PatchTST/iTrans have **no** 60/120/180/300 cells                                             |
| Binary weather                 | p32 fullT H720 frac50 refit, clip-from-720        | 100% n=9820 n_samp=10         | Killarney **5489026–45** (20 shards, all CD). Window-weighted prefix pMSE/aMSE H96/192/336; unsuffixed = H720. Old 10% aMSE **2954294** n=982 n_samp=1 and old pMSE **2650285** not in cells. |
| Binary PeMS                    | native L=96 H=96                                  | 100% **eval-anchor** n_samp=1 | Job **2954248** on ckpt 2673014, n=3304, aMSE 0.4038 (table 0.404). Binary-prob 0.398 is older n_samp=10. H12/24/48 never trained |




## Ranked-row wins


| PatchTST | MMPD-anchor | Binary-anchor | iTransformer |
| -------- | ----------- | ------------- | ------------ |
| 15       | 19          | 11            | 3            |


48 ranked horizon rows (ties counted for each winner; none left after the L-match pass). Avg rows are per-dataset means of numeric cells and are not in that tally. Dataset-avg winners: PatchTST 3, MMPD-anchor 6, Binary-anchor 3, iTransformer 1. Weather avg is a 0.227 tie (Binary-anchor and MMPD-anchor), counted for both. Binary-prob is omitted from the rank. Exchange binary is with-pretrain dedicated-H. Illness is PWB+linear (Binary U=217 dedicated-H **5557476–78 / 5548243 / 5557547**). Solar binary 100% of96 moved H96/192/336 from Binary-anchor to MMPD-anchor. PatchTST L=336 `test_100pct` replaced the L=512 cells on ETT, weather, electricity, and traffic H96/H192/H720: ETTh1 H336 is MMPD-anchor alone (0.434); weather H96 is MMPD-anchor 0.149, H336 Binary-anchor 0.244, H720 Binary-anchor 0.318; electricity H336/H720 are Binary-anchor 0.165/0.202; traffic H96/H192 are PatchTST 0.367/0.387. Traffic H336 stays L=512. Solar PatchTST stays L=512. PeMS PatchTST L=96 `test_100pct` (Vast fair, 19 Sep) replaced the L=512 cells; H12 and H96 now go to iTransformer (0.095 / 0.400), H24 stays iTransformer 0.143, H48 goes to Binary-anchor 0.240. PeMS dataset-avg is now iTransformer.

## MSE by dataset and horizon

Source: Killarney PatchTST under `temp/baselines_canvas128_subset/results/hz*_lb336/` field `eval_subsets.test_100pct.mse` (`seq_len=336`, log `eval_test_fraction=0`), except solar and traffic H336 which stay `hz*_lb512/` `test_100pct` and are marked L=512, and PeMS which is Vast L=96 `results/hz*_lb96/patchtst/PeMS/`; iTransformer still 11 Sep Narval `hz*/itransformer`; MMPD `results/datasets/09-11-mmpd-fullT-*/partials` (`anchor_mse_test_100pct`); exchange PatchTST / iTransformer / MMPD from Killarney `hz{H}_lb336_pwb_linear/` (`test_100pct.mse`; MMPD root `anchor_mse` — no `anchor_mse_test_100pct`); illness PatchTST / iTransformer / MMPD from rented-L40S PWB+linear L=104 (`test_100pct.mse`; MMPD `anchor_mse`, half-test, not 100%); binary from staged_eval JSON / wandb shards as footnoted. solar binary from Killarney of96 `windows.jsonl` 100% (n=9792). Avg is the mean of numeric cells in that dataset block (3 dp; dashes skipped). Traffic avg includes the marked L=512 H336 cell. Dynamic uses native H 60/120/180/300 (not the 96/192/336/720 grid).


| Dataset     | H   | Binary-anchor | Binary-prob | PatchTST  | iTransformer | MMPD-anchor |
| ----------- | --- | ------------- | ----------- | --------- | ------------ | ----------- |
| ETTh1       | 96  | 0.425         | —           | 0.385     | 0.406        | **0.372**   |
| ETTh1       | 192 | 0.447         | —           | 0.414     | 0.447        | **0.404**   |
| ETTh1       | 336 | 0.455         | —           | 0.441     | 0.471        | **0.434**   |
| ETTh1       | 720 | 0.529         | —           | 0.456     | 0.599        | **0.443**   |
| ETTh1       | Avg | 0.464         | —           | 0.424     | 0.481        | **0.413**   |
| ETTh2       | 96  | 0.325         | 0.323       | **0.275** | 0.327        | 0.287       |
| ETTh2       | 192 | 0.343         | 0.342       | **0.339** | 0.416        | 0.357       |
| ETTh2       | 336 | **0.349**     | 0.350       | 0.367     | 0.440        | 0.366       |
| ETTh2       | 720 | 0.432         | 0.438       | **0.392** | 0.440        | 0.403       |
| ETTh2       | Avg | 0.362         | 0.363       | **0.343** | 0.406        | 0.353       |
| ETTm1       | 96  | 0.312         | 0.291       | 0.292     | 0.318        | **0.279**   |
| ETTm1       | 192 | 0.339         | 0.322       | 0.334     | 0.359        | **0.313**   |
| ETTm1       | 336 | 0.371         | 0.356       | 0.366     | 0.386        | **0.351**   |
| ETTm1       | 720 | 0.427         | 0.412       | 0.419     | 0.450        | **0.404**   |
| ETTm1       | Avg | 0.362         | 0.345       | 0.353     | 0.378        | **0.337**   |
| ETTm2       | 96  | 0.185         | 0.181       | **0.165** | 0.193        | 0.182       |
| ETTm2       | 192 | 0.243         | 0.241       | **0.221** | 0.251        | 0.232       |
| ETTm2       | 336 | 0.296         | 0.296       | **0.277** | 0.302        | 0.290       |
| ETTm2       | 720 | 0.385         | 0.387       | **0.369** | 0.395        | 0.375       |
| ETTm2       | Avg | 0.277         | 0.276       | **0.258** | 0.285        | 0.270       |
| weather     | 96  | 0.152         | 0.149 n=10  | 0.150     | 0.161        | **0.149**   |
| weather     | 192 | **0.192**     | 0.189 n=10  | 0.195     | 0.206        | 0.193       |
| weather     | 336 | **0.244**     | 0.241 n=10  | 0.248     | 0.255        | 0.246       |
| weather     | 720 | **0.318**     | 0.316 n=10  | 0.321     | 0.327        | 0.319       |
| weather     | Avg | **0.227**     | 0.224       | 0.229     | 0.237        | **0.227**   |
| electricity | 96  | **0.127**     | 0.118 n=10  | 0.130     | 0.135        | 0.134       |
| electricity | 192 | **0.145**     | 0.136 n=10  | 0.150     | 0.154        | 0.150       |
| electricity | 336 | **0.165**     | 0.155 n=10  | 0.166     | 0.169        | 0.166       |
| electricity | 720 | **0.202**     | 0.188 n=10  | 0.210     | 0.206        | 0.205       |
| electricity | Avg | **0.160**     | 0.149       | 0.164     | 0.166        | 0.164       |
| traffic     | 96  | 0.399         | —           | **0.367** | 0.397        | 0.376       |
| traffic     | 192 | 0.409         | —           | **0.387** | 0.417        | 0.400       |
| traffic     | 336 | 0.424         | —           | 0.420 L=512 | 0.433      | **0.405**   |
| traffic     | 720 | 0.476         | —           | 0.457     | 0.467        | **0.440**   |
| traffic     | Avg | 0.427         | —           | 0.408     | 0.429        | **0.405**   |
| solar       | 96  | 0.189         | 0.181       | 0.214 L=512 | 0.208      | **0.184**   |
| solar       | 192 | 0.203         | 0.195       | 0.230 L=512 | 0.227      | **0.189**   |
| solar       | 336 | 0.212         | 0.204       | 0.246 L=512 | 0.224      | **0.200**   |
| solar       | 720 | 0.221         | 0.212       | 0.239 L=512 | 0.221      | **0.200**   |
| solar       | Avg | 0.206         | 0.198       | 0.232 L=512 | 0.220      | **0.193**   |
| exchange    | 96  | 0.095         | 0.086       | **0.089** | 0.096        | 0.095       |
| exchange    | 192 | **0.177**     | 0.175       | 0.186     | 0.192        | 0.206       |
| exchange    | 336 | 0.360         | 0.337       | **0.352** | 0.382        | 0.403       |
| exchange    | 720 | **0.912**     | 0.915       | 0.920     | 0.914        | 0.984       |
| exchange    | Avg | **0.386**     | 0.378       | 0.387     | 0.396        | 0.422       |
| illness     | 24  | 2.249         | 2.320       | **1.883** | 2.136        | 2.193       |
| illness     | 36  | 2.666         | 2.520       | **1.875** | 2.181        | 3.185       |
| illness     | 48  | 2.251         | 2.134       | **1.616** | 2.396        | 3.575       |
| illness     | 60  | 2.058         | 2.056       | **1.738** | 2.497        | 2.288       |
| illness     | Avg | 2.306         | 2.258       | **1.778** | 2.303        | 2.810       |
| PeMS        | 12  | 0.102         | 0.099 n=10  | 0.101     | **0.095**    | 0.105       |
| PeMS        | 24  | 0.148         | 0.144 n=10  | 0.161     | **0.143**    | 0.170       |
| PeMS        | 48  | **0.240**     | 0.236 n=10  | 0.294     | 0.243        | 0.311       |
| PeMS        | 96  | 0.404         | 0.398 n=10  | 0.510     | **0.400**    | 0.527       |
| PeMS        | Avg | 0.224         | 0.219       | 0.267     | **0.220**    | 0.278       |
| dynamic     | 60  | —             | —           | —         | —            | **0.221**   |
| dynamic     | 120 | —             | —           | —         | —            | **0.294**   |
| dynamic     | 180 | —             | —           | —         | —            | **0.345**   |
| dynamic     | 300 | 0.512         | 0.477       | —         | —            | **0.414**   |
| dynamic     | Avg | 0.512         | 0.477       | —         | —            | **0.319**   |


## CRPS by dataset and horizon

PatchTST, iTransformer, and Binary-anchor are point forecasts and have no CRPS. Binary-prob CRPS exists when `n_samp=10`. MMPD CRPS is from 100% dumps that match this table's MMPD-anchor MSE: ETTh1/ETTh2 (10 Sep Narval), 11 Sep `09-11-mmpd-fullT` on Narval (ETTm1/m2, weather, electricity, solar, PeMS), and exchange PWB+linear on Killarney. Illness / dynamic / traffic were not in this backfill. Bold is lowest CRPS in the row among filled cells.

| Dataset     | H   | Binary-prob | MMPD |
| ----------- | --- | ----------- | ---- |
| ETTh1       | 96  | 0.311       | **0.297** |
| ETTh1       | 192 | 0.330       | **0.320** |
| ETTh1       | 336 | 0.343       | **0.338** |
| ETTh1       | 720 | 0.380       | **0.359** |
| ETTh1       | Avg | 0.341       | **0.329** |
| ETTh2       | 96  | 0.314       | **0.268** |
| ETTh2       | 192 | 0.328       | **0.308** |
| ETTh2       | 336 | 0.337       | **0.318** |
| ETTh2       | 720 | 0.388       | **0.340** |
| ETTh2       | Avg | 0.342       | **0.309** |
| ETTm1       | 96  | 0.272       | **0.247** |
| ETTm1       | 192 | 0.289       | **0.266** |
| ETTm1       | 336 | 0.307       | **0.297** |
| ETTm1       | 720 | **0.335**   | 0.339 |
| ETTm1       | Avg | 0.301       | **0.287** |
| ETTm2       | 96  | **0.213**   | 0.217 |
| ETTm2       | 192 | 0.248       | **0.236** |
| ETTm2       | 336 | **0.280**   | 0.292 |
| ETTm2       | 720 | **0.332**   | 0.333 |
| ETTm2       | Avg | **0.268**   | 0.270 |
| weather     | 96  | 0.156       | **0.149** |
| weather     | 192 | 0.192       | **0.182** |
| weather     | 336 | 0.227       | **0.216** |
| weather     | 720 | 0.272       | **0.263** |
| weather     | Avg | 0.212       | **0.203** |
| electricity | 96  | **0.169**   | 0.177 |
| electricity | 192 | **0.183**   | 0.186 |
| electricity | 336 | **0.198**   | 0.201 |
| electricity | 720 | **0.220**   | 0.231 |
| electricity | Avg | **0.193**   | 0.199 |
| traffic     | 96  | —           | —    |
| traffic     | 192 | —           | —    |
| traffic     | 336 | —           | —    |
| traffic     | 720 | —           | —    |
| traffic     | Avg | —           | —    |
| solar       | 96  | 0.160       | **0.145** |
| solar       | 192 | 0.170       | **0.148** |
| solar       | 336 | 0.176       | **0.159** |
| solar       | 720 | 0.179       | **0.157** |
| solar       | Avg | 0.171       | **0.152** |
| exchange    | 96  | **0.166**   | 0.190 |
| exchange    | 192 | **0.248**   | 0.265 |
| exchange    | 336 | 0.357       | **0.310** |
| exchange    | 720 | **0.638**   | 1.184 |
| exchange    | Avg | **0.352**   | 0.487 |
| illness     | 24  | **0.775**   | —    |
| illness     | 36  | **0.824**   | —    |
| illness     | 48  | **0.767**   | —    |
| illness     | 60  | **0.778**   | —    |
| illness     | Avg | **0.786**   | —    |
| PeMS        | 12  | **0.158**   | 0.162 |
| PeMS        | 24  | **0.196**   | 0.210 |
| PeMS        | 48  | **0.266**   | 0.289 |
| PeMS        | 96  | **0.375**   | 0.389 |
| PeMS        | Avg | **0.249**   | 0.263 |
| dynamic     | 60  | —           | —    |
| dynamic     | 120 | —           | —    |
| dynamic     | 180 | —           | —    |
| dynamic     | 300 | **0.298**   | —    |
| dynamic     | Avg | **0.298**   | —    |

Binary-prob sources match the MSE table dumps: ETTh2/ETTm1/ETTm2 Narval clip-from-720 100% (`reports/narval-horizon-accuracy.md`); weather Killarney **5489026–45** window-weighted `crps_h96/_h192/_h336` + unsuffixed `crps`; electricity **5471563–72** same; solar of96 unique windows 0–9791 `windows.jsonl`; illness dedicated n=10 **5557476–78 / 5557547**; PeMS H96 n_samp=10 100%; dynamic native H=300 n=333. ETTh1 Binary-prob CRPS is Killarney dedicated-H n=10 `eval_resume/canvas128_p64x6_cap1x2x_nopre_itrans_hz{96,192,336,720}_s1_n10_steps4` n=1838 (not the eval-anchor **5474030–34** `n_samp=0` MSE jobs, and not the `_fulltest` n=2785 dump). Exchange Binary-prob H96 is job **5057766** n=939; H192/336/720 are Narval `09-11-2890531–33` `partials/exchange_rate_staged_s1_prob.json` (mse 0.175 / 0.337 / 0.915, n=939 / 939 / 798). MMPD ETTh1/ETTh2 from the 10 Sep Narval 100% jsons. Remaining MMPD CRPS from Narval `$SCRATCH/ts-sandbox-fullT/results/datasets/09-11-mmpd-fullT-*/partials/*_mmpd.json` `crps_test_100pct` (anchor MSE matches this table). Exchange MMPD from Killarney `hz{H}_lb336_pwb_linear/partials/exchange_rate_mmpd.json`. PeMS Binary-prob H12/24/48/96 are unique-window means over Narval n=10 H=96 fulltest shards `canvas128_p32x6_pems_lb96_hz96_r0_ms35_kv_a100_s1_n10_steps4_fulltest_shard{0–9}of10` (n=3304; jsonl H96 CRPS 0.37545 matches the old H96 cell; prefixes computed from `samples`). Illness / dynamic / traffic were not in this CRPS backfill.


**Solar binary (of96 vs of10 vs 7v).** Old of10 cells aMSE 0.182 / 0.188 / 0.189 / 0.224 (pMSE 0.177 / 0.184 / 0.188 / 0.216) are the mean of 3988 compact rows: partial prefixes of all 10 shards spread over the **full** 0–9791 index space, not a sequential first-third of the test tail. Only 1677 of those windows sit in 0–3987; 2311 are later (max compact index 9188). Sequential 0–3987 would be aMSE 0.219 / 0.236 / 0.256 / 0.272 — **not** the old table. Current cells: of96 **resume from of10**, then leftover jobs filled the rest (66 shards needed leftover work); equal-window mean over unique windows 0–9791 (n=9792, no double-count, no missing tail), n_samp=10, ckpt **2650286**. Same full-T 70/20/10 split as iTransformer / MMPD. **Not** 7v T=30k (Vast A100 `…solar_alabama_7v_rep_T30k…` aMSE **1.6559**, n=5905).

**PatchTST solar.** Cells stay Killarney L=512 `test_100pct` (0.214 / 0.230 / 0.246 / 0.239), marked L=512. No `hz*_lb336/patchtst/solar_Alabama/` on Killarney as of 23 Sep 2026. Prior L-match note: L=336 dumps were YAML 10% only (`eval_test_fraction=0.1`, empty `eval_subsets`; root mse 0.228 / 0.251 / 0.264 / 0.264). No L=336 `test_100pct`; 10% is not in cells.

ETTh1 binary-anchor is Killarney dedicated-H eval-anchor jobs **5474030/31/33/34** (n=1838, iTrans encoder, configs `…nopretrain_itrans_hz{96,192,336,720}`; not clip-from-720). Binary-prob n=10 was not run. Solar binary / PatchTST solar: see notes above. Weather binary is Killarney **5489026–45** 20/20 `PIPELINE COMPLETE`, window-weighted n=9820 (unique windows 0–9819, no overlap; each shard n=491). pMSE 0.149 / 0.189 / 0.241 / 0.316, aMSE 0.152 / 0.192 / 0.244 / 0.318 (n_samp=10; aMSE is the n=1 deterministic decode on the same windows). Old 10% / old-protocol: aMSE job 2954294 n=982 n_samp=1 0.149/0.194/0.249/0.322; pMSE job 2650285 n=9820 0.152/0.194/0.244/0.321. Equal-weight shard mean of H720 pMSE is 0.316 (same as window-weighted because shards are equal-sized); babysit “shard mean 0.318” was H720 aMSE, not a 4-horizon row. Electricity binary-anchor is n=1 eval-anchor (n=4541); binary-prob is n=10 efficient-pack 100% (Killarney 5471563–5471572, window-weighted n=4541; pMSE 0.118 / 0.136 / 0.155 / 0.188, CRPS 0.169 / 0.183 / 0.198 / 0.220). PeMS binary H12/24/48/96 are clip prefixes from the H=96 model (job 2954248, n_samp=1, n=3304, aMSE 0.1019 / 0.1476 / 0.2403 / 0.4038; table 0.102 / 0.148 / 0.240 / 0.404). Binary-prob is older n_samp=10, same clip. PatchTST PeMS is Vast L=96 `test_100pct` (`seq_len=96`, `eval_test_fraction=0`) from the fair run 19 Sep: H12 mse 0.100746 (table 0.101), H24 0.160688 (0.161), H48 0.294152 (0.294), H96 0.510390 (0.510). Old Killarney L=512 eval-only jobs **5572811/13/14** (0.086 / 0.164 / 0.196) and failed H24 **5572812** are not in the cells. iTransformer / MMPD PeMS rows are still dedicated-H 11 Sep fullT 100%. iTransformer H12 is 0.095 from job 2952243 `test_100pct`. PatchTST ETT / weather / electricity / traffic H96/H192/H720 are Killarney L=336 `test_100pct` (`eval_test_fraction=0`). Traffic H336 job **5608062** resumed from epoch 75 and stopped at 10% test (mse 0.399); no `test_100pct`, so the cell stays L=512 0.420. Exchange PatchTST / iTransformer / MMPD are Killarney PWB+linear L=336 (not 11 Sep from-scratch); binary exchange is still with-pretrain dedicated-H. Illness PatchTST / iTransformer / MMPD are rented-L40S PWB+linear L=104 (not L=148 scratch / not Killarney 5555984–87 scratch / not 11 Sep); MMPD-anchor is half-test n=85/79/73/67 (of 170/158/146/134). Binary illness H24/36/48: Killarney **5557476–78** unique-seg n=170/158/146 n_samp=10, aMSE 2.2487 / 2.6657 / 2.2514, pMSE 2.3203 / 2.5197 / 2.1344. H60: **5548243** aMSE 2.0575 n_win=134 n_samp=0; Binary-prob **5557547** pMSE 2.0563 n_samp=10. Dynamic is native H 60/120/180/300: MMPD 11 Sep L=600 dedicated-H; binary L=600 H=300 n=333 n_samp=10. PatchTST/iTransformer have no cells on that grid (their 96/192/336/720 numbers are not shown). Binary H 60/120/180 were not evaluated.

## Missing and failed cells

PatchTST traffic H336 100% did not finish (job **5608062**, `eval_test_fraction=0.1`, root mse 0.399, empty `eval_subsets`). That cell stays L=512 0.420. PatchTST solar L=336 100% is not on Killarney; cells stay L=512 100% and are marked. Elec H96 and traffic H720 L=336 `test_100pct` are filled (5560059 mse 0.130; 5588235 mse 0.457). PatchTST PeMS L=96 `test_100pct` is filled on all four horizons (Vast fair, 19 Sep). Old Killarney L=512 H24 **5572812** still failed on a stale `PeMS.csv` handle; that leftover and jobs **5572811/13/14** are not in the cells. Binary dynamic H 60/120/180 were not evaluated; 2874036 failed compile OOM (resub **2977741** pending). Solar binary 100% of96 is filled (n=9792; leftover shards **5440840–99 / 5440921–26** on ckpt **2650286**). Old of10 n≈3988 and 7v T=30k aMSE 1.6559 are not in cells. Electricity binary-prob n=10 100% is filled (Killarney 5471563–5471572); Binary-anchor stays the n=1 eval-anchor numbers. PeMS binary H12 / 24 / 48 are clip prefixes from the H=96 eval (no dedicated-H yaml). Illness MMPD is half-test n=85/79/73/67 (of 170/158/146/134), not 100%. Narval electricity fullT **2954292** train finished then CUDA OOM in staged_eval (not in this table).

## Sources

- PatchTST: Killarney `$SCRATCH/ts-sandbox/temp/baselines_canvas128_subset/results/hz{H}_lb336/patchtst/<dataset>/patchtst_summary.json` field `eval_subsets.test_100pct.mse`, log `eval_test_fraction=0`. ETTh1 jobs **5554643 / 5555672 / 5555673 / 5555602** → 0.385031 / 0.413864 / 0.440842 / 0.456226 (table 0.385 / 0.414 / 0.441 / 0.456). ETTh2 **5554644 / 5555603 / 5555604 / 5555605** → 0.274875 / 0.338509 / 0.367271 / 0.391571 (0.275 / 0.339 / 0.367 / 0.392). ETTm1 **5554646 / 5555979 / 5555980 / 5555981** → 0.291859 / 0.333594 / 0.365539 / 0.418647 (0.292 / 0.334 / 0.366 / 0.419). ETTm2 **5555982 / 5555610 / 5555611 / 5555983** → 0.164781 / 0.220680 / 0.276961 / 0.368942 (0.165 / 0.221 / 0.277 / 0.369). Weather eval-only **5588228–31** → 0.149855 / 0.195429 / 0.247722 / 0.321253 (0.150 / 0.195 / 0.248 / 0.321). Electricity **5560059 / 5560060 / 5555627 / 5555628** → 0.129903 / 0.149670 / 0.166330 / 0.210417 (0.130 / 0.150 / 0.166 / 0.210). Traffic **5608060 / 5608061** H96/H192 (resumed epoch 65) → 0.367337 / 0.386973 (0.367 / 0.387); H720 **5588235** → 0.457198 (0.457); H336 **5608062** (resumed epoch 75) is 10% only (0.399), not in the cell. The L=336 cells above replace the old `hz*_lb512/` numbers. Traffic H336 and solar stay L=512 `test_100pct`, marked in the table. PeMS is Vast L=96 `results/hz{12,24,48,96}_lb96/patchtst/PeMS/patchtst_summary.json` `test_100pct.mse` 0.100746 / 0.160688 / 0.294152 / 0.510390 (table 0.101 / 0.161 / 0.294 / 0.510; fair log `temp/baselines_canvas128_subset/logs/pems_patchtst_fair_all.log`, done 19 Sep 2026 23:16:58 UTC). Old Killarney L=512 jobs **5572811 / 5572813 / 5572814** (0.086 / 0.164 / 0.196) and failed H24 **5572812** are leftovers, not in cells. Solar PatchTST cells are the L=512 `test_100pct` column (0.214 / 0.230 / 0.246 / 0.239), marked L=512. No Killarney `hz{H}_lb336/patchtst/solar_Alabama/`. Prior L-match (Vast) root `mse` 0.228 / 0.251 / 0.264 / 0.264 was `eval_test_fraction=0.1` with empty `eval_subsets` (not in cells). Exchange PatchTST is `hz{H}_lb336_pwb_linear/patchtst/exchange_rate/patchtst_summary.json` `test_100pct.mse` 0.088683 / 0.185948 / 0.351673 / 0.920222 (table 0.089 / 0.186 / 0.352 / 0.920; jobs **5441136 / 5445488 / 5445490 / 5445492**). Illness PatchTST is rented-L40S `hz{H}_lb104_pwb_linear/patchtst/illness/patchtst_summary.json` `test_100pct.mse` 1.883 / 1.875 / 1.616 / 1.738 (not L=148 scratch; not Killarney scratch `hz*_lb104` 1.816 / 2.099 / 1.736 / 1.662; not Killarney 5555984–87).
- iTransformer: `$SCRATCH/ts-sandbox-fullT/temp/baselines_canvas128_subset/results/hz*/itransformer/*/…_summary.json` field `eval_subsets.test_100pct.mse` (root `mse` is the 10% split; not used). Exchange iTransformer is Killarney `hz{H}_lb336_pwb_linear/itransformer/exchange_rate/itransformer_summary.json` `test_100pct.mse`. Illness iTransformer is rented-L40S `hz{H}_lb104_pwb_linear/itransformer/illness/itransformer_summary.json` `test_100pct.mse` 2.136 / 2.181 / 2.396 / 2.497 (not 11 Sep from-scratch).
- MMPD: `$SCRATCH/ts-sandbox-fullT/results/datasets/09-11-mmpd-fullT-*/partials/*_mmpd.json` field `anchor_mse_test_100pct`. Exchange MMPD is Killarney `$SCRATCH/ts-sandbox/results/datasets/hz{H}_lb336_pwb_linear/partials/exchange_rate_mmpd.json` root `anchor_mse` (`anchor_mse_test_100pct` absent; jobs 5445493–5501). Illness MMPD is rented-L40S `results/datasets/hz{H}_lb104_pwb_linear/metrics.json` `illness.mmpd.anchor_mse` 2.193 / 3.185 / 3.575 / 2.288; half-test n=85/79/73/67 (of 170/158/146/134), not 100%.
- Binary ETTh1: Killarney **5474030 / 5474031 / 5474033 / 5474034** (`PIPELINE COMPLETE`, eval-anchor `n_samp=0`), configs `binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_cap1x2x_nopretrain_itrans_hz{96,192,336,720}`, dedicated H (not clip-from-720), n=1838, iTrans encoder not patch-decoder. Log `staged eval done` aMSE **0.4251 / 0.4472 / 0.4555 / 0.5295** (table 0.425 / 0.447 / 0.455 / 0.529). Binary-prob MSE cells stay dashed. Binary-prob CRPS is the later n=10 `eval_resume/…nopre_itrans_hz{H}_s1_n10_steps4` jsonl n=1838 (0.311 / 0.330 / 0.343 / 0.380).
- Binary ETTh2 / ETTm1 / ETTm2: Narval `$SCRATCH/ts-sandbox/results/datasets/09-08-26729*-*fullT_hz720*kv_a100/partials/*_fulltest.json` (n=2161 / 10801).
- Binary weather: Killarney **5489026–45** (`PIPELINE COMPLETE`), config `…p32x6_weather_fullT_hz720_nostitch_nopretrain_hpsearch_frac50_refit_fulltest_effpack`, `eval_resume/canvas128_p32x6_weather_hz720_hpsearch_frac50_refit_s1_n10_steps4_fulltest_shard{0–19}of20/weather_allv_s1_full/summary.json` + `windows.jsonl`. Window-weighted mean of per-window MSE (mean over windows×variates×horizon; prefix keys `mse_h96/_h192/_h336` + unsuffixed `mse` = H720; same for `anchor_mse*`). n=9820, n_samp=10, 20 equal shards of 491. Unrounded pMSE 0.14877 / 0.18927 / 0.24057 / 0.31566; aMSE 0.15234 / 0.19218 / 0.24352 / 0.31780. Old 10% aMSE Narval **2954294** n=982 n_samp=1 and old pMSE **2650285** n=9820 kept as protocol leftovers only.
- Binary electricity-anchor: 12 eval-anchor shards 2831625–2831641, n_windows-weighted (sum 4541).
- Binary electricity-prob: Killarney n=10 efficient-pack shards **5471563–5471572** (10/10 `PIPELINE COMPLETE`), `eval_resume/..._fulltest_effpack_shard{0–9}of10/electricity_allv_s1_full/summary.json`, n_windows-weighted (sum **4541**/4541). pMSE `mse_h96/_h192/_h336` + unsuffixed `mse` (H720); CRPS same pattern. wandb `ts-sandbox-leaderboard` runs 19y08s88 / ez2syc45 / jgdkyey9 / wwz2bdg4 / g5sow05y / q70f9pis / 60iiwakq / f57q9cyy / xysxccfq / 3fmzrh4l (per-shard; table uses the jsonl merge, not one shard).
- Binary traffic: wandb `eval/staged_anchor_mse{_h96,_h192,_h336}` on jobs 5356912–5356928 (17/17 shards; equal-weight mean, shards sized for n=2789).
- Binary PeMS H96: eval-anchor **2954248** on ckpt 2673014, aMSE 0.4038 (table 0.404), n=3304, n_samp=1. Binary-prob 0.398 is older n_samp=10 100% (n=3304).
- Binary dynamic: Narval 10/10 fulltest shards, L=600 native H=300, n=333, n_samp=10. H 60/120/180 not evaluated.
- MMPD dynamic: 11 Sep L=600 dedicated-H `09-11-mmpd-fullT-lb600-hz{60,120,180,300}` `anchor_mse_test_100pct` (H=300 job 2874118/19).
- Binary solar: Killarney of96 100% **resume from of10** on ckpt **2650286**, local merge `temp/in_vs_error/solar_of96_jsonl/canvas128_p32x6_fullT_hz720_r0_ms35_kv_a100_s1_n10_steps4_fulltest_solar_shard{0–95}of96/solar_Alabama_allv_s1_full/windows.jsonl` (96/96 shards, 102 windows each, unique `window_index` 0–9791, n=9792, n_samp=10). Equal-window mean; H96/192/336 = prefix keys `anchor_mse_h*` / `mse_h*`; H720 = unsuffixed `anchor_mse` / `mse`. Unrounded aMSE 0.1892465730 / 0.2026233283 / 0.2119834312 / 0.2208550479; pMSE 0.1812980346 / 0.1951222774 / 0.2042704540 / 0.2119882270 (table 0.189 / 0.203 / 0.212 / 0.221 and 0.181 / 0.195 / 0.204 / 0.212). of96 reused cancelled of10 windows; leftover jobs filled the rest (66 shards needed leftover work). Old of10 n=3988 scattered prefixes over 0–9791 (cancelled 2801390–2801425; cells 0.182 / 0.188 / 0.189 / 0.224) not in cells. 7v T=30k Vast A100 `results/vast-a100-solar-pull/datasets/09-20-vast-solar_Alabama-…T30k…/partials/solar_Alabama_staged_s1_anchor.json` `anchor_mse` **1.6559** n=5905 is a different split — not in cells.
- Binary exchange H96: with-pretrain cap1x2x; aMSE 0.095 / pMSE 0.086 (job 5057766).
- Binary exchange H192/336/720: Narval resume 2952251–53 of 2890531–33, staged eval `anchor_mse` / `mse` 0.1766/0.1753, 0.3596/0.3368, 0.9116/0.9153.
- Binary illness: Killarney U=217 PWB+linear dedicated-H. H24/36/48 **5557476–78** unique-seg n=170/158/146 n_samp=10, log `staged eval done` aMSE 2.2487 / 2.6657 / 2.2514 pMSE 2.3203 / 2.5197 / 2.1344. H60 **5548243** n_samp=0 n_win=134 aMSE 2.0575 (table 2.058); Binary-prob **5557547** n_samp=10 pMSE 2.0563 (aMSE 2.0575). Old Narval 2952185–87 / H60 5057771 2.478 not in cells.
- PeMS iTransformer H12: `hz12/itransformer/PeMS/itransformer_summary.json` `test_100pct.mse` 0.0954 (job 2952243).

