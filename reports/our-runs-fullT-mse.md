# Our runs — MSE vs PatchTST / iTransformer / MMPD

File-of-record numbers from Narval `$SCRATCH/ts-sandbox-fullT` (11 Sep fullT iTransformer / MMPD) plus Killarney PatchTST/64 (L=512) and completed binary 100% evals. **Not paper tables.** Bold is lowest MSE among Binary-anchor, PatchTST, iTransformer, and MMPD-anchor. Binary-prob is shown but not ranked.

Tags: our runs only · test 100% · MSE ↓ · pulled 14 Sep 2026 · elec n=10 16 Sep 2026 · ETTh1 itrans 16 Sep 2026

Canvas: [our-runs-fullT-mse.canvas.tsx](/home/cao/.cursor/projects/home-cao-ts-sandbox/canvases/our-runs-fullT-mse.canvas.tsx)

## Protocol (not matched across columns)

PatchTST is Killarney L=512 (illness L=148, PeMS L=512) dedicated-H `test_100pct` (overwrites the 11 Sep L=336 /42 column). iTransformer / MMPD are still 11 Sep fullT L=336 except illness L=104, PeMS L=96, dynamic L=600. Exchange PatchTST, iTransformer, and MMPD-anchor cells are Killarney PWB+linear synth pretrain then real finetune (L=336), not 11 Sep from-scratch. Binary exchange is still with-pretrain dedicated-H. Binary exchange and illness are cap1x2x with-pretrain, dedicated H, n_samp=10 (jobs 5057766 / 5057771 / 2952251–53 / 2952185–87). Binary electricity-anchor is prior clip-from-720 100% eval-anchor (n=1, n_win=4541, Narval 2831625–2831641); binary-prob is Killarney n=10 efficient-pack 100% (jobs 5471563–5471572, n_win=4541). Binary traffic is prior clip-from-720 100% eval-anchor (n=2789). Binary ETTh1 is Killarney dedicated-H iTrans-encoder eval-anchor (jobs 5474030/31/33/34, n=1838, configs `…nopretrain_itrans_hz{96,192,336,720}`); n=10 pMSE was not run. Binary ETTh2 / ETTm1 / ETTm2 are older n_samp=10 100% clip-from-max-H. Binary dynamic is L=600 native H=300 (n=333, n_samp=10); H 60/120/180 binary were not evaluated. Binary weather is mixed n_samp like PeMS: anchor is Narval **2954294** clip-from-720 100% (n=982, n_samp=1, prefixes H96/192/336); binary-prob is older n_samp=10 (job **2650285**, n=9820). Binary PeMS H96 is native L=96 (eval-anchor 2954248 on ckpt 2673014, n=3304, n_samp=1, aMSE 0.4038; table 0.404). Binary-prob 0.398 is older n_samp=10. H12 / 24 / 48 binary never trained. Solar binary is a pooled partial (3988/9792 windows from cancelled 10 Sep shards), not a finished 100% eval. Do not read a column win as the same recipe.


| Column                         | Train                                             | Eval                          | Notes                                                                                                                             |
| ------------------------------ | ------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| PatchTST                       | Killarney L=512 uncapped (illness L=148)          | `test_100pct` MSE             | Dedicated H. PeMS L=512 with `test_max_windows=50`. Elec H96 / traffic H720 100% not in (dash). Exchange is `hz*_lb336_pwb_linear`, not L=512. |
| iTransformer                   | 11 Sep fullT uncapped                             | `test_100pct` MSE             | Dedicated H. L=336 except illness L=104, PeMS L=96. Exchange is PWB+linear, not 11 Sep.                                            |
| MMPD-anchor                    | 11 Sep fullT uncapped                             | `anchor_mse_test_100pct`      | Same H/L grid except dynamic (L=600, H in 60/120/180/300). Exchange is PWB+linear `anchor_mse` (no `anchor_mse_test_100pct`).     |
| Binary exchange / illness      | cap1x2x **with-pretrain**, dedicated H, n_samp=10 | staged anchor / prob MSE      | Exchange H96 job 5057766; H192/336/720 jobs 2952251–53. Illness H60 job 5057771; H24/36/48 jobs 2952185–87                        |
| Binary electricity / traffic   | prior uncapped fullT, clip-from-720               | 100% eval-anchor + elec n=10  | Elec aMSE n=1 Narval 2831625–2831641; elec pMSE/CRPS n=10 Killarney 5471563–72 (n_win=4541). Traffic n=2789, Killarney 5356912–5356928 |
| Binary ETTh1                   | 5057773 HPs except `guidance_type: itransformer`, dedicated H, nopre | eval-anchor n=1838            | Jobs 5474030/31/33/34; configs `…nopretrain_itrans_hz{96,192,336,720}`. iTrans encoder, not decoder. pMSE not run.               |
| Binary ETTh2 / m1 / m2         | older n_samp=10 100% clip-from-max-H              | staged MSE                    | Solar binary is **partial 3988/9792**, not 100%                                                                                   |
| Binary dynamic                 | L=600 native H=300, n_samp=10 100%                | staged MSE                    | n=333. H 60/120/180 never evaluated. PatchTST/iTrans have **no** 60/120/180/300 cells                                             |
| Binary weather                 | p32 fullT fixed-HP, clip-from-720                 | 100% staged mixed n_samp      | Anchor: job **2954294** (ckpt 2874032), n=982, n_samp=1. Binary-prob is older n_samp=10 100% (job **2650285**, n=9820). Prefix aMSE/pMSE H96/192/336; unsuffixed = H720 |
| Binary PeMS                    | native L=96 H=96                                  | 100% **eval-anchor** n_samp=1 | Job **2954248** on ckpt 2673014, n=3304, aMSE 0.4038 (table 0.404). Binary-prob 0.398 is older n_samp=10. H12/24/48 never trained |




## Ranked-row wins


| PatchTST | MMPD-anchor | Binary-anchor | iTransformer |
| -------- | ----------- | ------------- | ------------ |
| 22       | 18          | 8             | 2            |


48 ranked horizon rows (ties counted for each winner). Avg rows are per-dataset means of numeric cells and are not in that tally. Dataset-avg winners: PatchTST 4, MMPD-anchor 5, Binary-anchor 2, iTransformer 1. Binary-prob is omitted from the rank. Exchange/illness binary is with-pretrain dedicated-H.

## MSE by dataset and horizon

Source: Killarney PatchTST/64 under `temp/baselines_canvas128_subset/results/hz*_lb512/` (illness `hz*_lb148/`) field `eval_subsets.test_100pct.mse` when present; iTransformer still 11 Sep Narval `hz*/itransformer`; MMPD `results/datasets/09-11-mmpd-fullT-*/partials` (`anchor_mse_test_100pct`); exchange PatchTST / iTransformer / MMPD from Killarney `hz{H}_lb336_pwb_linear/` (`test_100pct.mse`; MMPD root `anchor_mse` — no `anchor_mse_test_100pct`); binary from staged_eval JSON / wandb shards as footnoted. Avg is the mean of numeric cells in that dataset block (3 dp; dashes skipped). Dynamic uses native H 60/120/180/300 (not the 96/192/336/720 grid).


| Dataset     | H   | Binary-anchor | Binary-prob | PatchTST  | iTransformer | MMPD-anchor |
| ----------- | --- | ------------- | ----------- | --------- | ------------ | ----------- |
| ETTh1       | 96  | 0.425         | —           | 0.374     | 0.406        | **0.372**   |
| ETTh1       | 192 | 0.447         | —           | 0.413     | 0.447        | **0.404**   |
| ETTh1       | 336 | 0.455         | —           | **0.434** | 0.471        | **0.434**   |
| ETTh1       | 720 | 0.529         | —           | 0.455     | 0.599        | **0.443**   |
| ETTh1       | Avg | 0.464         | —           | 0.419     | 0.481        | **0.413**   |
| ETTh2       | 96  | 0.325         | 0.323       | **0.274** | 0.327        | 0.287       |
| ETTh2       | 192 | 0.343         | 0.342       | **0.341** | 0.416        | 0.357       |
| ETTh2       | 336 | **0.349**     | 0.350       | 0.365     | 0.440        | 0.366       |
| ETTh2       | 720 | 0.432         | 0.438       | **0.390** | 0.440        | 0.403       |
| ETTh2       | Avg | 0.362         | 0.363       | **0.343** | 0.406        | 0.353       |
| ETTm1       | 96  | 0.312         | 0.291       | 0.290     | 0.318        | **0.279**   |
| ETTm1       | 192 | 0.339         | 0.322       | 0.334     | 0.359        | **0.313**   |
| ETTm1       | 336 | 0.371         | 0.356       | 0.370     | 0.386        | **0.351**   |
| ETTm1       | 720 | 0.427         | 0.412       | 0.416     | 0.450        | **0.404**   |
| ETTm1       | Avg | 0.362         | 0.345       | 0.353     | 0.378        | **0.337**   |
| ETTm2       | 96  | 0.185         | 0.181       | **0.166** | 0.193        | 0.182       |
| ETTm2       | 192 | 0.243         | 0.241       | **0.222** | 0.251        | 0.232       |
| ETTm2       | 336 | 0.296         | 0.296       | **0.273** | 0.302        | 0.290       |
| ETTm2       | 720 | 0.385         | 0.387       | **0.363** | 0.395        | 0.375       |
| ETTm2       | Avg | 0.277         | 0.276       | **0.256** | 0.285        | 0.270       |
| weather     | 96  | 0.149         | 0.152 n=10  | **0.148** | 0.161        | 0.149       |
| weather     | 192 | 0.194         | 0.194 n=10  | **0.193** | 0.206        | **0.193**   |
| weather     | 336 | 0.249         | 0.244 n=10  | **0.244** | 0.255        | 0.246       |
| weather     | 720 | 0.322         | 0.321 n=10  | **0.314** | 0.327        | 0.319       |
| weather     | Avg | 0.229         | 0.228       | **0.225** | 0.237        | 0.227       |
| electricity | 96  | **0.127**     | 0.118 n=10  | —         | 0.135        | 0.134       |
| electricity | 192 | **0.145**     | 0.136 n=10  | 0.150     | 0.154        | 0.150       |
| electricity | 336 | 0.165         | 0.155 n=10  | **0.163** | 0.169        | 0.166       |
| electricity | 720 | 0.202         | 0.188 n=10  | **0.200** | 0.206        | 0.205       |
| electricity | Avg | **0.160**     | 0.149       | 0.171     | 0.166        | 0.164       |
| traffic     | 96  | 0.399         | —           | 0.398     | 0.397        | **0.376**   |
| traffic     | 192 | 0.409         | —           | 0.411     | 0.417        | **0.400**   |
| traffic     | 336 | 0.424         | —           | 0.420     | 0.433        | **0.405**   |
| traffic     | 720 | 0.476         | —           | —         | 0.467        | **0.440**   |
| traffic     | Avg | 0.427         | —           | 0.410     | 0.429        | **0.405**   |
| solar       | 96  | **0.182**     | 0.177       | 0.214     | 0.208        | 0.184       |
| solar       | 192 | **0.188**     | 0.184       | 0.230     | 0.227        | 0.189       |
| solar       | 336 | **0.189**     | 0.188       | 0.246     | 0.224        | 0.200       |
| solar       | 720 | 0.224         | 0.216       | 0.239     | 0.221        | **0.200**   |
| solar       | Avg | 0.196         | 0.191       | 0.232     | 0.220        | **0.193**   |
| exchange    | 96  | 0.095         | 0.086       | **0.089** | 0.096        | 0.095       |
| exchange    | 192 | **0.177**     | 0.175       | 0.186     | 0.192        | 0.206       |
| exchange    | 336 | 0.360         | 0.337       | **0.352** | 0.382        | 0.403       |
| exchange    | 720 | **0.912**     | 0.915       | 0.920     | 0.914        | 0.984       |
| exchange    | Avg | **0.386**     | 0.378       | 0.387     | 0.396        | 0.422       |
| illness     | 24  | 2.397         | 2.218       | **2.016** | 2.085        | 3.671       |
| illness     | 36  | 2.582         | 2.308       | 2.989     | **2.171**    | 3.567       |
| illness     | 48  | 2.117         | 1.897       | **1.712** | 2.124        | 3.824       |
| illness     | 60  | 2.478         | 2.225       | 2.236     | **2.165**    | 4.080       |
| illness     | Avg | 2.394         | 2.162       | 2.238     | **2.136**    | 3.786       |
| PeMS        | 12  | 0.102         | 0.099 n=10  | **0.088** | 0.095        | 0.105       |
| PeMS        | 24  | 0.148         | 0.144 n=10  | **0.117** | 0.143        | 0.170       |
| PeMS        | 48  | 0.240         | 0.236 n=10  | **0.152** | 0.243        | 0.311       |
| PeMS        | 96  | 0.404         | 0.398 n=10  | **0.175** | 0.400        | 0.527       |
| PeMS        | Avg | 0.224         | 0.219       | **0.133** | 0.220        | 0.278       |
| dynamic     | 60  | —             | —           | —         | —            | **0.221**   |
| dynamic     | 120 | —             | —           | —         | —            | **0.294**   |
| dynamic     | 180 | —             | —           | —         | —            | **0.345**   |
| dynamic     | 300 | 0.512         | 0.477       | —         | —            | **0.414**   |
| dynamic     | Avg | 0.512         | 0.477       | —         | —            | **0.319**   |


ETTh1 binary-anchor is Killarney dedicated-H eval-anchor jobs **5474030/31/33/34** (n=1838, iTrans encoder, configs `…nopretrain_itrans_hz{96,192,336,720}`; not clip-from-720). Binary-prob n=10 was not run. Solar binary is a pooled **partial 3988/9792** (cancelled 10 Sep shards), not a finished 100% eval. Weather binary-anchor is job 2954294 n_samp=1 (n=982); binary-prob is older n_samp=10 (job 2650285, n=9820). Electricity binary-anchor is n=1 eval-anchor (n=4541); binary-prob is n=10 efficient-pack 100% (Killarney 5471563–5471572, window-weighted n=4541; pMSE 0.118 / 0.136 / 0.155 / 0.188, CRPS 0.169 / 0.183 / 0.198 / 0.220). PeMS binary H12/24/48/96 are clip prefixes from the H=96 model (job 2954248, n_samp=1, n=3304, aMSE 0.1019 / 0.1476 / 0.2403 / 0.4038; table 0.102 / 0.148 / 0.240 / 0.404). Binary-prob is older n_samp=10, same clip. PatchTST PeMS is Killarney L=512 with `eval_test_fraction=1.0` and `test_max_windows=50` (not L=96 100%). iTransformer / MMPD PeMS rows are still dedicated-H 11 Sep fullT 100%. iTransformer H12 is 0.095 from job 2952243 `test_100pct`. PatchTST elec H96 and traffic H720 100% evals did not finish (10% / 10-window only); those cells are dashes. Exchange PatchTST / iTransformer / MMPD are Killarney PWB+linear L=336 (not 11 Sep from-scratch); binary exchange is still with-pretrain dedicated-H. Dynamic is native H 60/120/180/300: MMPD 11 Sep L=600 dedicated-H; binary L=600 H=300 n=333 n_samp=10. PatchTST/iTransformer have no cells on that grid (their 96/192/336/720 numbers are not shown). Binary H 60/120/180 were not evaluated.

## Missing and failed cells

PatchTST elec H96 100% and traffic H720 100% did not finish on Killarney L=512 (elec H96 10% mse 0.117; traffic H720 10-window mse 0.443). Binary dynamic H 60/120/180 were not evaluated; 2874036 failed compile OOM (resub **2977741** pending). Binary solar 100% shards were cancelled; table uses pooled 3988/9792 completed windows (not 100%). Electricity binary-prob n=10 100% is filled (Killarney 5471563–5471572); Binary-anchor stays the n=1 eval-anchor numbers. PeMS binary H12 / 24 / 48 are clip prefixes from the H=96 eval (no dedicated-H yaml). Narval electricity fullT **2954292** train finished then CUDA OOM in staged_eval (not in this table).

## Sources

- PatchTST/64: Killarney `$SCRATCH/ts-sandbox/temp/baselines_canvas128_subset/results/hz*_lb512/patchtst/*/patchtst_summary.json` field `eval_subsets.test_100pct.mse` (illness `hz*_lb148/`). Root `mse` is the 10% split when `test_100pct` is missing; those cells are dashed (elec H96, traffic H720). Exchange PatchTST is `hz{H}_lb336_pwb_linear/patchtst/exchange_rate/patchtst_summary.json` `test_100pct.mse` (jobs 5441135/36/56 H96; 5445487–92 H192–720).
- iTransformer: `$SCRATCH/ts-sandbox-fullT/temp/baselines_canvas128_subset/results/hz*/itransformer/*/…_summary.json` field `eval_subsets.test_100pct.mse` (root `mse` is the 10% split; not used). Exchange iTransformer is Killarney `hz{H}_lb336_pwb_linear/itransformer/exchange_rate/itransformer_summary.json` `test_100pct.mse`.
- MMPD: `$SCRATCH/ts-sandbox-fullT/results/datasets/09-11-mmpd-fullT-*/partials/*_mmpd.json` field `anchor_mse_test_100pct`. Exchange MMPD is Killarney `$SCRATCH/ts-sandbox/results/datasets/hz{H}_lb336_pwb_linear/partials/exchange_rate_mmpd.json` root `anchor_mse` (`anchor_mse_test_100pct` absent; jobs 5445493–5501).
- Binary ETTh1: Killarney **5474030 / 5474031 / 5474033 / 5474034** (`PIPELINE COMPLETE`, eval-anchor `n_samp=0`), configs `binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_cap1x2x_nopretrain_itrans_hz{96,192,336,720}`, dedicated H (not clip-from-720), n=1838, iTrans encoder not patch-decoder. Log `staged eval done` aMSE **0.4251 / 0.4472 / 0.4555 / 0.5295** (table 0.425 / 0.447 / 0.455 / 0.529). Binary-prob dashed — n=10 was not run (`prob_mse=nan`).
- Binary ETTh2 / ETTm1 / ETTm2: Narval `$SCRATCH/ts-sandbox/results/datasets/09-08-26729*-*fullT_hz720*kv_a100/partials/*_fulltest.json` (n=2161 / 10801).
- Binary weather-anchor: Narval **2954294** `$SCRATCH/ts-sandbox-fullT/results/datasets/09-11-2874032-weather-…/partials/weather_staged_s1_prob.json` n=982 n_samp=1; `anchor_mse` = H720, `anchor_mse_h96/_h192/_h336` prefixes.
- Binary weather-prob: Narval **2650285** `$SCRATCH/ts-sandbox/results/datasets/09-08-2650285-weather-…/partials/weather_staged_s1_prob_fulltest.json` n=9820 (n_samp=10); `mse` = H720, `mse_h96/_h192/_h336` prefixes.
- Binary electricity-anchor: 12 eval-anchor shards 2831625–2831641, n_windows-weighted (sum 4541).
- Binary electricity-prob: Killarney n=10 efficient-pack shards **5471563–5471572** (10/10 `PIPELINE COMPLETE`), `eval_resume/..._fulltest_effpack_shard{0–9}of10/electricity_allv_s1_full/summary.json`, n_windows-weighted (sum **4541**/4541). pMSE `mse_h96/_h192/_h336` + unsuffixed `mse` (H720); CRPS same pattern. wandb `ts-sandbox-leaderboard` runs 19y08s88 / ez2syc45 / jgdkyey9 / wwz2bdg4 / g5sow05y / q70f9pis / 60iiwakq / f57q9cyy / xysxccfq / 3fmzrh4l (per-shard; table uses the jsonl merge, not one shard).
- Binary traffic: wandb `eval/staged_anchor_mse{_h96,_h192,_h336}` on jobs 5356912–5356928 (17/17 shards; equal-weight mean, shards sized for n=2789).
- Binary PeMS H96: eval-anchor **2954248** on ckpt 2673014, aMSE 0.4038 (table 0.404), n=3304, n_samp=1. Binary-prob 0.398 is older n_samp=10 100% (n=3304).
- Binary dynamic: Narval 10/10 fulltest shards, L=600 native H=300, n=333, n_samp=10. H 60/120/180 not evaluated.
- MMPD dynamic: 11 Sep L=600 dedicated-H `09-11-mmpd-fullT-lb600-hz{60,120,180,300}` `anchor_mse_test_100pct` (H=300 job 2874118/19).
- Binary solar: cancelled 10 Sep 10-shard 100% prob eval (jobs 2801390–2801425, clip-from-720, L=336, n_samp=10). Pooled **3988** unique windows from `eval_resume/..._fulltest_solar_shard{0–9}of10/solar_Alabama_allv_s1_full/windows.jsonl` (equal-window mean; shards uneven so not mean-of-shard-means). H96/192/336 = prefix keys; H720 = unsuffixed `anchor_mse` / `mse` (native H=720). Summary `n_done` summed to 3987 (shard3 summary lagged jsonl by 1).
- Binary exchange H96 / illness H60: with-pretrain cap1x2x; aMSE 0.095 / 2.478 and pMSE 0.086 / 2.225 (job 5057766 / 5057771).
- Binary exchange H192/336/720: Narval resume 2952251–53 of 2890531–33, staged eval `anchor_mse` / `mse` 0.1766/0.1753, 0.3596/0.3368, 0.9116/0.9153.
- Binary illness H24/36/48: Narval 2952185–87, aMSE/pMSE 2.3970/2.2180, 2.5823/2.3082, 2.1166/1.8973.
- PeMS iTransformer H12: `hz12/itransformer/PeMS/itransformer_summary.json` `test_100pct.mse` 0.0954 (job 2952243).

