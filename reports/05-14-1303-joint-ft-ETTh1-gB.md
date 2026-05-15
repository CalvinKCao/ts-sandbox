# Run report: 05-14-1303-joint-ft-ETTh1-gB

- Job ID: 14
- Log: results/logs/05-14-1303-joint-ft-ETTh1-gB.log
- Duration: 0h 50m 9s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581303  task=finetune  ghost=gB  log: ./results/logs/05-14-1303-joint-ft-ETTh1-gB.log
Node: kn071  started: 2026-05-14T17:32:42-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=ETTh1 n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:32:56,360 - INFO - traffic.csv already exists
2026-05-14 17:32:56,448 - INFO - ============================================================
2026-05-14 17:32:56,449 - INFO - Joint finetune (e2e): subset=ETTh1, dim=7, epochs=10, trials=3
2026-05-14 17:32:56,449 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gB.pt
2026-05-14 17:32:56,449 - INFO - ============================================================
2026-05-14 17:32:56,940 - INFO - [joint_finetune_ETTh1_gB] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:32:57,140 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:32:57,141 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:32:57,197 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:32:57,292 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:32:57,292 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:32:57,292 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:32:57,316 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:32:57,318 - INFO - DiffusionTSF initialized:
2026-05-14 17:32:57,318 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:32:57,319 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:32:57,319 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:32:57,349 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:33:20,449 - INFO - [warmup] epoch 1/10 | train loss 1.2489 (noise 0.0000, aux 1.2489) | val 1.4957 | lr d=1.16e-04 i=4.35e-04 | 19.5s
2026-05-14 17:37:08,399 - INFO - [joint] epoch 2/10 | train loss 1.2302 (noise 0.0248, aux 1.1643) | val 1.5055 | lr d=1.07e-04 i=4.04e-04 | 208.1s
2026-05-14 17:40:55,774 - INFO - [joint] epoch 3/10 | train loss 1.1422 (noise 0.0184, aux 1.0871) | val 1.5888 | lr d=9.43e-05 i=3.55e-04 | 207.5s
2026-05-14 17:44:42,883 - INFO - [joint] epoch 4/10 | train loss 1.0424 (noise 0.0167, aux 0.9932) | val 1.5152 | lr d=7.79e-05 i=2.93e-04 | 207.4s
2026-05-14 17:48:30,106 - INFO - [joint] epoch 5/10 | train loss 0.9353 (noise 0.0144, aux 0.8905) | val 1.5359 | lr d=5.98e-05 i=2.24e-04 | 207.5s
2026-05-14 17:52:16,721 - INFO - [joint] epoch 6/10 | train loss 0.8133 (noise 0.0135, aux 0.7716) | val 1.5277 | lr d=4.17e-05 i=1.55e-04 | 206.9s
2026-05-14 17:56:02,974 - INFO - [joint] epoch 7/10 | train loss 0.7108 (noise 0.0126, aux 0.6716) | val 1.5324 | lr d=2.54e-05 i=9.29e-05 | 206.6s
2026-05-14 17:56:02,975 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:56:03,217 - INFO - [joint_finetune_ETTh1_gB] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 17:56:03,274 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:56:03,274 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:56:03,275 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:56:03,346 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:56:03,346 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:56:03,346 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:56:03,348 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:56:03,348 - INFO - DiffusionTSF initialized:
2026-05-14 17:56:03,348 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:56:03,348 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:56:03,348 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:56:03,371 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:56:22,230 - INFO - [warmup] epoch 1/10 | train loss 1.2168 (noise 0.0000, aux 1.2168) | val 1.4851 | lr d=2.63e-04 i=1.94e-04 | 18.0s
2026-05-14 18:00:08,448 - INFO - [joint] epoch 2/10 | train loss 1.1293 (noise 0.0282, aux 1.0577) | val 1.6000 | lr d=2.44e-04 i=1.80e-04 | 206.5s
2026-05-14 18:03:54,738 - INFO - [joint] epoch 3/10 | train loss 0.9794 (noise 0.0180, aux 0.9257) | val 1.6138 | lr d=2.15e-04 i=1.58e-04 | 206.5s
2026-05-14 18:07:40,949 - INFO - [joint] epoch 4/10 | train loss 0.8676 (noise 0.0159, aux 0.8195) | val 1.6251 | lr d=1.77e-04 i=1.31e-04 | 206.5s
2026-05-14 18:11:27,165 - INFO - [joint] epoch 5/10 | train loss 0.7696 (noise 0.0139, aux 0.7269) | val 1.6232 | lr d=1.36e-04 i=1.00e-04 | 206.5s
2026-05-14 18:15:13,349 - INFO - [joint] epoch 6/10 | train loss 0.6931 (noise 0.0125, aux 0.6532) | val 1.6173 | lr d=9.45e-05 i=6.99e-05 | 206.5s
2026-05-14 18:18:59,550 - INFO - [joint] epoch 7/10 | train loss 0.6336 (noise 0.0116, aux 0.5976) | val 1.6202 | lr d=5.72e-05 i=4.25e-05 | 206.5s
2026-05-14 18:18:59,551 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:18:59,788 - INFO - [joint_finetune_ETTh1_gB] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 18:18:59,844 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 18:18:59,845 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:18:59,845 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:18:59,908 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:18:59,909 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:18:59,909 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:18:59,911 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:18:59,911 - INFO - DiffusionTSF initialized:
2026-05-14 18:18:59,911 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:18:59,911 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:18:59,911 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:18:59,932 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 18:19:18,838 - INFO - [warmup] epoch 1/10 | train loss 1.2400 (noise 0.0000, aux 1.2400) | val 1.4357 | lr d=6.99e-05 i=6.99e-05 | 18.0s
2026-05-14 18:23:05,024 - INFO - [joint] epoch 2/10 | train loss 1.1688 (noise 0.0241, aux 1.1057) | val 1.5621 | lr d=6.48e-05 i=6.48e-05 | 206.5s
2026-05-14 18:23:05,185 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 18:23:05,186 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:23:05,186 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:23:05,248 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:23:05,248 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:23:05,248 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:23:05,251 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:23:05,251 - INFO - DiffusionTSF initialized:
2026-05-14 18:23:05,251 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:23:05,251 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:23:05,251 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:23:05,390 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTh1_joint_finetuned_gB.pt (val=1.5055, epoch=2)
Done (worker).
```
