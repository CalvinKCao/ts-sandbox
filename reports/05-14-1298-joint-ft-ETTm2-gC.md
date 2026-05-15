# Run report: 05-14-1298-joint-ft-ETTm2-gC

- Job ID: 14
- Log: results/logs/05-14-1298-joint-ft-ETTm2-gC.log
- Duration: 3h 55m 32s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581298  task=finetune  ghost=gC  log: ./results/logs/05-14-1298-joint-ft-ETTm2-gC.log
Node: kn051  started: 2026-05-14T17:51:29-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=ETTm2 n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:51:37,636 - INFO - traffic.csv already exists
2026-05-14 17:51:37,715 - INFO - ============================================================
2026-05-14 17:51:37,715 - INFO - Joint finetune (e2e): subset=ETTm2, dim=7, epochs=10, trials=3
2026-05-14 17:51:37,715 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gC.pt
2026-05-14 17:51:37,716 - INFO - ============================================================
2026-05-14 17:51:38,161 - INFO - [joint_finetune_ETTm2_gC] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:51:38,240 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:51:38,240 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:51:38,241 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:51:38,311 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:51:38,311 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:51:38,311 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:51:38,313 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:51:38,313 - INFO - DiffusionTSF initialized:
2026-05-14 17:51:38,313 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:51:38,313 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:51:38,313 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:51:38,334 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:52:58,171 - INFO - [warmup] epoch 1/10 | train loss 1.4146 (noise 0.0000, aux 1.4146) | val 1.4847 | lr d=1.16e-04 i=4.35e-04 | 75.2s
2026-05-14 18:08:24,712 - INFO - [joint] epoch 2/10 | train loss 1.4042 (noise 0.0169, aux 1.3524) | val 1.5273 | lr d=1.07e-04 i=4.04e-04 | 845.6s
2026-05-14 18:23:48,925 - INFO - [joint] epoch 3/10 | train loss 1.3754 (noise 0.0128, aux 1.3336) | val 1.5242 | lr d=9.43e-05 i=3.55e-04 | 843.3s
2026-05-14 18:39:14,855 - INFO - [joint] epoch 4/10 | train loss 1.3549 (noise 0.0115, aux 1.3179) | val 1.5166 | lr d=7.79e-05 i=2.93e-04 | 844.8s
2026-05-14 18:54:42,585 - INFO - [joint] epoch 5/10 | train loss 1.3282 (noise 0.0105, aux 1.2941) | val 1.5303 | lr d=5.98e-05 i=2.24e-04 | 846.6s
2026-05-14 19:10:10,020 - INFO - [joint] epoch 6/10 | train loss 1.2848 (noise 0.0095, aux 1.2534) | val 1.5306 | lr d=4.17e-05 i=1.55e-04 | 846.1s
2026-05-14 19:25:37,325 - INFO - [joint] epoch 7/10 | train loss 1.2193 (noise 0.0089, aux 1.1914) | val 1.5643 | lr d=2.54e-05 i=9.29e-05 | 846.2s
2026-05-14 19:41:04,167 - INFO - [joint] epoch 8/10 | train loss 1.1473 (noise 0.0082, aux 1.1223) | val 1.5564 | lr d=1.24e-05 i=4.37e-05 | 845.9s
2026-05-14 19:56:31,509 - INFO - [joint] epoch 9/10 | train loss 1.0856 (noise 0.0075, aux 1.0625) | val 1.5806 | lr d=4.05e-06 i=1.21e-05 | 846.3s
2026-05-14 19:56:31,510 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 19:56:31,763 - INFO - [joint_finetune_ETTm2_gC] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 19:56:31,814 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 19:56:31,814 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 19:56:31,815 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 19:56:31,882 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 19:56:31,882 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 19:56:31,882 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 19:56:31,885 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 19:56:31,885 - INFO - DiffusionTSF initialized:
2026-05-14 19:56:31,885 - INFO -   Variables: 7 (multivariate)
2026-05-14 19:56:31,885 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 19:56:31,885 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 19:56:31,906 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 19:57:50,531 - INFO - [warmup] epoch 1/10 | train loss 1.3548 (noise 0.0000, aux 1.3548) | val 1.4906 | lr d=2.63e-04 i=1.94e-04 | 75.1s
2026-05-14 20:13:15,416 - INFO - [joint] epoch 2/10 | train loss 1.2736 (noise 0.0177, aux 1.2205) | val 1.5509 | lr d=2.44e-04 i=1.80e-04 | 844.1s
2026-05-14 20:28:39,497 - INFO - [joint] epoch 3/10 | train loss 1.1424 (noise 0.0124, aux 1.1007) | val 1.6091 | lr d=2.15e-04 i=1.58e-04 | 843.2s
2026-05-14 20:44:03,354 - INFO - [joint] epoch 4/10 | train loss 1.0150 (noise 0.0106, aux 0.9786) | val 1.6389 | lr d=1.77e-04 i=1.31e-04 | 843.1s
2026-05-14 20:59:27,469 - INFO - [joint] epoch 5/10 | train loss 0.8976 (noise 0.0092, aux 0.8655) | val 1.6592 | lr d=1.36e-04 i=1.00e-04 | 843.3s
2026-05-14 21:14:54,830 - INFO - [joint] epoch 6/10 | train loss 0.7964 (noise 0.0086, aux 0.7674) | val 1.6140 | lr d=9.45e-05 i=6.99e-05 | 846.2s
2026-05-14 21:30:22,359 - INFO - [joint] epoch 7/10 | train loss 0.7172 (noise 0.0074, aux 0.6921) | val 1.6368 | lr d=5.72e-05 i=4.25e-05 | 846.4s
2026-05-14 21:30:22,360 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 21:30:22,616 - INFO - [joint_finetune_ETTm2_gC] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 21:30:22,666 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 21:30:22,666 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:30:22,667 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:30:22,733 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:30:22,733 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:30:22,733 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:30:22,736 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:30:22,736 - INFO - DiffusionTSF initialized:
2026-05-14 21:30:22,736 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:30:22,736 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:30:22,736 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:30:22,757 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 21:31:41,724 - INFO - [warmup] epoch 1/10 | train loss 1.3542 (noise 0.0000, aux 1.3542) | val 1.4920 | lr d=6.99e-05 i=6.99e-05 | 75.4s
2026-05-14 21:47:09,324 - INFO - [joint] epoch 2/10 | train loss 1.2774 (noise 0.0166, aux 1.2259) | val 1.5872 | lr d=6.48e-05 i=6.48e-05 | 846.5s
2026-05-14 21:47:09,490 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 21:47:09,490 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:47:09,491 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:47:09,556 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:47:09,557 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:47:09,557 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:47:09,559 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:47:09,559 - INFO - DiffusionTSF initialized:
2026-05-14 21:47:09,559 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:47:09,559 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:47:09,559 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:47:09,708 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTm2_joint_finetuned_gC.pt (val=1.5166, epoch=4)
Done (worker).
```
