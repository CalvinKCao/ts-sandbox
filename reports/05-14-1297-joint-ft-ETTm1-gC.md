# Run report: 05-14-1297-joint-ft-ETTm1-gC

- Job ID: 14
- Log: results/logs/05-14-1297-joint-ft-ETTm1-gC.log
- Duration: 5h 26m 22s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581297  task=finetune  ghost=gC  log: ./results/logs/05-14-1297-joint-ft-ETTm1-gC.log
Node: kn077  started: 2026-05-14T17:45:55-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=ETTm1 n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:46:06,947 - INFO - traffic.csv already exists
2026-05-14 17:46:07,031 - INFO - ============================================================
2026-05-14 17:46:07,031 - INFO - Joint finetune (e2e): subset=ETTm1, dim=7, epochs=10, trials=3
2026-05-14 17:46:07,032 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gC.pt
2026-05-14 17:46:07,032 - INFO - ============================================================
2026-05-14 17:46:07,613 - INFO - [joint_finetune_ETTm1_gC] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:46:07,786 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:46:07,787 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:46:07,838 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:46:07,935 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:46:07,935 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:46:07,935 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:46:07,958 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:46:07,960 - INFO - DiffusionTSF initialized:
2026-05-14 17:46:07,960 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:46:07,960 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:46:07,960 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:46:07,990 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:47:32,459 - INFO - [warmup] epoch 1/10 | train loss 1.5288 (noise 0.0000, aux 1.5288) | val 1.3367 | lr d=1.16e-04 i=4.35e-04 | 78.7s
2026-05-14 18:02:53,658 - INFO - [joint] epoch 2/10 | train loss 1.4986 (noise 0.0175, aux 1.4461) | val 1.3837 | lr d=1.07e-04 i=4.04e-04 | 840.8s
2026-05-14 18:18:14,374 - INFO - [joint] epoch 3/10 | train loss 1.4705 (noise 0.0136, aux 1.4278) | val 1.3783 | lr d=9.43e-05 i=3.55e-04 | 840.1s
2026-05-14 18:33:34,805 - INFO - [joint] epoch 4/10 | train loss 1.4435 (noise 0.0126, aux 1.4052) | val 1.3894 | lr d=7.79e-05 i=2.93e-04 | 839.9s
2026-05-14 18:48:54,662 - INFO - [joint] epoch 5/10 | train loss 1.4044 (noise 0.0116, aux 1.3690) | val 1.3708 | lr d=5.98e-05 i=2.24e-04 | 839.5s
2026-05-14 19:04:14,390 - INFO - [joint] epoch 6/10 | train loss 1.3365 (noise 0.0109, aux 1.3038) | val 1.3709 | lr d=4.17e-05 i=1.55e-04 | 839.3s
2026-05-14 19:19:34,041 - INFO - [joint] epoch 7/10 | train loss 1.2483 (noise 0.0100, aux 1.2192) | val 1.3663 | lr d=2.54e-05 i=9.29e-05 | 839.3s
2026-05-14 19:34:54,419 - INFO - [joint] epoch 8/10 | train loss 1.1600 (noise 0.0092, aux 1.1338) | val 1.3641 | lr d=1.24e-05 i=4.37e-05 | 839.8s
2026-05-14 19:50:14,250 - INFO - [joint] epoch 9/10 | train loss 1.0849 (noise 0.0086, aux 1.0606) | val 1.3678 | lr d=4.05e-06 i=1.21e-05 | 839.4s
2026-05-14 20:05:34,313 - INFO - [joint] epoch 10/10 | train loss 1.0407 (noise 0.0084, aux 1.0176) | val 1.3749 | lr d=1.18e-06 i=1.18e-06 | 839.6s
2026-05-14 20:05:34,563 - INFO - [joint_finetune_ETTm1_gC] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 20:05:34,685 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 20:05:34,685 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 20:05:34,686 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 20:05:34,756 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 20:05:34,757 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 20:05:34,757 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 20:05:34,759 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 20:05:34,759 - INFO - DiffusionTSF initialized:
2026-05-14 20:05:34,759 - INFO -   Variables: 7 (multivariate)
2026-05-14 20:05:34,759 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 20:05:34,759 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 20:05:34,780 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 20:06:52,299 - INFO - [warmup] epoch 1/10 | train loss 1.4177 (noise 0.0000, aux 1.4177) | val 1.3639 | lr d=2.63e-04 i=1.94e-04 | 74.1s
2026-05-14 20:22:14,034 - INFO - [joint] epoch 2/10 | train loss 1.2539 (noise 0.0184, aux 1.1996) | val 1.4586 | lr d=2.44e-04 i=1.80e-04 | 841.1s
2026-05-14 20:37:35,161 - INFO - [joint] epoch 3/10 | train loss 1.0537 (noise 0.0130, aux 1.0118) | val 1.4794 | lr d=2.15e-04 i=1.58e-04 | 840.6s
2026-05-14 20:52:56,674 - INFO - [joint] epoch 4/10 | train loss 0.8861 (noise 0.0117, aux 0.8483) | val 1.5268 | lr d=1.77e-04 i=1.31e-04 | 841.0s
2026-05-14 21:08:18,353 - INFO - [joint] epoch 5/10 | train loss 0.7472 (noise 0.0105, aux 0.7140) | val 1.5036 | lr d=1.36e-04 i=1.00e-04 | 841.0s
2026-05-14 21:23:40,518 - INFO - [joint] epoch 6/10 | train loss 0.6431 (noise 0.0091, aux 0.6141) | val 1.5233 | lr d=9.45e-05 i=6.99e-05 | 841.5s
2026-05-14 21:39:02,814 - INFO - [joint] epoch 7/10 | train loss 0.5698 (noise 0.0085, aux 0.5439) | val 1.4913 | lr d=5.72e-05 i=4.25e-05 | 841.7s
2026-05-14 21:39:02,815 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 21:39:03,150 - INFO - [joint_finetune_ETTm1_gC] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 21:39:03,201 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 21:39:03,202 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:39:03,202 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:39:03,272 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:39:03,272 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:39:03,272 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:39:03,274 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:39:03,274 - INFO - DiffusionTSF initialized:
2026-05-14 21:39:03,274 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:39:03,274 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:39:03,274 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:39:03,294 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 21:40:19,914 - INFO - [warmup] epoch 1/10 | train loss 1.4374 (noise 0.0000, aux 1.4374) | val 1.3466 | lr d=6.99e-05 i=6.99e-05 | 73.2s
2026-05-14 21:55:42,037 - INFO - [joint] epoch 2/10 | train loss 1.2893 (noise 0.0171, aux 1.2373) | val 1.4153 | lr d=6.48e-05 i=6.48e-05 | 841.6s
2026-05-14 22:11:03,716 - INFO - [joint] epoch 3/10 | train loss 1.1349 (noise 0.0136, aux 1.0925) | val 1.4636 | lr d=5.70e-05 i=5.70e-05 | 841.1s
2026-05-14 22:26:24,960 - INFO - [joint] epoch 4/10 | train loss 1.0096 (noise 0.0126, aux 0.9711) | val 1.4950 | lr d=4.71e-05 i=4.71e-05 | 840.8s
2026-05-14 22:41:45,686 - INFO - [joint] epoch 5/10 | train loss 0.9079 (noise 0.0121, aux 0.8728) | val 1.5247 | lr d=3.62e-05 i=3.62e-05 | 840.3s
2026-05-14 22:57:06,713 - INFO - [joint] epoch 6/10 | train loss 0.8332 (noise 0.0111, aux 0.8007) | val 1.5045 | lr d=2.52e-05 i=2.52e-05 | 840.4s
2026-05-14 23:12:28,305 - INFO - [joint] epoch 7/10 | train loss 0.7759 (noise 0.0108, aux 0.7460) | val 1.4999 | lr d=1.53e-05 i=1.53e-05 | 841.0s
2026-05-14 23:12:28,307 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 23:12:28,607 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 23:12:28,608 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 23:12:28,608 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 23:12:28,675 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 23:12:28,675 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 23:12:28,675 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 23:12:28,677 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 23:12:28,677 - INFO - DiffusionTSF initialized:
2026-05-14 23:12:28,678 - INFO -   Variables: 7 (multivariate)
2026-05-14 23:12:28,678 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 23:12:28,678 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 23:12:28,825 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTm1_joint_finetuned_gC.pt (val=1.3641, epoch=8)
Done (worker).
```
