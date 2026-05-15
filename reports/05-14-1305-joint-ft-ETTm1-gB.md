# Run report: 05-14-1305-joint-ft-ETTm1-gB

- Job ID: 14
- Log: results/logs/05-14-1305-joint-ft-ETTm1-gB.log
- Duration: 3h 39m 41s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581305  task=finetune  ghost=gB  log: ./results/logs/05-14-1305-joint-ft-ETTm1-gB.log
Node: kn059  started: 2026-05-14T17:38:19-04:00
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
2026-05-14 17:38:30,659 - INFO - traffic.csv already exists
2026-05-14 17:38:30,744 - INFO - ============================================================
2026-05-14 17:38:30,744 - INFO - Joint finetune (e2e): subset=ETTm1, dim=7, epochs=10, trials=3
2026-05-14 17:38:30,744 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gB.pt
2026-05-14 17:38:30,745 - INFO - ============================================================
2026-05-14 17:38:31,220 - INFO - [joint_finetune_ETTm1_gB] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:38:31,356 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:38:31,356 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:38:31,357 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:38:31,428 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:38:31,428 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:38:31,428 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:38:31,430 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:38:31,431 - INFO - DiffusionTSF initialized:
2026-05-14 17:38:31,431 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:38:31,431 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:38:31,431 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:38:31,451 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:39:58,619 - INFO - [warmup] epoch 1/10 | train loss 1.5268 (noise 0.0000, aux 1.5268) | val 1.4121 | lr d=1.16e-04 i=4.35e-04 | 81.5s
2026-05-14 17:55:26,489 - INFO - [joint] epoch 2/10 | train loss 1.4998 (noise 0.0170, aux 1.4476) | val 1.3943 | lr d=1.07e-04 i=4.04e-04 | 846.5s
2026-05-14 18:10:50,365 - INFO - [joint] epoch 3/10 | train loss 1.4583 (noise 0.0119, aux 1.4177) | val 1.3816 | lr d=9.43e-05 i=3.55e-04 | 843.0s
2026-05-14 18:26:13,881 - INFO - [joint] epoch 4/10 | train loss 1.4271 (noise 0.0110, aux 1.3904) | val 1.4041 | lr d=7.79e-05 i=2.93e-04 | 842.7s
2026-05-14 18:41:37,697 - INFO - [joint] epoch 5/10 | train loss 1.3825 (noise 0.0098, aux 1.3494) | val 1.4010 | lr d=5.98e-05 i=2.24e-04 | 843.1s
2026-05-14 18:57:02,107 - INFO - [joint] epoch 6/10 | train loss 1.3048 (noise 0.0091, aux 1.2744) | val 1.3979 | lr d=4.17e-05 i=1.55e-04 | 843.3s
2026-05-14 19:12:24,886 - INFO - [joint] epoch 7/10 | train loss 1.2156 (noise 0.0083, aux 1.1884) | val 1.3841 | lr d=2.54e-05 i=9.29e-05 | 841.7s
2026-05-14 19:27:47,814 - INFO - [joint] epoch 8/10 | train loss 1.1194 (noise 0.0079, aux 1.0950) | val 1.3867 | lr d=1.24e-05 i=4.37e-05 | 841.7s
2026-05-14 19:27:47,814 - INFO - Early stopping at epoch 8 (patience=5)
2026-05-14 19:27:48,051 - INFO - [joint_finetune_ETTm1_gB] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 19:27:48,117 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 19:27:48,118 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 19:27:48,118 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 19:27:48,196 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 19:27:48,196 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 19:27:48,196 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 19:27:48,198 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 19:27:48,198 - INFO - DiffusionTSF initialized:
2026-05-14 19:27:48,198 - INFO -   Variables: 7 (multivariate)
2026-05-14 19:27:48,198 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 19:27:48,198 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 19:27:48,219 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 19:29:06,541 - INFO - [warmup] epoch 1/10 | train loss 1.4212 (noise 0.0000, aux 1.4212) | val 1.3673 | lr d=2.63e-04 i=1.94e-04 | 74.9s
2026-05-14 19:44:29,710 - INFO - [joint] epoch 2/10 | train loss 1.2512 (noise 0.0178, aux 1.1986) | val 1.4075 | lr d=2.44e-04 i=1.80e-04 | 842.4s
2026-05-14 19:59:52,716 - INFO - [joint] epoch 3/10 | train loss 1.0528 (noise 0.0113, aux 1.0143) | val 1.4662 | lr d=2.15e-04 i=1.58e-04 | 842.1s
2026-05-14 20:15:16,404 - INFO - [joint] epoch 4/10 | train loss 0.8840 (noise 0.0096, aux 0.8504) | val 1.4606 | lr d=1.77e-04 i=1.31e-04 | 842.7s
2026-05-14 20:30:40,379 - INFO - [joint] epoch 5/10 | train loss 0.7476 (noise 0.0085, aux 0.7179) | val 1.5261 | lr d=1.36e-04 i=1.00e-04 | 843.0s
2026-05-14 20:46:04,214 - INFO - [joint] epoch 6/10 | train loss 0.6445 (noise 0.0084, aux 0.6167) | val 1.5310 | lr d=9.45e-05 i=6.99e-05 | 842.9s
2026-05-14 21:01:28,603 - INFO - [joint] epoch 7/10 | train loss 0.5694 (noise 0.0075, aux 0.5449) | val 1.4869 | lr d=5.72e-05 i=4.25e-05 | 843.5s
2026-05-14 21:01:28,605 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 21:01:28,848 - INFO - [joint_finetune_ETTm1_gB] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 21:01:28,898 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 21:01:28,898 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:01:28,898 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:01:28,962 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:01:28,962 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:01:28,962 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:01:28,964 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:01:28,964 - INFO - DiffusionTSF initialized:
2026-05-14 21:01:28,964 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:01:28,964 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:01:28,964 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:01:28,984 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 21:02:47,339 - INFO - [warmup] epoch 1/10 | train loss 1.4365 (noise 0.0000, aux 1.4365) | val 1.3454 | lr d=6.99e-05 i=6.99e-05 | 74.9s
2026-05-14 21:18:10,643 - INFO - [joint] epoch 2/10 | train loss 1.2902 (noise 0.0170, aux 1.2384) | val 1.4307 | lr d=6.48e-05 i=6.48e-05 | 842.4s
2026-05-14 21:18:10,812 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 21:18:10,812 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:18:10,812 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:18:10,877 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:18:10,877 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:18:10,877 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:18:10,879 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:18:10,879 - INFO - DiffusionTSF initialized:
2026-05-14 21:18:10,879 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:18:10,879 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:18:10,879 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:18:11,048 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTm1_joint_finetuned_gB.pt (val=1.3816, epoch=3)
Done (worker).
```
