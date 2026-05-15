# Run report: 05-14-1304-joint-ft-ETTh2-gB

- Job ID: 14
- Log: results/logs/05-14-1304-joint-ft-ETTh2-gB.log
- Duration: 0h 57m 4s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581304  task=finetune  ghost=gB  log: ./results/logs/05-14-1304-joint-ft-ETTh2-gB.log
Node: kn060  started: 2026-05-14T17:37:46-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=ETTh2 n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:38:00,027 - INFO - traffic.csv already exists
2026-05-14 17:38:00,127 - INFO - ============================================================
2026-05-14 17:38:00,127 - INFO - Joint finetune (e2e): subset=ETTh2, dim=7, epochs=10, trials=3
2026-05-14 17:38:00,127 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gB.pt
2026-05-14 17:38:00,127 - INFO - ============================================================
2026-05-14 17:38:00,628 - INFO - [joint_finetune_ETTh2_gB] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:38:00,829 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:38:00,829 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:38:00,884 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:38:00,963 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:38:00,963 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:38:00,963 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:38:00,987 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:38:00,989 - INFO - DiffusionTSF initialized:
2026-05-14 17:38:00,989 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:38:00,989 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:38:00,989 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:38:01,015 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:38:14,647 - INFO - [warmup] epoch 1/10 | train loss 1.4618 (noise 0.0000, aux 1.4618) | val 1.6068 | lr d=1.16e-04 i=4.35e-04 | 10.2s
2026-05-14 17:42:01,660 - INFO - [joint] epoch 2/10 | train loss 1.4300 (noise 0.0243, aux 1.3642) | val 1.6272 | lr d=1.07e-04 i=4.04e-04 | 207.4s
2026-05-14 17:45:47,774 - INFO - [joint] epoch 3/10 | train loss 1.3469 (noise 0.0178, aux 1.2923) | val 1.7129 | lr d=9.43e-05 i=3.55e-04 | 206.4s
2026-05-14 17:49:33,855 - INFO - [joint] epoch 4/10 | train loss 1.2448 (noise 0.0160, aux 1.1958) | val 1.6876 | lr d=7.79e-05 i=2.93e-04 | 206.4s
2026-05-14 17:53:19,878 - INFO - [joint] epoch 5/10 | train loss 1.1266 (noise 0.0140, aux 1.0814) | val 1.6376 | lr d=5.98e-05 i=2.24e-04 | 206.4s
2026-05-14 17:57:05,754 - INFO - [joint] epoch 6/10 | train loss 1.0017 (noise 0.0130, aux 0.9602) | val 1.7425 | lr d=4.17e-05 i=1.55e-04 | 206.2s
2026-05-14 18:00:51,802 - INFO - [joint] epoch 7/10 | train loss 0.8814 (noise 0.0118, aux 0.8429) | val 1.7142 | lr d=2.54e-05 i=9.29e-05 | 206.4s
2026-05-14 18:00:51,803 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:00:52,065 - INFO - [joint_finetune_ETTh2_gB] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 18:00:52,116 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 18:00:52,116 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:00:52,116 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:00:52,182 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:00:52,182 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:00:52,182 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:00:52,184 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:00:52,185 - INFO - DiffusionTSF initialized:
2026-05-14 18:00:52,185 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:00:52,185 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:00:52,185 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:00:52,207 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 18:01:01,069 - INFO - [warmup] epoch 1/10 | train loss 1.4126 (noise 0.0000, aux 1.4126) | val 1.5562 | lr d=2.63e-04 i=1.94e-04 | 8.0s
2026-05-14 18:04:46,952 - INFO - [joint] epoch 2/10 | train loss 1.3088 (noise 0.0278, aux 1.2373) | val 1.7263 | lr d=2.44e-04 i=1.80e-04 | 206.2s
2026-05-14 18:08:32,976 - INFO - [joint] epoch 3/10 | train loss 1.1329 (noise 0.0175, aux 1.0799) | val 1.7309 | lr d=2.15e-04 i=1.58e-04 | 206.3s
2026-05-14 18:12:18,907 - INFO - [joint] epoch 4/10 | train loss 1.0003 (noise 0.0154, aux 0.9529) | val 1.6948 | lr d=1.77e-04 i=1.31e-04 | 206.3s
2026-05-14 18:16:04,855 - INFO - [joint] epoch 5/10 | train loss 0.8954 (noise 0.0136, aux 0.8522) | val 1.7464 | lr d=1.36e-04 i=1.00e-04 | 206.2s
2026-05-14 18:19:50,857 - INFO - [joint] epoch 6/10 | train loss 0.8099 (noise 0.0113, aux 0.7713) | val 1.7435 | lr d=9.45e-05 i=6.99e-05 | 206.4s
2026-05-14 18:23:36,728 - INFO - [joint] epoch 7/10 | train loss 0.7448 (noise 0.0106, aux 0.7100) | val 1.7728 | lr d=5.72e-05 i=4.25e-05 | 206.2s
2026-05-14 18:27:22,730 - INFO - [joint] epoch 8/10 | train loss 0.6993 (noise 0.0099, aux 0.6671) | val 1.7367 | lr d=2.76e-05 i=2.07e-05 | 206.3s
2026-05-14 18:31:08,608 - INFO - [joint] epoch 9/10 | train loss 0.6679 (noise 0.0096, aux 0.6385) | val 1.7612 | lr d=8.54e-06 i=6.79e-06 | 206.2s
2026-05-14 18:31:08,608 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 18:31:08,877 - INFO - [joint_finetune_ETTh2_gB] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 18:31:08,938 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 18:31:08,939 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:31:08,939 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:31:09,013 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:31:09,013 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:31:09,013 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:31:09,016 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:31:09,016 - INFO - DiffusionTSF initialized:
2026-05-14 18:31:09,016 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:31:09,016 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:31:09,016 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:31:09,036 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 18:31:18,584 - INFO - [warmup] epoch 1/10 | train loss 1.4222 (noise 0.0000, aux 1.4222) | val 1.5330 | lr d=6.99e-05 i=6.99e-05 | 8.7s
2026-05-14 18:35:04,429 - INFO - [joint] epoch 2/10 | train loss 1.3411 (noise 0.0230, aux 1.2785) | val 1.6769 | lr d=6.48e-05 i=6.48e-05 | 206.2s
2026-05-14 18:35:04,612 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 18:35:04,613 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:35:04,613 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:35:04,679 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:35:04,679 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:35:04,679 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:35:04,681 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:35:04,681 - INFO - DiffusionTSF initialized:
2026-05-14 18:35:04,681 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:35:04,682 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:35:04,682 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:35:04,833 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTh2_joint_finetuned_gB.pt (val=1.6272, epoch=2)
Done (worker).
```
