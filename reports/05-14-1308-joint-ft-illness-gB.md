# Run report: 05-14-1308-joint-ft-illness-gB

- Job ID: 14
- Log: results/logs/05-14-1308-joint-ft-illness-gB.log
- Duration: 0h 3m 57s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581308  task=finetune  ghost=gB  log: ./results/logs/05-14-1308-joint-ft-illness-gB.log
Node: kn067  started: 2026-05-14T17:50:29-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=illness n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:50:43,113 - INFO - traffic.csv already exists
2026-05-14 17:50:43,199 - INFO - ============================================================
2026-05-14 17:50:43,199 - INFO - Joint finetune (e2e): subset=illness, dim=7, epochs=10, trials=3
2026-05-14 17:50:43,199 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gB.pt
2026-05-14 17:50:43,199 - INFO - ============================================================
2026-05-14 17:50:43,698 - INFO - [joint_finetune_illness_gB] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:50:43,889 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:50:43,889 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:50:43,950 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:50:44,051 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:50:44,051 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:50:44,051 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:50:44,075 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:50:44,077 - INFO - DiffusionTSF initialized:
2026-05-14 17:50:44,077 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:50:44,077 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:50:44,077 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:50:44,109 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:50:49,025 - INFO - [warmup] epoch 1/10 | train loss 0.8405 (noise 0.0000, aux 0.8405) | val 0.6175 | lr d=1.16e-04 i=4.35e-04 | 2.3s
2026-05-14 17:51:02,140 - INFO - [joint] epoch 2/10 | train loss 0.6898 (noise 0.0408, aux 0.5953) | val 0.6074 | lr d=1.07e-04 i=4.04e-04 | 13.1s
2026-05-14 17:51:14,191 - INFO - [joint] epoch 3/10 | train loss 0.5174 (noise 0.0242, aux 0.4533) | val 0.5186 | lr d=9.43e-05 i=3.55e-04 | 12.0s
2026-05-14 17:51:26,150 - INFO - [joint] epoch 4/10 | train loss 0.4229 (noise 0.0209, aux 0.3635) | val 0.4805 | lr d=7.79e-05 i=2.93e-04 | 11.9s
2026-05-14 17:51:38,113 - INFO - [joint] epoch 5/10 | train loss 0.3481 (noise 0.0179, aux 0.2923) | val 0.5872 | lr d=5.98e-05 i=2.24e-04 | 11.9s
2026-05-14 17:51:50,023 - INFO - [joint] epoch 6/10 | train loss 0.2856 (noise 0.0163, aux 0.2326) | val 0.6068 | lr d=4.17e-05 i=1.55e-04 | 11.9s
2026-05-14 17:52:01,919 - INFO - [joint] epoch 7/10 | train loss 0.2499 (noise 0.0150, aux 0.2010) | val 0.5304 | lr d=2.54e-05 i=9.29e-05 | 11.9s
2026-05-14 17:52:13,877 - INFO - [joint] epoch 8/10 | train loss 0.2191 (noise 0.0149, aux 0.1709) | val 0.6037 | lr d=1.24e-05 i=4.37e-05 | 11.9s
2026-05-14 17:52:25,842 - INFO - [joint] epoch 9/10 | train loss 0.2074 (noise 0.0156, aux 0.1584) | val 0.5781 | lr d=4.05e-06 i=1.21e-05 | 11.9s
2026-05-14 17:52:25,842 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 17:52:26,111 - INFO - [joint_finetune_illness_gB] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 17:52:26,160 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:52:26,161 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:52:26,161 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:52:26,224 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:52:26,224 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:52:26,224 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:52:26,226 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:52:26,227 - INFO - DiffusionTSF initialized:
2026-05-14 17:52:26,227 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:52:26,227 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:52:26,227 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:52:26,248 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:52:26,982 - INFO - [warmup] epoch 1/10 | train loss 0.7896 (noise 0.0000, aux 0.7896) | val 0.5630 | lr d=2.63e-04 i=1.94e-04 | 0.7s
2026-05-14 17:52:38,958 - INFO - [joint] epoch 2/10 | train loss 0.7151 (noise 0.1040, aux 0.5499) | val 0.5147 | lr d=2.44e-04 i=1.80e-04 | 12.0s
2026-05-14 17:52:50,976 - INFO - [joint] epoch 3/10 | train loss 0.5229 (noise 0.0263, aux 0.4511) | val 0.5762 | lr d=2.15e-04 i=1.58e-04 | 11.9s
2026-05-14 17:53:02,941 - INFO - [joint] epoch 4/10 | train loss 0.4557 (noise 0.0228, aux 0.3919) | val 0.4566 | lr d=1.77e-04 i=1.31e-04 | 11.9s
2026-05-14 17:53:14,976 - INFO - [joint] epoch 5/10 | train loss 0.4037 (noise 0.0246, aux 0.3419) | val 0.5374 | lr d=1.36e-04 i=1.00e-04 | 11.9s
2026-05-14 17:53:26,940 - INFO - [joint] epoch 6/10 | train loss 0.3648 (noise 0.0210, aux 0.3045) | val 0.5417 | lr d=9.45e-05 i=6.99e-05 | 11.9s
2026-05-14 17:53:38,903 - INFO - [joint] epoch 7/10 | train loss 0.3310 (noise 0.0159, aux 0.2765) | val 0.4979 | lr d=5.72e-05 i=4.25e-05 | 11.9s
2026-05-14 17:53:50,863 - INFO - [joint] epoch 8/10 | train loss 0.3059 (noise 0.0155, aux 0.2560) | val 0.4933 | lr d=2.76e-05 i=2.07e-05 | 11.9s
2026-05-14 17:54:02,828 - INFO - [joint] epoch 9/10 | train loss 0.2971 (noise 0.0186, aux 0.2438) | val 0.5031 | lr d=8.54e-06 i=6.79e-06 | 11.9s
2026-05-14 17:54:02,829 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 17:54:03,115 - INFO - [joint_finetune_illness_gB] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 17:54:03,164 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:54:03,164 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:54:03,164 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:54:03,230 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:54:03,230 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:54:03,230 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:54:03,232 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:54:03,232 - INFO - DiffusionTSF initialized:
2026-05-14 17:54:03,232 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:54:03,232 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:54:03,232 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:54:03,356 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:54:04,094 - INFO - [warmup] epoch 1/10 | train loss 0.8765 (noise 0.0000, aux 0.8765) | val 0.4010 | lr d=6.99e-05 i=6.99e-05 | 0.7s
2026-05-14 17:54:16,125 - INFO - [joint] epoch 2/10 | train loss 0.6701 (noise 0.0234, aux 0.5988) | val 0.5534 | lr d=6.48e-05 i=6.48e-05 | 12.0s
2026-05-14 17:54:28,113 - INFO - [joint] epoch 3/10 | train loss 0.5822 (noise 0.0230, aux 0.5238) | val 0.4815 | lr d=5.70e-05 i=5.70e-05 | 11.9s
2026-05-14 17:54:40,124 - INFO - [joint] epoch 4/10 | train loss 0.5332 (noise 0.0168, aux 0.4818) | val 0.5695 | lr d=4.71e-05 i=4.71e-05 | 11.9s
2026-05-14 17:54:40,296 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:54:40,296 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:54:40,296 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:54:40,360 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:54:40,360 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:54:40,360 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:54:40,362 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:54:40,363 - INFO - DiffusionTSF initialized:
2026-05-14 17:54:40,363 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:54:40,363 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:54:40,363 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:54:40,527 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/illness_joint_finetuned_gB.pt (val=0.4566, epoch=4)
Done (worker).
```
