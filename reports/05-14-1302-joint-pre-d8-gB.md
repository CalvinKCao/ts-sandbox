# Run report: 05-14-1302-joint-pre-d8-gB

- Job ID: 14
- Log: results/logs/05-14-1302-joint-pre-d8-gB.log
- Duration: 0h 36m 33s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```2026-05-14 16:30:17,068 - INFO - traffic.csv already exists
2026-05-14 16:30:17,141 - INFO - ============================================================
2026-05-14 16:30:17,141 - INFO - Joint pretrain (e2e): dim=8, epochs=15 (warmup=1), trials=4
2026-05-14 16:30:17,141 - INFO - aux_weight=1.00 | ghost=B
2026-05-14 16:30:17,141 - INFO - ============================================================
2026-05-14 16:30:17,143 - WARNING - RealTS: calculated pool_size 55714 exceeds SYNTHETIC_SAMPLES_CAP 50000. Capping and disabling epoch-stride for safety.
2026-05-14 16:30:17,143 - INFO - RealTS initialized: 4096 samples/epoch, lookback=96, forecast=96, variables=8, pool_rows=50000, epoch_stride=False (train_n=3687, val_tail=409, cap=15)
2026-05-14 16:30:17,150 - INFO - Reusing existing synthetic pool /scratch/ccao87/ts-sandbox/synth_data/synth_pool_v8_L192.npy (has 50000 samples, need 50000)
2026-05-14 16:30:17,150 - INFO - Created synthetic-only dataloader: 4096 samples/epoch (Pool: 4096), lookback=96, forecast=96, variables=8
2026-05-14 16:30:17,151 - INFO - [joint_pretrain_dim8_gB] Trial 1/4: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 16:30:17,377 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 16:30:17,378 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:30:17,379 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:30:17,448 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:30:17,449 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:30:17,449 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:30:17,451 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:30:17,451 - INFO - DiffusionTSF initialized:
2026-05-14 16:30:17,451 - INFO -   Variables: 8 (multivariate)
2026-05-14 16:30:17,451 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:30:17,451 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:30:17,468 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:30:28,934 - INFO - [warmup] epoch 1/15 | train loss 2.3728 (noise 0.0000, aux 2.3728) | val 2.2970 | lr d=1.17e-04 i=4.41e-04 | 10.1s
2026-05-14 16:32:16,989 - INFO - [joint] epoch 2/15 | train loss 2.4892 (noise 0.1211, aux 2.3041) | val 2.3690 | lr d=1.13e-04 i=4.27e-04 | 104.7s
2026-05-14 16:34:04,502 - INFO - [joint] epoch 3/15 | train loss 2.3550 (noise 0.0378, aux 2.2695) | val 2.3477 | lr d=1.07e-04 i=4.04e-04 | 104.1s
2026-05-14 16:35:52,131 - INFO - [joint] epoch 4/15 | train loss 2.3008 (noise 0.0267, aux 2.2298) | val 2.3539 | lr d=9.90e-05 i=3.73e-04 | 104.2s
2026-05-14 16:37:39,607 - INFO - [joint] epoch 5/15 | train loss 2.2405 (noise 0.0232, aux 2.1776) | val 2.3501 | lr d=8.91e-05 i=3.35e-04 | 104.2s
2026-05-14 16:39:27,103 - INFO - [joint] epoch 6/15 | train loss 2.1589 (noise 0.0209, aux 2.0988) | val 2.3375 | lr d=7.79e-05 i=2.93e-04 | 104.2s
2026-05-14 16:41:14,593 - INFO - [joint] epoch 7/15 | train loss 2.0637 (noise 0.0187, aux 2.0072) | val 2.3748 | lr d=6.59e-05 i=2.47e-04 | 104.1s
2026-05-14 16:43:02,111 - INFO - [joint] epoch 8/15 | train loss 1.9468 (noise 0.0169, aux 1.8946) | val 2.4620 | lr d=5.37e-05 i=2.01e-04 | 104.2s
2026-05-14 16:44:49,546 - INFO - [joint] epoch 9/15 | train loss 1.8240 (noise 0.0161, aux 1.7751) | val 2.4492 | lr d=4.17e-05 i=1.55e-04 | 104.1s
2026-05-14 16:46:37,051 - INFO - [joint] epoch 10/15 | train loss 1.7077 (noise 0.0151, aux 1.6601) | val 2.5165 | lr d=3.05e-05 i=1.12e-04 | 104.2s
2026-05-14 16:48:24,453 - INFO - [joint] epoch 11/15 | train loss 1.5971 (noise 0.0139, aux 1.5517) | val 2.5333 | lr d=2.06e-05 i=7.48e-05 | 104.1s
2026-05-14 16:48:24,453 - INFO - Early stopping at epoch 11 (patience=5)
2026-05-14 16:48:24,710 - INFO - [joint_pretrain_dim8_gB] Trial 2/4: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 16:48:24,763 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 16:48:24,763 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:48:24,763 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:48:24,830 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:48:24,831 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:48:24,831 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:48:24,833 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:48:24,833 - INFO - DiffusionTSF initialized:
2026-05-14 16:48:24,833 - INFO -   Variables: 8 (multivariate)
2026-05-14 16:48:24,833 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:48:24,833 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:48:24,848 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:48:34,321 - INFO - [warmup] epoch 1/15 | train loss 2.3613 (noise 0.0000, aux 2.3613) | val 2.2830 | lr d=2.67e-04 i=1.96e-04 | 9.3s
2026-05-14 16:50:21,767 - INFO - [joint] epoch 2/15 | train loss 2.4297 (noise 0.0969, aux 2.2713) | val 2.3488 | lr d=2.58e-04 i=1.90e-04 | 104.1s
2026-05-14 16:52:09,659 - INFO - [joint] epoch 3/15 | train loss 2.2982 (noise 0.0295, aux 2.2242) | val 2.3430 | lr d=2.44e-04 i=1.80e-04 | 104.5s
2026-05-14 16:53:57,507 - INFO - [joint] epoch 4/15 | train loss 2.2342 (noise 0.0209, aux 2.1730) | val 2.3180 | lr d=2.25e-04 i=1.66e-04 | 104.4s
2026-05-14 16:55:45,172 - INFO - [joint] epoch 5/15 | train loss 2.1557 (noise 0.0172, aux 2.1006) | val 2.3547 | lr d=2.03e-04 i=1.49e-04 | 104.3s
2026-05-14 16:57:32,718 - INFO - [joint] epoch 6/15 | train loss 2.0632 (noise 0.0179, aux 2.0101) | val 2.4392 | lr d=1.77e-04 i=1.31e-04 | 104.2s
2026-05-14 16:59:20,332 - INFO - [joint] epoch 7/15 | train loss 1.9516 (noise 0.0151, aux 1.9029) | val 2.4809 | lr d=1.50e-04 i=1.10e-04 | 104.3s
2026-05-14 17:01:07,908 - INFO - [joint] epoch 8/15 | train loss 1.8499 (noise 0.0146, aux 1.8028) | val 2.5055 | lr d=1.22e-04 i=8.99e-05 | 104.2s
2026-05-14 17:02:55,545 - INFO - [joint] epoch 9/15 | train loss 1.7530 (noise 0.0133, aux 1.7097) | val 2.6410 | lr d=9.45e-05 i=6.99e-05 | 104.3s
2026-05-14 17:02:55,546 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 17:02:55,853 - INFO - [joint_pretrain_dim8_gB] Trial 3/4: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 17:02:55,908 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:02:55,908 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:02:55,908 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:02:55,980 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:02:55,980 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:02:55,980 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:02:55,983 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:02:55,983 - INFO - DiffusionTSF initialized:
2026-05-14 17:02:55,983 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:02:55,983 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:02:55,983 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:02:55,999 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:03:05,286 - INFO - [warmup] epoch 1/15 | train loss 2.4008 (noise 0.0000, aux 2.4008) | val 2.3103 | lr d=7.08e-05 i=7.08e-05 | 9.1s
2026-05-14 17:04:52,872 - INFO - [joint] epoch 2/15 | train loss 2.4945 (noise 0.1449, aux 2.2807) | val 2.3674 | lr d=6.85e-05 i=6.85e-05 | 104.3s
2026-05-14 17:04:53,024 - INFO - [joint_pretrain_dim8_gB] Trial 4/4: diffusion_lr=5.72e-05, itrans_lr=3.67e-04
2026-05-14 17:04:53,076 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:04:53,076 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:04:53,077 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:04:53,147 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:04:53,147 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:04:53,147 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:04:53,150 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:04:53,150 - INFO - DiffusionTSF initialized:
2026-05-14 17:04:53,150 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:04:53,150 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:04:53,150 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:04:53,163 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:05:02,767 - INFO - [warmup] epoch 1/15 | train loss 2.3612 (noise 0.0000, aux 2.3612) | val 2.3468 | lr d=5.65e-05 i=3.63e-04 | 9.4s
2026-05-14 17:06:50,395 - INFO - [joint] epoch 2/15 | train loss 2.5066 (noise 0.1451, aux 2.2912) | val 2.3630 | lr d=5.47e-05 i=3.52e-04 | 104.3s
2026-05-14 17:06:50,559 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:06:50,559 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:06:50,559 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:06:50,625 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:06:50,625 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:06:50,625 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:06:50,627 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:06:50,627 - INFO - DiffusionTSF initialized:
2026-05-14 17:06:50,627 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:06:50,627 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:06:50,628 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:06:50,786 - INFO - Joint pretrain checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim8/joint_pretrained_gB.pt (val=2.3180, epoch=4)
Done (worker).
```
