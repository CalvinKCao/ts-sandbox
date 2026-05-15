# Run report: 05-14-1294-joint-pre-d8-gC

- Job ID: 14
- Log: results/logs/05-14-1294-joint-pre-d8-gC.log
- Duration: 0h 41m 42s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```2026-05-14 16:30:15,760 - INFO - ============================================================
2026-05-14 16:30:15,760 - WARNING - RealTS: calculated pool_size 55714 exceeds SYNTHETIC_SAMPLES_CAP 50000. Capping and disabling epoch-stride for safety.
2026-05-14 16:30:15,761 - INFO - RealTS initialized: 4096 samples/epoch, lookback=96, forecast=96, variables=8, pool_rows=50000, epoch_stride=False (train_n=3687, val_tail=409, cap=15)
2026-05-14 16:30:15,763 - INFO - Reusing existing synthetic pool /scratch/ccao87/ts-sandbox/synth_data/synth_pool_v8_L192.npy (has 50000 samples, need 50000)
2026-05-14 16:30:15,763 - INFO - Created synthetic-only dataloader: 4096 samples/epoch (Pool: 4096), lookback=96, forecast=96, variables=8
2026-05-14 16:30:15,765 - INFO - [joint_pretrain_dim8_gC] Trial 1/4: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 16:30:15,984 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 16:30:15,984 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:30:15,985 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:30:16,055 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:30:16,056 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:30:16,056 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:30:16,058 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:30:16,058 - INFO - DiffusionTSF initialized:
2026-05-14 16:30:16,058 - INFO -   Variables: 8 (multivariate)
2026-05-14 16:30:16,058 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:30:16,058 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:30:16,075 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:30:27,096 - INFO - [warmup] epoch 1/15 | train loss 2.3724 (noise 0.0000, aux 2.3724) | val 2.3157 | lr d=1.17e-04 i=4.41e-04 | 9.7s
2026-05-14 16:32:14,535 - INFO - [joint] epoch 2/15 | train loss 2.4713 (noise 0.1072, aux 2.3011) | val 2.3703 | lr d=1.13e-04 i=4.27e-04 | 104.1s
2026-05-14 16:34:01,313 - INFO - [joint] epoch 3/15 | train loss 2.3534 (noise 0.0355, aux 2.2705) | val 2.3649 | lr d=1.07e-04 i=4.04e-04 | 103.4s
2026-05-14 16:35:48,128 - INFO - [joint] epoch 4/15 | train loss 2.2996 (noise 0.0245, aux 2.2319) | val 2.3508 | lr d=9.90e-05 i=3.73e-04 | 103.5s
2026-05-14 16:37:35,037 - INFO - [joint] epoch 5/15 | train loss 2.2434 (noise 0.0222, aux 2.1821) | val 2.3634 | lr d=8.91e-05 i=3.35e-04 | 103.5s
2026-05-14 16:39:21,829 - INFO - [joint] epoch 6/15 | train loss 2.1633 (noise 0.0200, aux 2.1054) | val 2.3661 | lr d=7.79e-05 i=2.93e-04 | 103.5s
2026-05-14 16:41:08,502 - INFO - [joint] epoch 7/15 | train loss 2.0656 (noise 0.0187, aux 2.0096) | val 2.3789 | lr d=6.59e-05 i=2.47e-04 | 103.4s
2026-05-14 16:42:55,360 - INFO - [joint] epoch 8/15 | train loss 1.9499 (noise 0.0169, aux 1.8986) | val 2.4229 | lr d=5.37e-05 i=2.01e-04 | 103.6s
2026-05-14 16:44:42,247 - INFO - [joint] epoch 9/15 | train loss 1.8322 (noise 0.0164, aux 1.7835) | val 2.4523 | lr d=4.17e-05 i=1.55e-04 | 103.6s
2026-05-14 16:44:42,248 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 16:44:42,499 - INFO - [joint_pretrain_dim8_gC] Trial 2/4: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 16:44:42,553 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 16:44:42,554 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:44:42,554 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:44:42,618 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:44:42,618 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:44:42,618 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:44:42,620 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:44:42,620 - INFO - DiffusionTSF initialized:
2026-05-14 16:44:42,620 - INFO -   Variables: 8 (multivariate)
2026-05-14 16:44:42,620 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:44:42,620 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:44:42,636 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:44:52,139 - INFO - [warmup] epoch 1/15 | train loss 2.3604 (noise 0.0000, aux 2.3604) | val 2.2849 | lr d=2.67e-04 i=1.96e-04 | 9.3s
2026-05-14 16:46:39,064 - INFO - [joint] epoch 2/15 | train loss 2.4158 (noise 0.0904, aux 2.2667) | val 2.3400 | lr d=2.58e-04 i=1.90e-04 | 103.6s
2026-05-14 16:48:26,335 - INFO - [joint] epoch 3/15 | train loss 2.3016 (noise 0.0275, aux 2.2291) | val 2.3412 | lr d=2.44e-04 i=1.80e-04 | 103.9s
2026-05-14 16:50:13,169 - INFO - [joint] epoch 4/15 | train loss 2.2367 (noise 0.0199, aux 2.1746) | val 2.3719 | lr d=2.25e-04 i=1.66e-04 | 103.5s
2026-05-14 16:52:00,106 - INFO - [joint] epoch 5/15 | train loss 2.1592 (noise 0.0191, aux 2.1014) | val 2.4118 | lr d=2.03e-04 i=1.49e-04 | 103.6s
2026-05-14 16:53:46,900 - INFO - [joint] epoch 6/15 | train loss 2.0612 (noise 0.0176, aux 2.0087) | val 2.3927 | lr d=1.77e-04 i=1.31e-04 | 103.5s
2026-05-14 16:55:33,675 - INFO - [joint] epoch 7/15 | train loss 1.9500 (noise 0.0148, aux 1.9007) | val 2.4387 | lr d=1.50e-04 i=1.10e-04 | 103.5s
2026-05-14 16:55:33,676 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 16:55:33,929 - INFO - [joint_pretrain_dim8_gC] Trial 3/4: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 16:55:33,979 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 16:55:33,979 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:55:33,979 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:55:34,043 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:55:34,043 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:55:34,043 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:55:34,045 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:55:34,045 - INFO - DiffusionTSF initialized:
2026-05-14 16:55:34,045 - INFO -   Variables: 8 (multivariate)
2026-05-14 16:55:34,045 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:55:34,045 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:55:34,060 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:55:43,569 - INFO - [warmup] epoch 1/15 | train loss 2.4021 (noise 0.0000, aux 2.4021) | val 2.2865 | lr d=7.08e-05 i=7.08e-05 | 9.4s
2026-05-14 16:57:30,276 - INFO - [joint] epoch 2/15 | train loss 2.4703 (noise 0.1237, aux 2.2808) | val 2.3431 | lr d=6.85e-05 i=6.85e-05 | 103.4s
2026-05-14 16:59:17,112 - INFO - [joint] epoch 3/15 | train loss 2.3350 (noise 0.0422, aux 2.2420) | val 2.3421 | lr d=6.48e-05 i=6.48e-05 | 103.5s
2026-05-14 17:01:04,119 - INFO - [joint] epoch 4/15 | train loss 2.2889 (noise 0.0311, aux 2.2123) | val 2.3201 | lr d=5.99e-05 i=5.99e-05 | 103.6s
2026-05-14 17:02:51,165 - INFO - [joint] epoch 5/15 | train loss 2.2488 (noise 0.0249, aux 2.1802) | val 2.3325 | lr d=5.39e-05 i=5.39e-05 | 103.7s
2026-05-14 17:04:38,550 - INFO - [joint] epoch 6/15 | train loss 2.2051 (noise 0.0238, aux 2.1400) | val 2.3217 | lr d=4.71e-05 i=4.71e-05 | 104.1s
2026-05-14 17:06:25,928 - INFO - [joint] epoch 7/15 | train loss 2.1616 (noise 0.0224, aux 2.0981) | val 2.3786 | lr d=3.99e-05 i=3.99e-05 | 104.1s
2026-05-14 17:08:13,056 - INFO - [joint] epoch 8/15 | train loss 2.1104 (noise 0.0195, aux 2.0533) | val 2.3861 | lr d=3.25e-05 i=3.25e-05 | 103.8s
2026-05-14 17:10:00,041 - INFO - [joint] epoch 9/15 | train loss 2.0645 (noise 0.0184, aux 2.0096) | val 2.3816 | lr d=2.52e-05 i=2.52e-05 | 103.7s
2026-05-14 17:10:00,043 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 17:10:00,293 - INFO - [joint_pretrain_dim8_gC] Trial 4/4: diffusion_lr=5.72e-05, itrans_lr=3.67e-04
2026-05-14 17:10:00,349 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:10:00,349 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:10:00,349 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:10:00,417 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:10:00,417 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:10:00,417 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:10:00,419 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:10:00,420 - INFO - DiffusionTSF initialized:
2026-05-14 17:10:00,420 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:10:00,420 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:10:00,420 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:10:00,435 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:10:10,067 - INFO - [warmup] epoch 1/15 | train loss 2.3616 (noise 0.0000, aux 2.3616) | val 2.3477 | lr d=5.65e-05 i=3.63e-04 | 9.5s
2026-05-14 17:11:57,086 - INFO - [joint] epoch 2/15 | train loss 2.4920 (noise 0.1358, aux 2.2887) | val 2.4139 | lr d=5.47e-05 i=3.52e-04 | 103.7s
2026-05-14 17:11:57,264 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:11:57,265 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:11:57,265 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:11:57,329 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:11:57,329 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:11:57,329 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:11:57,332 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:11:57,332 - INFO - DiffusionTSF initialized:
2026-05-14 17:11:57,332 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:11:57,332 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:11:57,332 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:11:57,469 - INFO - Joint pretrain checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim8/joint_pretrained_gC.pt (val=2.3201, epoch=4)
Done (worker).
```
