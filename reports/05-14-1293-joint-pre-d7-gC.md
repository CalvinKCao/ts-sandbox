# Run report: 05-14-1293-joint-pre-d7-gC

- Job ID: 14
- Log: results/logs/05-14-1293-joint-pre-d7-gC.log
- Duration: 0h 42m 42s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```2026-05-14 16:29:46,954 - INFO - Created synthetic-only dataloader: 4096 samples/epoch (Pool: 4096), lookback=96, forecast=96, variables=7
2026-05-14 16:29:46,956 - INFO - [joint_pretrain_dim7_gC] Trial 1/4: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 16:29:47,239 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 16:29:47,239 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:29:47,240 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:29:47,310 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:29:47,310 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:29:47,310 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:29:47,312 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:29:47,312 - INFO - DiffusionTSF initialized:
2026-05-14 16:29:47,312 - INFO -   Variables: 7 (multivariate)
2026-05-14 16:29:47,312 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:29:47,312 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:29:47,330 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:29:59,651 - INFO - [warmup] epoch 1/15 | train loss 2.3599 (noise 0.0000, aux 2.3599) | val 2.3268 | lr d=1.17e-04 i=4.41e-04 | 9.7s
2026-05-14 16:31:33,527 - INFO - [joint] epoch 2/15 | train loss 2.4646 (noise 0.1114, aux 2.2892) | val 2.3924 | lr d=1.13e-04 i=4.27e-04 | 91.0s
2026-05-14 16:33:06,859 - INFO - [joint] epoch 3/15 | train loss 2.3413 (noise 0.0318, aux 2.2609) | val 2.3747 | lr d=1.07e-04 i=4.04e-04 | 90.4s
2026-05-14 16:34:40,386 - INFO - [joint] epoch 4/15 | train loss 2.2853 (noise 0.0240, aux 2.2174) | val 2.3600 | lr d=9.90e-05 i=3.73e-04 | 90.6s
2026-05-14 16:36:13,737 - INFO - [joint] epoch 5/15 | train loss 2.2385 (noise 0.0209, aux 2.1773) | val 2.3470 | lr d=8.91e-05 i=3.35e-04 | 90.4s
2026-05-14 16:37:47,042 - INFO - [joint] epoch 6/15 | train loss 2.1663 (noise 0.0198, aux 2.1086) | val 2.4068 | lr d=7.79e-05 i=2.93e-04 | 90.4s
2026-05-14 16:39:20,277 - INFO - [joint] epoch 7/15 | train loss 2.0690 (noise 0.0188, aux 2.0142) | val 2.3763 | lr d=6.59e-05 i=2.47e-04 | 90.3s
2026-05-14 16:40:53,659 - INFO - [joint] epoch 8/15 | train loss 1.9638 (noise 0.0164, aux 1.9127) | val 2.4383 | lr d=5.37e-05 i=2.01e-04 | 90.5s
2026-05-14 16:42:26,856 - INFO - [joint] epoch 9/15 | train loss 1.8522 (noise 0.0161, aux 1.8030) | val 2.4253 | lr d=4.17e-05 i=1.55e-04 | 90.3s
2026-05-14 16:44:00,189 - INFO - [joint] epoch 10/15 | train loss 1.7361 (noise 0.0153, aux 1.6885) | val 2.4713 | lr d=3.05e-05 i=1.12e-04 | 90.4s
2026-05-14 16:44:00,190 - INFO - Early stopping at epoch 10 (patience=5)
2026-05-14 16:44:00,443 - INFO - [joint_pretrain_dim7_gC] Trial 2/4: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 16:44:00,492 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 16:44:00,493 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:44:00,493 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:44:00,557 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:44:00,558 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:44:00,558 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:44:00,560 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:44:00,560 - INFO - DiffusionTSF initialized:
2026-05-14 16:44:00,560 - INFO -   Variables: 7 (multivariate)
2026-05-14 16:44:00,560 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:44:00,560 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:44:00,575 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:44:09,944 - INFO - [warmup] epoch 1/15 | train loss 2.3460 (noise 0.0000, aux 2.3460) | val 2.2996 | lr d=2.67e-04 i=1.96e-04 | 9.2s
2026-05-14 16:45:43,159 - INFO - [joint] epoch 2/15 | train loss 2.3996 (noise 0.0901, aux 2.2506) | val 2.3485 | lr d=2.58e-04 i=1.90e-04 | 90.3s
2026-05-14 16:47:16,500 - INFO - [joint] epoch 3/15 | train loss 2.2771 (noise 0.0285, aux 2.2036) | val 2.3714 | lr d=2.44e-04 i=1.80e-04 | 90.4s
2026-05-14 16:48:49,671 - INFO - [joint] epoch 4/15 | train loss 2.2131 (noise 0.0213, aux 2.1513) | val 2.3434 | lr d=2.25e-04 i=1.66e-04 | 90.3s
2026-05-14 16:50:22,990 - INFO - [joint] epoch 5/15 | train loss 2.1272 (noise 0.0175, aux 2.0723) | val 2.3739 | lr d=2.03e-04 i=1.49e-04 | 90.4s
2026-05-14 16:51:56,135 - INFO - [joint] epoch 6/15 | train loss 2.0268 (noise 0.0172, aux 1.9732) | val 2.3891 | lr d=1.77e-04 i=1.31e-04 | 90.3s
2026-05-14 16:53:29,355 - INFO - [joint] epoch 7/15 | train loss 1.9168 (noise 0.0158, aux 1.8671) | val 2.5010 | lr d=1.50e-04 i=1.10e-04 | 90.3s
2026-05-14 16:55:02,607 - INFO - [joint] epoch 8/15 | train loss 1.7987 (noise 0.0149, aux 1.7505) | val 2.5518 | lr d=1.22e-04 i=8.99e-05 | 90.4s
2026-05-14 16:56:35,921 - INFO - [joint] epoch 9/15 | train loss 1.7087 (noise 0.0151, aux 1.6622) | val 2.5616 | lr d=9.45e-05 i=6.99e-05 | 90.4s
2026-05-14 16:56:35,921 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 16:56:36,160 - INFO - [joint_pretrain_dim7_gC] Trial 3/4: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 16:56:36,217 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 16:56:36,218 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:56:36,218 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:56:36,294 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:56:36,294 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:56:36,294 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:56:36,296 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:56:36,297 - INFO - DiffusionTSF initialized:
2026-05-14 16:56:36,297 - INFO -   Variables: 7 (multivariate)
2026-05-14 16:56:36,297 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:56:36,297 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:56:36,312 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:56:45,645 - INFO - [warmup] epoch 1/15 | train loss 2.3819 (noise 0.0000, aux 2.3819) | val 2.3045 | lr d=7.08e-05 i=7.08e-05 | 9.2s
2026-05-14 16:58:18,763 - INFO - [joint] epoch 2/15 | train loss 2.4512 (noise 0.1217, aux 2.2648) | val 2.3635 | lr d=6.85e-05 i=6.85e-05 | 90.2s
2026-05-14 16:59:52,129 - INFO - [joint] epoch 3/15 | train loss 2.3205 (noise 0.0404, aux 2.2283) | val 2.3323 | lr d=6.48e-05 i=6.48e-05 | 90.4s
2026-05-14 17:01:25,425 - INFO - [joint] epoch 4/15 | train loss 2.2713 (noise 0.0322, aux 2.1925) | val 2.3211 | lr d=5.99e-05 i=5.99e-05 | 90.3s
2026-05-14 17:02:58,682 - INFO - [joint] epoch 5/15 | train loss 2.2257 (noise 0.0247, aux 2.1571) | val 2.3011 | lr d=5.39e-05 i=5.39e-05 | 90.3s
2026-05-14 17:04:31,824 - INFO - [joint] epoch 6/15 | train loss 2.1832 (noise 0.0217, aux 2.1210) | val 2.3434 | lr d=4.71e-05 i=4.71e-05 | 90.2s
2026-05-14 17:06:04,947 - INFO - [joint] epoch 7/15 | train loss 2.1394 (noise 0.0214, aux 2.0785) | val 2.3585 | lr d=3.99e-05 i=3.99e-05 | 90.2s
2026-05-14 17:07:38,171 - INFO - [joint] epoch 8/15 | train loss 2.0862 (noise 0.0205, aux 2.0290) | val 2.3715 | lr d=3.25e-05 i=3.25e-05 | 90.3s
2026-05-14 17:09:11,483 - INFO - [joint] epoch 9/15 | train loss 2.0385 (noise 0.0193, aux 1.9821) | val 2.3943 | lr d=2.52e-05 i=2.52e-05 | 90.4s
2026-05-14 17:10:44,717 - INFO - [joint] epoch 10/15 | train loss 1.9954 (noise 0.0179, aux 1.9413) | val 2.4226 | lr d=1.84e-05 i=1.84e-05 | 90.3s
2026-05-14 17:10:44,719 - INFO - Early stopping at epoch 10 (patience=5)
2026-05-14 17:10:44,962 - INFO - [joint_pretrain_dim7_gC] Trial 4/4: diffusion_lr=5.72e-05, itrans_lr=3.67e-04
2026-05-14 17:10:45,012 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:10:45,013 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:10:45,013 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:10:45,082 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:10:45,082 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:10:45,082 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:10:45,085 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:10:45,085 - INFO - DiffusionTSF initialized:
2026-05-14 17:10:45,085 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:10:45,085 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:10:45,085 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:10:45,100 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:10:54,535 - INFO - [warmup] epoch 1/15 | train loss 2.3485 (noise 0.0000, aux 2.3485) | val 2.2894 | lr d=5.65e-05 i=3.63e-04 | 9.3s
2026-05-14 17:12:27,863 - INFO - [joint] epoch 2/15 | train loss 2.4713 (noise 0.1331, aux 2.2705) | val 2.3778 | lr d=5.47e-05 i=3.52e-04 | 90.4s
2026-05-14 17:12:28,017 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:12:28,017 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:12:28,017 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:12:28,083 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:12:28,083 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:12:28,083 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:12:28,085 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:12:28,085 - INFO - DiffusionTSF initialized:
2026-05-14 17:12:28,085 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:12:28,085 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:12:28,085 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:12:28,205 - INFO - Joint pretrain checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gC.pt (val=2.3011, epoch=5)
Done (worker).
```
