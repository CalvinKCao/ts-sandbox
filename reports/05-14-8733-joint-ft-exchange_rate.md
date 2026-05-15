# Run report: 05-14-8733-joint-ft-exchange_rate

- Job ID: 14
- Log: results/logs/05-14-8733-joint-ft-exchange_rate.log
- Duration: 1h 0m 1s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```Job ID: 3578733  task=finetune  log: ./results/logs/05-14-8733-joint-ft-exchange_rate.log
Node: kn056  started: 2026-05-14T12:46:20-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=exchange_rate n_variates=8 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 12:46:28,300 - INFO - traffic.csv already exists
2026-05-14 12:46:28,378 - INFO - ============================================================
2026-05-14 12:46:28,378 - INFO - Joint finetune (e2e): subset=exchange_rate, dim=8, epochs=10, trials=3
2026-05-14 12:46:28,378 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim8/joint_pretrained.pt
2026-05-14 12:46:28,378 - INFO - ============================================================
2026-05-14 12:46:28,665 - INFO - [joint_finetune_exchange_rate] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 12:46:28,739 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 12:46:28,739 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 12:46:28,740 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 12:46:28,808 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 12:46:28,808 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 12:46:28,808 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 12:46:28,811 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 12:46:28,811 - INFO - DiffusionTSF initialized:
2026-05-14 12:46:28,811 - INFO -   Variables: 8 (multivariate)
2026-05-14 12:46:28,811 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 12:46:28,811 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 12:46:28,832 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 12:46:41,289 - INFO - [warmup] epoch 1/10 | train loss 105960865236.1719 (noise 0.0000, aux 105960865236.1719) | val 9.5213 | lr d=1.16e-04 i=4.35e-04 | 11.1s
2026-05-14 12:49:10,249 - INFO - [joint] epoch 2/10 | train loss 114960326956.7856 (noise 0.0226, aux 114960326956.7377) | val 7.5915 | lr d=1.07e-04 i=4.04e-04 | 143.6s
2026-05-14 12:51:38,794 - INFO - [joint] epoch 3/10 | train loss 116587260248.5985 (noise 0.0167, aux 116587260248.5598) | val 8.1170 | lr d=9.43e-05 i=3.55e-04 | 143.1s
2026-05-14 12:54:07,201 - INFO - [joint] epoch 4/10 | train loss 121172551608.8369 (noise 0.0149, aux 121172551608.7993) | val 7.6940 | lr d=7.79e-05 i=2.93e-04 | 143.1s
2026-05-14 12:56:35,563 - INFO - [joint] epoch 5/10 | train loss 120019613947.7045 (noise 0.0129, aux 120019613947.6709) | val 7.8593 | lr d=5.98e-05 i=2.24e-04 | 143.0s
2026-05-14 12:59:03,814 - INFO - [joint] epoch 6/10 | train loss 119853360901.8413 (noise 0.0119, aux 119853360901.8105) | val 7.3881 | lr d=4.17e-05 i=1.55e-04 | 142.9s
2026-05-14 13:01:32,414 - INFO - [joint] epoch 7/10 | train loss 121776473732.9396 (noise 0.0114, aux 121776473732.9096) | val 7.6996 | lr d=2.54e-05 i=9.29e-05 | 143.2s
2026-05-14 13:04:00,861 - INFO - [joint] epoch 8/10 | train loss 121803639493.1747 (noise 0.0110, aux 121803639493.1469) | val 7.6291 | lr d=1.24e-05 i=4.37e-05 | 143.1s
2026-05-14 13:06:29,330 - INFO - [joint] epoch 9/10 | train loss 121856970265.5688 (noise 0.0101, aux 121856970265.5421) | val 7.6300 | lr d=4.05e-06 i=1.21e-05 | 143.1s
2026-05-14 13:08:58,119 - INFO - [joint] epoch 10/10 | train loss 121770075082.1343 (noise 0.0105, aux 121770075082.1075) | val 7.6035 | lr d=1.18e-06 i=1.18e-06 | 143.4s
2026-05-14 13:08:58,363 - INFO - [joint_finetune_exchange_rate] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 13:08:58,434 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:08:58,434 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:08:58,435 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:08:58,504 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:08:58,504 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:08:58,504 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:08:58,507 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:08:58,507 - INFO - DiffusionTSF initialized:
2026-05-14 13:08:58,507 - INFO -   Variables: 8 (multivariate)
2026-05-14 13:08:58,507 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:08:58,507 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:08:58,528 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 13:09:09,398 - INFO - [warmup] epoch 1/10 | train loss 100062174737.3941 (noise 0.0000, aux 100062174737.3941) | val 7.2997 | lr d=2.63e-04 i=1.94e-04 | 10.6s
2026-05-14 13:11:37,804 - INFO - [joint] epoch 2/10 | train loss 77965425047.3967 (noise 0.0313, aux 77965425047.3392) | val 8.2613 | lr d=2.44e-04 i=1.80e-04 | 143.1s
2026-05-14 13:14:06,301 - INFO - [joint] epoch 3/10 | train loss 55022626141.9830 (noise 0.0164, aux 55022626141.9444) | val 10.2602 | lr d=2.15e-04 i=1.58e-04 | 143.1s
2026-05-14 13:16:34,802 - INFO - [joint] epoch 4/10 | train loss 35961431377.4095 (noise 0.0136, aux 35961431377.3733) | val 7.6919 | lr d=1.77e-04 i=1.31e-04 | 143.1s
2026-05-14 13:19:03,457 - INFO - [joint] epoch 5/10 | train loss 28094645106.1424 (noise 0.0123, aux 28094645106.1093) | val 6.9875 | lr d=1.36e-04 i=1.00e-04 | 143.2s
2026-05-14 13:21:31,989 - INFO - [joint] epoch 6/10 | train loss 19456589998.4646 (noise 0.0111, aux 19456589998.4336) | val 7.2967 | lr d=9.45e-05 i=6.99e-05 | 143.1s
2026-05-14 13:24:00,458 - INFO - [joint] epoch 7/10 | train loss 14915595663.7354 (noise 0.0102, aux 14915595663.7074) | val 6.7382 | lr d=5.72e-05 i=4.25e-05 | 143.1s
2026-05-14 13:26:28,995 - INFO - [joint] epoch 8/10 | train loss 12647277889.6812 (noise 0.0095, aux 12647277889.6552) | val 6.9650 | lr d=2.76e-05 i=2.07e-05 | 143.1s
2026-05-14 13:28:57,435 - INFO - [joint] epoch 9/10 | train loss 11522545732.7902 (noise 0.0092, aux 11522545732.7661) | val 6.9733 | lr d=8.54e-06 i=6.79e-06 | 143.1s
2026-05-14 13:31:25,899 - INFO - [joint] epoch 10/10 | train loss 11053114674.7981 (noise 0.0091, aux 11053114674.7739) | val 7.1687 | lr d=1.98e-06 i=1.98e-06 | 143.1s
2026-05-14 13:31:26,164 - INFO - [joint_finetune_exchange_rate] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 13:31:26,250 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:31:26,250 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:31:26,250 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:31:26,320 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:31:26,320 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:31:26,320 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:31:26,322 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:31:26,322 - INFO - DiffusionTSF initialized:
2026-05-14 13:31:26,322 - INFO -   Variables: 8 (multivariate)
2026-05-14 13:31:26,322 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:31:26,322 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:31:26,343 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 13:31:37,121 - INFO - [warmup] epoch 1/10 | train loss 97663176729.5040 (noise 0.0000, aux 97663176729.5040) | val 6.7499 | lr d=6.99e-05 i=6.99e-05 | 10.6s
2026-05-14 13:34:05,491 - INFO - [joint] epoch 2/10 | train loss 81388271486.4060 (noise 0.0214, aux 81388271486.3599) | val 6.5582 | lr d=6.48e-05 i=6.48e-05 | 143.0s
2026-05-14 13:36:34,281 - INFO - [joint] epoch 3/10 | train loss 64459635622.1194 (noise 0.0170, aux 64459635622.0805) | val 6.8167 | lr d=5.70e-05 i=5.70e-05 | 143.4s
2026-05-14 13:39:03,020 - INFO - [joint] epoch 4/10 | train loss 50204439192.5105 (noise 0.0146, aux 50204439192.4735) | val 6.5712 | lr d=4.71e-05 i=4.71e-05 | 143.4s
2026-05-14 13:41:31,750 - INFO - [joint] epoch 5/10 | train loss 40243000727.2639 (noise 0.0142, aux 40243000727.2289) | val 6.7288 | lr d=3.62e-05 i=3.62e-05 | 143.4s
2026-05-14 13:44:00,536 - INFO - [joint] epoch 6/10 | train loss 34122365262.3900 (noise 0.0134, aux 34122365262.3556) | val 6.9182 | lr d=2.52e-05 i=2.52e-05 | 143.4s
2026-05-14 13:46:29,470 - INFO - [joint] epoch 7/10 | train loss 29932436467.8131 (noise 0.0121, aux 29932436467.7811) | val 7.6396 | lr d=1.53e-05 i=1.53e-05 | 143.6s
2026-05-14 13:46:29,472 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 13:46:29,763 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:46:29,763 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:46:29,763 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:46:29,828 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:46:29,829 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:46:29,829 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:46:29,831 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:46:29,831 - INFO - DiffusionTSF initialized:
2026-05-14 13:46:29,831 - INFO -   Variables: 8 (multivariate)
2026-05-14 13:46:29,831 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:46:29,831 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:46:29,990 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/exchange_rate_joint_finetuned.pt (val=6.5582, epoch=2)
Done (worker).
```
