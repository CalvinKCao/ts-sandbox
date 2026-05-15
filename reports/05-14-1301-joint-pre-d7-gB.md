# Run report: 05-14-1301-joint-pre-d7-gB

- Job ID: 14
- Log: results/logs/05-14-1301-joint-pre-d7-gB.log
- Duration: 0h 39m 49s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```2026-05-14 16:30:15,535 - INFO - RealTS initialized: 4096 samples/epoch, lookback=96, forecast=96, variables=7, pool_rows=50000, epoch_stride=False (train_n=3687, val_tail=409, cap=15)
2026-05-14 16:30:15,537 - INFO - Reusing existing synthetic pool /scratch/ccao87/ts-sandbox/synth_data/synth_pool_v7_L192.npy (has 50000 samples, need 50000)
2026-05-14 16:30:15,537 - INFO - Created synthetic-only dataloader: 4096 samples/epoch (Pool: 4096), lookback=96, forecast=96, variables=7
2026-05-14 16:30:15,539 - INFO - [joint_pretrain_dim7_gB] Trial 1/4: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 16:30:15,754 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 16:30:15,754 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:30:15,755 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:30:15,823 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:30:15,824 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:30:15,824 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:30:15,826 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:30:15,826 - INFO - DiffusionTSF initialized:
2026-05-14 16:30:15,826 - INFO -   Variables: 7 (multivariate)
2026-05-14 16:30:15,826 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:30:15,826 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:30:15,842 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:30:26,845 - INFO - [warmup] epoch 1/15 | train loss 2.3612 (noise 0.0000, aux 2.3612) | val 2.3301 | lr d=1.17e-04 i=4.41e-04 | 9.7s
2026-05-14 16:32:01,414 - INFO - [joint] epoch 2/15 | train loss 2.4781 (noise 0.1256, aux 2.2871) | val 2.3844 | lr d=1.13e-04 i=4.27e-04 | 91.7s
2026-05-14 16:33:35,340 - INFO - [joint] epoch 3/15 | train loss 2.3437 (noise 0.0337, aux 2.2609) | val 2.3787 | lr d=1.07e-04 i=4.04e-04 | 90.9s
2026-05-14 16:35:09,370 - INFO - [joint] epoch 4/15 | train loss 2.2881 (noise 0.0253, aux 2.2182) | val 2.3370 | lr d=9.90e-05 i=3.73e-04 | 91.0s
2026-05-14 16:36:43,367 - INFO - [joint] epoch 5/15 | train loss 2.2229 (noise 0.0217, aux 2.1597) | val 2.3372 | lr d=8.91e-05 i=3.35e-04 | 91.0s
2026-05-14 16:38:17,182 - INFO - [joint] epoch 6/15 | train loss 2.1405 (noise 0.0202, aux 2.0813) | val 2.3597 | lr d=7.79e-05 i=2.93e-04 | 90.9s
2026-05-14 16:39:51,026 - INFO - [joint] epoch 7/15 | train loss 2.0539 (noise 0.0193, aux 1.9976) | val 2.4533 | lr d=6.59e-05 i=2.47e-04 | 90.9s
2026-05-14 16:41:24,969 - INFO - [joint] epoch 8/15 | train loss 1.9452 (noise 0.0163, aux 1.8933) | val 2.4695 | lr d=5.37e-05 i=2.01e-04 | 91.0s
2026-05-14 16:42:58,746 - INFO - [joint] epoch 9/15 | train loss 1.8301 (noise 0.0158, aux 1.7811) | val 2.4788 | lr d=4.17e-05 i=1.55e-04 | 90.9s
2026-05-14 16:42:58,747 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 16:42:58,994 - INFO - [joint_pretrain_dim7_gB] Trial 2/4: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 16:42:59,045 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 16:42:59,045 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:42:59,045 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:42:59,108 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:42:59,108 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:42:59,108 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:42:59,110 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:42:59,110 - INFO - DiffusionTSF initialized:
2026-05-14 16:42:59,111 - INFO -   Variables: 7 (multivariate)
2026-05-14 16:42:59,111 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:42:59,111 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:42:59,126 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:43:08,511 - INFO - [warmup] epoch 1/15 | train loss 2.3429 (noise 0.0000, aux 2.3429) | val 2.2732 | lr d=2.67e-04 i=1.96e-04 | 9.2s
2026-05-14 16:44:42,257 - INFO - [joint] epoch 2/15 | train loss 2.4061 (noise 0.0938, aux 2.2518) | val 2.3455 | lr d=2.58e-04 i=1.90e-04 | 90.8s
2026-05-14 16:46:16,194 - INFO - [joint] epoch 3/15 | train loss 2.2734 (noise 0.0271, aux 2.2020) | val 2.3270 | lr d=2.44e-04 i=1.80e-04 | 91.0s
2026-05-14 16:47:50,139 - INFO - [joint] epoch 4/15 | train loss 2.2092 (noise 0.0201, aux 2.1485) | val 2.3452 | lr d=2.25e-04 i=1.66e-04 | 91.0s
2026-05-14 16:49:24,079 - INFO - [joint] epoch 5/15 | train loss 2.1257 (noise 0.0175, aux 2.0711) | val 2.3833 | lr d=2.03e-04 i=1.49e-04 | 91.0s
2026-05-14 16:50:57,935 - INFO - [joint] epoch 6/15 | train loss 2.0187 (noise 0.0168, aux 1.9672) | val 2.4331 | lr d=1.77e-04 i=1.31e-04 | 90.9s
2026-05-14 16:52:31,790 - INFO - [joint] epoch 7/15 | train loss 1.9135 (noise 0.0158, aux 1.8628) | val 2.4786 | lr d=1.50e-04 i=1.10e-04 | 90.9s
2026-05-14 16:54:05,759 - INFO - [joint] epoch 8/15 | train loss 1.8048 (noise 0.0138, aux 1.7588) | val 2.5542 | lr d=1.22e-04 i=8.99e-05 | 91.1s
2026-05-14 16:54:05,759 - INFO - Early stopping at epoch 8 (patience=5)
2026-05-14 16:54:05,990 - INFO - [joint_pretrain_dim7_gB] Trial 3/4: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 16:54:06,043 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 16:54:06,043 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 16:54:06,044 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 16:54:06,115 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 16:54:06,116 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 16:54:06,116 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 16:54:06,118 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 16:54:06,118 - INFO - DiffusionTSF initialized:
2026-05-14 16:54:06,118 - INFO -   Variables: 7 (multivariate)
2026-05-14 16:54:06,118 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 16:54:06,118 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 16:54:06,133 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 16:54:15,450 - INFO - [warmup] epoch 1/15 | train loss 2.3834 (noise 0.0000, aux 2.3834) | val 2.2922 | lr d=7.08e-05 i=7.08e-05 | 9.1s
2026-05-14 16:55:49,230 - INFO - [joint] epoch 2/15 | train loss 2.4686 (noise 0.1381, aux 2.2616) | val 2.3640 | lr d=6.85e-05 i=6.85e-05 | 90.9s
2026-05-14 16:57:23,190 - INFO - [joint] epoch 3/15 | train loss 2.3211 (noise 0.0442, aux 2.2259) | val 2.3355 | lr d=6.48e-05 i=6.48e-05 | 91.0s
2026-05-14 16:58:57,078 - INFO - [joint] epoch 4/15 | train loss 2.2673 (noise 0.0302, aux 2.1911) | val 2.3170 | lr d=5.99e-05 i=5.99e-05 | 90.9s
2026-05-14 17:00:30,986 - INFO - [joint] epoch 5/15 | train loss 2.2284 (noise 0.0251, aux 2.1605) | val 2.3082 | lr d=5.39e-05 i=5.39e-05 | 90.9s
2026-05-14 17:02:04,966 - INFO - [joint] epoch 6/15 | train loss 2.1871 (noise 0.0244, aux 2.1211) | val 2.3271 | lr d=4.71e-05 i=4.71e-05 | 91.0s
2026-05-14 17:03:38,816 - INFO - [joint] epoch 7/15 | train loss 2.1398 (noise 0.0197, aux 2.0797) | val 2.3638 | lr d=3.99e-05 i=3.99e-05 | 90.9s
2026-05-14 17:05:12,718 - INFO - [joint] epoch 8/15 | train loss 2.0906 (noise 0.0182, aux 2.0348) | val 2.3597 | lr d=3.25e-05 i=3.25e-05 | 91.0s
2026-05-14 17:06:46,627 - INFO - [joint] epoch 9/15 | train loss 2.0480 (noise 0.0184, aux 1.9928) | val 2.4025 | lr d=2.52e-05 i=2.52e-05 | 91.0s
2026-05-14 17:08:20,528 - INFO - [joint] epoch 10/15 | train loss 2.0037 (noise 0.0182, aux 1.9510) | val 2.3849 | lr d=1.84e-05 i=1.84e-05 | 91.0s
2026-05-14 17:08:20,529 - INFO - Early stopping at epoch 10 (patience=5)
2026-05-14 17:08:20,749 - INFO - [joint_pretrain_dim7_gB] Trial 4/4: diffusion_lr=5.72e-05, itrans_lr=3.67e-04
2026-05-14 17:08:20,799 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:08:20,799 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:08:20,800 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:08:20,863 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:08:20,863 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:08:20,863 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:08:20,866 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:08:20,866 - INFO - DiffusionTSF initialized:
2026-05-14 17:08:20,866 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:08:20,866 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:08:20,866 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:08:20,881 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:08:30,401 - INFO - [warmup] epoch 1/15 | train loss 2.3497 (noise 0.0000, aux 2.3497) | val 2.2998 | lr d=5.65e-05 i=3.63e-04 | 9.3s
2026-05-14 17:10:04,192 - INFO - [joint] epoch 2/15 | train loss 2.4861 (noise 0.1428, aux 2.2751) | val 2.4085 | lr d=5.47e-05 i=3.52e-04 | 90.9s
2026-05-14 17:10:04,349 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:10:04,349 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:10:04,349 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:10:04,418 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:10:04,418 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:10:04,418 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:10:04,420 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:10:04,420 - INFO - DiffusionTSF initialized:
2026-05-14 17:10:04,421 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:10:04,421 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:10:04,421 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:10:04,556 - INFO - Joint pretrain checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gB.pt (val=2.3082, epoch=5)
Done (worker).
```
