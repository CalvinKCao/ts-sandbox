# Run report: 05-14-1299-joint-ft-exchange_rate-gC

- Job ID: 14
- Log: results/logs/05-14-1299-joint-ft-exchange_rate-gC.log
- Duration: 0h 35m 16s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581299  task=finetune  ghost=gC  log: ./results/logs/05-14-1299-joint-ft-exchange_rate-gC.log
Node: kn064  started: 2026-05-14T17:37:46-04:00
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
2026-05-14 17:38:00,050 - INFO - traffic.csv already exists
2026-05-14 17:38:00,150 - INFO - ============================================================
2026-05-14 17:38:00,150 - INFO - Joint finetune (e2e): subset=exchange_rate, dim=8, epochs=10, trials=3
2026-05-14 17:38:00,150 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim8/joint_pretrained_gC.pt
2026-05-14 17:38:00,150 - INFO - ============================================================
2026-05-14 17:38:00,626 - INFO - [joint_finetune_exchange_rate_gC] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:38:00,847 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:38:00,848 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:38:00,904 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:38:01,001 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:38:01,002 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:38:01,002 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:38:01,027 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:38:01,029 - INFO - DiffusionTSF initialized:
2026-05-14 17:38:01,029 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:38:01,029 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:38:01,029 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:38:01,061 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:38:16,885 - INFO - [warmup] epoch 1/10 | train loss 2.7583 (noise 0.0000, aux 2.7583) | val 3.0356 | lr d=1.16e-04 i=4.35e-04 | 13.0s
2026-05-14 17:40:45,922 - INFO - [joint] epoch 2/10 | train loss 2.5238 (noise 0.0232, aux 2.4562) | val 3.2746 | lr d=1.07e-04 i=4.04e-04 | 143.7s
2026-05-14 17:43:14,296 - INFO - [joint] epoch 3/10 | train loss 2.2670 (noise 0.0170, aux 2.2113) | val 3.2084 | lr d=9.43e-05 i=3.55e-04 | 143.0s
2026-05-14 17:45:42,739 - INFO - [joint] epoch 4/10 | train loss 2.0482 (noise 0.0145, aux 1.9979) | val 3.5408 | lr d=7.79e-05 i=2.93e-04 | 143.1s
2026-05-14 17:48:11,024 - INFO - [joint] epoch 5/10 | train loss 1.7863 (noise 0.0133, aux 1.7393) | val 3.5470 | lr d=5.98e-05 i=2.24e-04 | 143.0s
2026-05-14 17:50:39,371 - INFO - [joint] epoch 6/10 | train loss 1.5292 (noise 0.0128, aux 1.4853) | val 3.6249 | lr d=4.17e-05 i=1.55e-04 | 143.0s
2026-05-14 17:53:07,633 - INFO - [joint] epoch 7/10 | train loss 1.2876 (noise 0.0122, aux 1.2458) | val 3.7830 | lr d=2.54e-05 i=9.29e-05 | 142.9s
2026-05-14 17:55:35,918 - INFO - [joint] epoch 8/10 | train loss 1.0935 (noise 0.0117, aux 1.0541) | val 3.6571 | lr d=1.24e-05 i=4.37e-05 | 143.0s
2026-05-14 17:55:35,919 - INFO - Early stopping at epoch 8 (patience=5)
2026-05-14 17:55:36,138 - INFO - [joint_finetune_exchange_rate_gC] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 17:55:36,190 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:55:36,190 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:55:36,191 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:55:36,258 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:55:36,258 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:55:36,258 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:55:36,260 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:55:36,260 - INFO - DiffusionTSF initialized:
2026-05-14 17:55:36,260 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:55:36,260 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:55:36,260 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:55:36,281 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:55:47,050 - INFO - [warmup] epoch 1/10 | train loss 2.5816 (noise 0.0000, aux 2.5816) | val 3.3906 | lr d=2.63e-04 i=1.94e-04 | 10.6s
2026-05-14 17:58:15,219 - INFO - [joint] epoch 2/10 | train loss 2.0806 (noise 0.0311, aux 2.0005) | val 3.6087 | lr d=2.44e-04 i=1.80e-04 | 142.9s
2026-05-14 18:00:43,557 - INFO - [joint] epoch 3/10 | train loss 1.6176 (noise 0.0167, aux 1.5627) | val 3.7890 | lr d=2.15e-04 i=1.58e-04 | 143.0s
2026-05-14 18:03:11,827 - INFO - [joint] epoch 4/10 | train loss 1.3238 (noise 0.0148, aux 1.2727) | val 3.9448 | lr d=1.77e-04 i=1.31e-04 | 143.0s
2026-05-14 18:05:40,070 - INFO - [joint] epoch 5/10 | train loss 1.1012 (noise 0.0128, aux 1.0569) | val 3.8862 | lr d=1.36e-04 i=1.00e-04 | 142.9s
2026-05-14 18:08:08,364 - INFO - [joint] epoch 6/10 | train loss 0.9316 (noise 0.0119, aux 0.8892) | val 3.9881 | lr d=9.45e-05 i=6.99e-05 | 143.0s
2026-05-14 18:10:36,677 - INFO - [joint] epoch 7/10 | train loss 0.8014 (noise 0.0117, aux 0.7617) | val 4.1693 | lr d=5.72e-05 i=4.25e-05 | 143.0s
2026-05-14 18:10:36,678 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:10:36,900 - INFO - [joint_finetune_exchange_rate_gC] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 18:10:36,952 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 18:10:36,952 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:10:36,952 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:10:37,021 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:10:37,021 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:10:37,021 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:10:37,024 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:10:37,024 - INFO - DiffusionTSF initialized:
2026-05-14 18:10:37,024 - INFO -   Variables: 8 (multivariate)
2026-05-14 18:10:37,024 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:10:37,024 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:10:37,044 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 18:10:48,308 - INFO - [warmup] epoch 1/10 | train loss 2.6422 (noise 0.0000, aux 2.6422) | val 3.0374 | lr d=6.99e-05 i=6.99e-05 | 11.0s
2026-05-14 18:13:16,452 - INFO - [joint] epoch 2/10 | train loss 2.2260 (noise 0.0214, aux 2.1612) | val 3.6941 | lr d=6.48e-05 i=6.48e-05 | 142.8s
2026-05-14 18:13:16,622 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 18:13:16,622 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:13:16,622 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:13:16,688 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:13:16,688 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:13:16,688 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:13:16,690 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:13:16,690 - INFO - DiffusionTSF initialized:
2026-05-14 18:13:16,690 - INFO -   Variables: 8 (multivariate)
2026-05-14 18:13:16,690 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:13:16,690 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:13:16,869 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/exchange_rate_joint_finetuned_gC.pt (val=3.2084, epoch=3)
Done (worker).
```
