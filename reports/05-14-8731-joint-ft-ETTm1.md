# Run report: 05-14-8731-joint-ft-ETTm1

- Job ID: 14
- Log: results/logs/05-14-8731-joint-ft-ETTm1.log
- Duration: 3h 23m 2s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3578731  task=finetune  log: ./results/logs/05-14-8731-joint-ft-ETTm1.log
Node: kn055  started: 2026-05-14T12:43:22-04:00
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
2026-05-14 12:43:35,767 - INFO - traffic.csv already exists
2026-05-14 12:43:35,851 - INFO - ============================================================
2026-05-14 12:43:35,852 - INFO - Joint finetune (e2e): subset=ETTm1, dim=7, epochs=10, trials=3
2026-05-14 12:43:35,852 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained.pt
2026-05-14 12:43:35,852 - INFO - ============================================================
2026-05-14 12:43:36,384 - INFO - [joint_finetune_ETTm1] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 12:43:36,568 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 12:43:36,568 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 12:43:36,619 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 12:43:36,711 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 12:43:36,711 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 12:43:36,711 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 12:43:36,734 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 12:43:36,736 - INFO - DiffusionTSF initialized:
2026-05-14 12:43:36,736 - INFO -   Variables: 7 (multivariate)
2026-05-14 12:43:36,736 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 12:43:36,737 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 12:43:36,765 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 12:44:59,789 - INFO - [warmup] epoch 1/10 | train loss 36448077498206.3281 (noise 0.0000, aux 36448077498206.3281) | val 79475015581418.0938 | lr d=1.16e-04 i=4.35e-04 | 77.0s
2026-05-14 13:00:24,549 - INFO - [joint] epoch 2/10 | train loss 35691881130594.2656 (noise 0.0179, aux 35691881130594.2031) | val 79771744681715.3281 | lr d=1.07e-04 i=4.04e-04 | 843.9s
2026-05-14 13:15:48,189 - INFO - [joint] epoch 3/10 | train loss 35355130838139.5000 (noise 0.0119, aux 35355130838139.4609) | val 79433766707750.4375 | lr d=9.43e-05 i=3.55e-04 | 842.8s
2026-05-14 13:31:11,829 - INFO - [joint] epoch 4/10 | train loss 35684301948675.3203 (noise 0.0111, aux 35684301948675.2969) | val 79458410588579.4844 | lr d=7.79e-05 i=2.93e-04 | 842.7s
2026-05-14 13:46:35,552 - INFO - [joint] epoch 5/10 | train loss 35468158579835.7500 (noise 0.0101, aux 35468158579835.7031) | val 80072991109838.8281 | lr d=5.98e-05 i=2.24e-04 | 842.8s
2026-05-14 14:01:59,872 - INFO - [joint] epoch 6/10 | train loss 32458117456630.8867 (noise 0.0091, aux 32458117456630.8516) | val 79484751206361.2031 | lr d=4.17e-05 i=1.55e-04 | 843.3s
2026-05-14 14:17:24,340 - INFO - [joint] epoch 7/10 | train loss 29816347561513.4805 (noise 0.0084, aux 29816347561513.4531) | val 79412967164498.1562 | lr d=2.54e-05 i=9.29e-05 | 843.6s
2026-05-14 14:32:48,261 - INFO - [joint] epoch 8/10 | train loss 28145989324452.0508 (noise 0.0079, aux 28145989324452.0273) | val 79841026220007.0625 | lr d=1.24e-05 i=4.37e-05 | 843.1s
2026-05-14 14:48:11,985 - INFO - [joint] epoch 9/10 | train loss 25945150982365.7422 (noise 0.0075, aux 25945150982365.7188) | val 80634838259263.5312 | lr d=4.05e-06 i=1.21e-05 | 843.0s
2026-05-14 15:03:35,716 - INFO - [joint] epoch 10/10 | train loss 25413664155001.6992 (noise 0.0073, aux 25413664155001.6797) | val 80547369091824.3750 | lr d=1.18e-06 i=1.18e-06 | 842.9s
2026-05-14 15:03:35,970 - INFO - [joint_finetune_ETTm1] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 15:03:36,019 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 15:03:36,019 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 15:03:36,019 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 15:03:36,080 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 15:03:36,080 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 15:03:36,080 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 15:03:36,082 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 15:03:36,083 - INFO - DiffusionTSF initialized:
2026-05-14 15:03:36,083 - INFO -   Variables: 7 (multivariate)
2026-05-14 15:03:36,083 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 15:03:36,083 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 15:03:36,103 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 15:04:59,377 - INFO - [warmup] epoch 1/10 | train loss 36154834194941.6250 (noise 0.0000, aux 36154834194941.6250) | val 79466694022043.5625 | lr d=2.63e-04 i=1.94e-04 | 79.8s
2026-05-14 15:20:22,694 - INFO - [joint] epoch 2/10 | train loss 34213576086496.3867 (noise 0.0182, aux 34213576086496.3438) | val 80061891147350.4531 | lr d=2.44e-04 i=1.80e-04 | 842.3s
2026-05-14 15:35:47,441 - INFO - [joint] epoch 3/10 | train loss 31531165911275.4648 (noise 0.0116, aux 31531165911275.4375) | val 80338030943042.7969 | lr d=2.15e-04 i=1.58e-04 | 843.6s
2026-05-14 15:51:12,404 - INFO - [joint] epoch 4/10 | train loss 29087775327452.8594 (noise 0.0103, aux 29087775327452.8242) | val 82119455308696.8594 | lr d=1.77e-04 i=1.31e-04 | 843.8s
2026-05-14 16:06:37,409 - INFO - [joint] epoch 5/10 | train loss 26942629366158.3516 (noise 0.0090, aux 26942629366158.3359) | val 81379790701112.5312 | lr d=1.36e-04 i=1.00e-04 | 843.8s
```
