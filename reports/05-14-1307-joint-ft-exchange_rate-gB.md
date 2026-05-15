# Run report: 05-14-1307-joint-ft-exchange_rate-gB

- Job ID: 14
- Log: results/logs/05-14-1307-joint-ft-exchange_rate-gB.log
- Duration: 0h 32m 56s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581307  task=finetune  ghost=gB  log: ./results/logs/05-14-1307-joint-ft-exchange_rate-gB.log
Node: kn047  started: 2026-05-14T17:12:00-04:00
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
2026-05-14 17:12:08,864 - INFO - traffic.csv already exists
2026-05-14 17:12:08,944 - INFO - ============================================================
2026-05-14 17:12:08,945 - INFO - Joint finetune (e2e): subset=exchange_rate, dim=8, epochs=10, trials=3
2026-05-14 17:12:08,945 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim8/joint_pretrained_gB.pt
2026-05-14 17:12:08,945 - INFO - ============================================================
2026-05-14 17:12:09,279 - INFO - [joint_finetune_exchange_rate_gB] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:12:09,354 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:12:09,354 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:12:09,355 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:12:09,427 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:12:09,427 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:12:09,427 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:12:09,430 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:12:09,430 - INFO - DiffusionTSF initialized:
2026-05-14 17:12:09,430 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:12:09,430 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:12:09,430 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:12:09,452 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:12:22,277 - INFO - [warmup] epoch 1/10 | train loss 2.7647 (noise 0.0000, aux 2.7647) | val 3.0559 | lr d=1.16e-04 i=4.35e-04 | 11.4s
2026-05-14 17:14:51,697 - INFO - [joint] epoch 2/10 | train loss 2.5555 (noise 0.0137, aux 2.5063) | val 3.2504 | lr d=1.07e-04 i=4.04e-04 | 144.1s
2026-05-14 17:17:20,663 - INFO - [joint] epoch 3/10 | train loss 2.3336 (noise 0.0121, aux 2.2899) | val 3.4264 | lr d=9.43e-05 i=3.55e-04 | 143.6s
2026-05-14 17:19:49,982 - INFO - [joint] epoch 4/10 | train loss 2.0808 (noise 0.0112, aux 2.0391) | val 3.4481 | lr d=7.79e-05 i=2.93e-04 | 143.9s
2026-05-14 17:22:18,981 - INFO - [joint] epoch 5/10 | train loss 1.8235 (noise 0.0102, aux 1.7848) | val 3.6766 | lr d=5.98e-05 i=2.24e-04 | 143.6s
2026-05-14 17:24:47,889 - INFO - [joint] epoch 6/10 | train loss 1.5520 (noise 0.0099, aux 1.5162) | val 3.8884 | lr d=4.17e-05 i=1.55e-04 | 143.5s
2026-05-14 17:27:16,850 - INFO - [joint] epoch 7/10 | train loss 1.3051 (noise 0.0096, aux 1.2713) | val 4.0680 | lr d=2.54e-05 i=9.29e-05 | 143.6s
2026-05-14 17:27:16,852 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:27:17,155 - INFO - [joint_finetune_exchange_rate_gB] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 17:27:17,225 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:27:17,225 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:27:17,228 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:27:17,303 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:27:17,303 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:27:17,303 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:27:17,307 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:27:17,307 - INFO - DiffusionTSF initialized:
2026-05-14 17:27:17,307 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:27:17,307 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:27:17,307 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:27:17,330 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:27:28,557 - INFO - [warmup] epoch 1/10 | train loss 2.5467 (noise 0.0000, aux 2.5467) | val 3.3096 | lr d=2.63e-04 i=1.94e-04 | 11.0s
2026-05-14 17:29:57,480 - INFO - [joint] epoch 2/10 | train loss 1.9350 (noise 0.0171, aux 1.8806) | val 3.6368 | lr d=2.44e-04 i=1.80e-04 | 143.6s
2026-05-14 17:32:26,614 - INFO - [joint] epoch 3/10 | train loss 1.5270 (noise 0.0128, aux 1.4815) | val 3.8607 | lr d=2.15e-04 i=1.58e-04 | 143.7s
2026-05-14 17:34:55,734 - INFO - [joint] epoch 4/10 | train loss 1.2418 (noise 0.0120, aux 1.1988) | val 3.9630 | lr d=1.77e-04 i=1.31e-04 | 143.7s
2026-05-14 17:37:24,793 - INFO - [joint] epoch 5/10 | train loss 1.0193 (noise 0.0103, aux 0.9808) | val 4.1065 | lr d=1.36e-04 i=1.00e-04 | 143.7s
2026-05-14 17:39:53,890 - INFO - [joint] epoch 6/10 | train loss 0.8649 (noise 0.0099, aux 0.8292) | val 3.9583 | lr d=9.45e-05 i=6.99e-05 | 143.7s
2026-05-14 17:42:23,009 - INFO - [joint] epoch 7/10 | train loss 0.7474 (noise 0.0090, aux 0.7146) | val 3.9065 | lr d=5.72e-05 i=4.25e-05 | 143.7s
2026-05-14 17:42:23,010 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:42:23,247 - INFO - [joint_finetune_exchange_rate_gB] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 17:42:23,306 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:42:23,306 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:42:23,307 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:42:23,375 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:42:23,375 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:42:23,375 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:42:23,377 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:42:23,377 - INFO - DiffusionTSF initialized:
2026-05-14 17:42:23,378 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:42:23,378 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:42:23,378 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:42:23,399 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:42:34,908 - INFO - [warmup] epoch 1/10 | train loss 2.5877 (noise 0.0000, aux 2.5877) | val 3.1301 | lr d=6.99e-05 i=6.99e-05 | 11.3s
2026-05-14 17:45:03,918 - INFO - [joint] epoch 2/10 | train loss 2.1329 (noise 0.0122, aux 2.0872) | val 3.6175 | lr d=6.48e-05 i=6.48e-05 | 143.6s
2026-05-14 17:45:04,103 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:45:04,104 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:45:04,104 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:45:04,172 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:45:04,172 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:45:04,172 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:45:04,175 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:45:04,175 - INFO - DiffusionTSF initialized:
2026-05-14 17:45:04,175 - INFO -   Variables: 8 (multivariate)
2026-05-14 17:45:04,175 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:45:04,175 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:45:04,454 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/exchange_rate_joint_finetuned_gB.pt (val=3.2504, epoch=2)
Done (worker).
```
