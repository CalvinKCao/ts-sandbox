# Run report: 05-14-1296-joint-ft-ETTh2-gC

- Job ID: 14
- Log: results/logs/05-14-1296-joint-ft-ETTh2-gC.log
- Duration: 1h 9m 12s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581296  task=finetune  ghost=gC  log: ./results/logs/05-14-1296-joint-ft-ETTh2-gC.log
Node: kn047  started: 2026-05-14T17:45:24-04:00
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
2026-05-14 17:45:35,223 - INFO - traffic.csv already exists
2026-05-14 17:45:35,306 - INFO - ============================================================
2026-05-14 17:45:35,306 - INFO - Joint finetune (e2e): subset=ETTh2, dim=7, epochs=10, trials=3
2026-05-14 17:45:35,306 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gC.pt
2026-05-14 17:45:35,306 - INFO - ============================================================
2026-05-14 17:45:35,668 - INFO - [joint_finetune_ETTh2_gC] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:45:35,773 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:45:35,773 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:45:35,774 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:45:35,845 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:45:35,845 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:45:35,845 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:45:35,848 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:45:35,848 - INFO - DiffusionTSF initialized:
2026-05-14 17:45:35,848 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:45:35,848 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:45:35,848 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:45:35,870 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:45:58,281 - INFO - [warmup] epoch 1/10 | train loss 1.4552 (noise 0.0000, aux 1.4552) | val 1.5884 | lr d=1.16e-04 i=4.35e-04 | 20.0s
2026-05-14 17:49:46,210 - INFO - [joint] epoch 2/10 | train loss 1.4278 (noise 0.0242, aux 1.3625) | val 1.6495 | lr d=1.07e-04 i=4.04e-04 | 208.1s
2026-05-14 17:53:33,624 - INFO - [joint] epoch 3/10 | train loss 1.3496 (noise 0.0183, aux 1.2945) | val 1.6924 | lr d=9.43e-05 i=3.55e-04 | 207.6s
2026-05-14 17:57:20,925 - INFO - [joint] epoch 4/10 | train loss 1.2621 (noise 0.0170, aux 1.2121) | val 1.7139 | lr d=7.79e-05 i=2.93e-04 | 207.5s
2026-05-14 18:01:08,018 - INFO - [joint] epoch 5/10 | train loss 1.1461 (noise 0.0151, aux 1.0994) | val 1.7338 | lr d=5.98e-05 i=2.24e-04 | 207.3s
2026-05-14 18:04:54,944 - INFO - [joint] epoch 6/10 | train loss 1.0217 (noise 0.0148, aux 0.9779) | val 1.7445 | lr d=4.17e-05 i=1.55e-04 | 207.2s
2026-05-14 18:08:42,058 - INFO - [joint] epoch 7/10 | train loss 0.9094 (noise 0.0135, aux 0.8680) | val 1.7495 | lr d=2.54e-05 i=9.29e-05 | 207.4s
2026-05-14 18:08:42,058 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:08:42,305 - INFO - [joint_finetune_ETTh2_gC] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 18:08:42,368 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 18:08:42,368 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:08:42,368 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:08:42,442 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:08:42,442 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:08:42,442 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:08:42,445 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:08:42,446 - INFO - DiffusionTSF initialized:
2026-05-14 18:08:42,446 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:08:42,446 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:08:42,446 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:08:42,468 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 18:09:01,904 - INFO - [warmup] epoch 1/10 | train loss 1.4059 (noise 0.0000, aux 1.4059) | val 1.5983 | lr d=2.63e-04 i=1.94e-04 | 18.5s
2026-05-14 18:12:49,185 - INFO - [joint] epoch 2/10 | train loss 1.3094 (noise 0.0277, aux 1.2378) | val 1.7440 | lr d=2.44e-04 i=1.80e-04 | 207.5s
2026-05-14 18:16:36,498 - INFO - [joint] epoch 3/10 | train loss 1.1368 (noise 0.0182, aux 1.0823) | val 1.7873 | lr d=2.15e-04 i=1.58e-04 | 207.4s
2026-05-14 18:20:23,660 - INFO - [joint] epoch 4/10 | train loss 1.0078 (noise 0.0158, aux 0.9591) | val 1.8132 | lr d=1.77e-04 i=1.31e-04 | 207.4s
2026-05-14 18:24:10,853 - INFO - [joint] epoch 5/10 | train loss 0.9035 (noise 0.0155, aux 0.8568) | val 1.7903 | lr d=1.36e-04 i=1.00e-04 | 207.4s
2026-05-14 18:27:58,029 - INFO - [joint] epoch 6/10 | train loss 0.8217 (noise 0.0132, aux 0.7795) | val 1.7695 | lr d=9.45e-05 i=6.99e-05 | 207.4s
2026-05-14 18:31:45,224 - INFO - [joint] epoch 7/10 | train loss 0.7561 (noise 0.0126, aux 0.7174) | val 1.7754 | lr d=5.72e-05 i=4.25e-05 | 207.4s
2026-05-14 18:31:45,225 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:31:45,493 - INFO - [joint_finetune_ETTh2_gC] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 18:31:45,556 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 18:31:45,556 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:31:45,557 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:31:45,634 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:31:45,634 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:31:45,634 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:31:45,637 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:31:45,637 - INFO - DiffusionTSF initialized:
2026-05-14 18:31:45,637 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:31:45,637 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:31:45,637 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:31:45,660 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 18:32:05,587 - INFO - [warmup] epoch 1/10 | train loss 1.4226 (noise 0.0000, aux 1.4226) | val 1.5415 | lr d=6.99e-05 i=6.99e-05 | 19.0s
2026-05-14 18:35:52,620 - INFO - [joint] epoch 2/10 | train loss 1.3370 (noise 0.0239, aux 1.2736) | val 1.6613 | lr d=6.48e-05 i=6.48e-05 | 207.3s
2026-05-14 18:39:39,665 - INFO - [joint] epoch 3/10 | train loss 1.2139 (noise 0.0191, aux 1.1581) | val 1.7408 | lr d=5.70e-05 i=5.70e-05 | 207.2s
2026-05-14 18:43:26,585 - INFO - [joint] epoch 4/10 | train loss 1.1181 (noise 0.0171, aux 1.0667) | val 1.7626 | lr d=4.71e-05 i=4.71e-05 | 207.1s
2026-05-14 18:47:13,466 - INFO - [joint] epoch 5/10 | train loss 1.0434 (noise 0.0160, aux 0.9952) | val 1.7384 | lr d=3.62e-05 i=3.62e-05 | 207.1s
2026-05-14 18:51:00,406 - INFO - [joint] epoch 6/10 | train loss 0.9864 (noise 0.0155, aux 0.9407) | val 1.7615 | lr d=2.52e-05 i=2.52e-05 | 207.2s
2026-05-14 18:54:47,386 - INFO - [joint] epoch 7/10 | train loss 0.9455 (noise 0.0138, aux 0.9035) | val 1.7337 | lr d=1.53e-05 i=1.53e-05 | 207.2s
2026-05-14 18:54:47,388 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:54:47,692 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 18:54:47,692 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:54:47,693 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:54:47,762 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:54:47,763 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:54:47,763 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:54:47,765 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:54:47,765 - INFO - DiffusionTSF initialized:
2026-05-14 18:54:47,765 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:54:47,765 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:54:47,765 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:54:47,943 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTh2_joint_finetuned_gC.pt (val=1.6495, epoch=2)
Done (worker).
```
