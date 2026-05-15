# Run report: 05-14-1295-joint-ft-ETTh1-gC

- Job ID: 14
- Log: results/logs/05-14-1295-joint-ft-ETTh1-gC.log
- Duration: 1h 8m 41s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581295  task=finetune  ghost=gC  log: ./results/logs/05-14-1295-joint-ft-ETTh1-gC.log
Node: kn052  started: 2026-05-14T17:12:31-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=ETTh1 n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:12:38,732 - INFO - traffic.csv already exists
2026-05-14 17:12:38,812 - INFO - ============================================================
2026-05-14 17:12:38,812 - INFO - Joint finetune (e2e): subset=ETTh1, dim=7, epochs=10, trials=3
2026-05-14 17:12:38,812 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gC.pt
2026-05-14 17:12:38,812 - INFO - ============================================================
2026-05-14 17:12:39,125 - INFO - [joint_finetune_ETTh1_gC] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:12:39,197 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:12:39,198 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:12:39,198 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:12:39,269 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:12:39,269 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:12:39,269 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:12:39,272 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:12:39,272 - INFO - DiffusionTSF initialized:
2026-05-14 17:12:39,272 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:12:39,272 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:12:39,272 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:12:39,293 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:12:59,668 - INFO - [warmup] epoch 1/10 | train loss 1.2459 (noise 0.0000, aux 1.2459) | val 1.4821 | lr d=1.16e-04 i=4.35e-04 | 18.4s
2026-05-14 17:16:45,733 - INFO - [joint] epoch 2/10 | train loss 1.2228 (noise 0.0245, aux 1.1572) | val 1.5377 | lr d=1.07e-04 i=4.04e-04 | 206.5s
2026-05-14 17:20:31,369 - INFO - [joint] epoch 3/10 | train loss 1.1380 (noise 0.0193, aux 1.0816) | val 1.5724 | lr d=9.43e-05 i=3.55e-04 | 206.0s
2026-05-14 17:24:16,853 - INFO - [joint] epoch 4/10 | train loss 1.0365 (noise 0.0176, aux 0.9857) | val 1.5584 | lr d=7.79e-05 i=2.93e-04 | 205.9s
2026-05-14 17:28:02,408 - INFO - [joint] epoch 5/10 | train loss 0.9214 (noise 0.0163, aux 0.8733) | val 1.5468 | lr d=5.98e-05 i=2.24e-04 | 206.0s
2026-05-14 17:31:47,856 - INFO - [joint] epoch 6/10 | train loss 0.8039 (noise 0.0155, aux 0.7594) | val 1.5664 | lr d=4.17e-05 i=1.55e-04 | 205.9s
2026-05-14 17:35:33,389 - INFO - [joint] epoch 7/10 | train loss 0.7021 (noise 0.0145, aux 0.6601) | val 1.5692 | lr d=2.54e-05 i=9.29e-05 | 206.0s
2026-05-14 17:35:33,393 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:35:33,650 - INFO - [joint_finetune_ETTh1_gC] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 17:35:33,701 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:35:33,701 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:35:33,701 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:35:33,769 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:35:33,769 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:35:33,769 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:35:33,771 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:35:33,772 - INFO - DiffusionTSF initialized:
2026-05-14 17:35:33,772 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:35:33,772 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:35:33,772 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:35:33,800 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:35:52,460 - INFO - [warmup] epoch 1/10 | train loss 1.2179 (noise 0.0000, aux 1.2179) | val 1.4718 | lr d=2.63e-04 i=1.94e-04 | 17.8s
2026-05-14 17:39:38,503 - INFO - [joint] epoch 2/10 | train loss 1.1333 (noise 0.0286, aux 1.0607) | val 1.5875 | lr d=2.44e-04 i=1.80e-04 | 206.4s
2026-05-14 17:43:24,199 - INFO - [joint] epoch 3/10 | train loss 0.9735 (noise 0.0188, aux 0.9183) | val 1.6682 | lr d=2.15e-04 i=1.58e-04 | 206.0s
2026-05-14 17:47:09,776 - INFO - [joint] epoch 4/10 | train loss 0.8521 (noise 0.0176, aux 0.8011) | val 1.6320 | lr d=1.77e-04 i=1.31e-04 | 206.0s
2026-05-14 17:50:55,399 - INFO - [joint] epoch 5/10 | train loss 0.7588 (noise 0.0159, aux 0.7114) | val 1.6350 | lr d=1.36e-04 i=1.00e-04 | 206.0s
2026-05-14 17:54:40,983 - INFO - [joint] epoch 6/10 | train loss 0.6829 (noise 0.0140, aux 0.6400) | val 1.6350 | lr d=9.45e-05 i=6.99e-05 | 206.0s
2026-05-14 17:58:26,640 - INFO - [joint] epoch 7/10 | train loss 0.6260 (noise 0.0138, aux 0.5857) | val 1.6315 | lr d=5.72e-05 i=4.25e-05 | 206.1s
2026-05-14 17:58:26,641 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:58:26,899 - INFO - [joint_finetune_ETTh1_gC] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 17:58:26,956 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:58:26,956 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:58:26,957 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:58:27,026 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:58:27,026 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:58:27,026 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:58:27,028 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:58:27,028 - INFO - DiffusionTSF initialized:
2026-05-14 17:58:27,029 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:58:27,029 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:58:27,029 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:58:27,049 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:58:45,944 - INFO - [warmup] epoch 1/10 | train loss 1.2349 (noise 0.0000, aux 1.2349) | val 1.4382 | lr d=6.99e-05 i=6.99e-05 | 18.0s
2026-05-14 18:02:31,490 - INFO - [joint] epoch 2/10 | train loss 1.1687 (noise 0.0239, aux 1.1056) | val 1.5497 | lr d=6.48e-05 i=6.48e-05 | 206.0s
2026-05-14 18:06:17,304 - INFO - [joint] epoch 3/10 | train loss 1.0634 (noise 0.0199, aux 1.0069) | val 1.6124 | lr d=5.70e-05 i=5.70e-05 | 206.1s
2026-05-14 18:10:02,949 - INFO - [joint] epoch 4/10 | train loss 0.9769 (noise 0.0179, aux 0.9251) | val 1.6061 | lr d=4.71e-05 i=4.71e-05 | 206.1s
2026-05-14 18:13:48,365 - INFO - [joint] epoch 5/10 | train loss 0.9147 (noise 0.0169, aux 0.8663) | val 1.6223 | lr d=3.62e-05 i=3.62e-05 | 205.8s
2026-05-14 18:17:33,879 - INFO - [joint] epoch 6/10 | train loss 0.8683 (noise 0.0165, aux 0.8216) | val 1.6171 | lr d=2.52e-05 i=2.52e-05 | 205.9s
2026-05-14 18:21:19,329 - INFO - [joint] epoch 7/10 | train loss 0.8301 (noise 0.0148, aux 0.7873) | val 1.6284 | lr d=1.53e-05 i=1.53e-05 | 205.9s
2026-05-14 18:21:19,331 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 18:21:19,652 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 18:21:19,652 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 18:21:19,652 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 18:21:19,722 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 18:21:19,722 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 18:21:19,722 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 18:21:19,726 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 18:21:19,727 - INFO - DiffusionTSF initialized:
2026-05-14 18:21:19,727 - INFO -   Variables: 7 (multivariate)
2026-05-14 18:21:19,727 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 18:21:19,727 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 18:21:19,875 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTh1_joint_finetuned_gC.pt (val=1.5377, epoch=2)
Done (worker).
```
