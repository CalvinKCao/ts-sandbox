# Run report: 05-14-1300-joint-ft-illness-gC

- Job ID: 14
- Log: results/logs/05-14-1300-joint-ft-illness-gC.log
- Duration: 0h 4m 6s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581300  task=finetune  ghost=gC  log: ./results/logs/05-14-1300-joint-ft-illness-gC.log
Node: kn067  started: 2026-05-14T17:55:01-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=illness n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:55:09,389 - INFO - traffic.csv already exists
2026-05-14 17:55:09,468 - INFO - ============================================================
2026-05-14 17:55:09,468 - INFO - Joint finetune (e2e): subset=illness, dim=7, epochs=10, trials=3
2026-05-14 17:55:09,468 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gC.pt
2026-05-14 17:55:09,468 - INFO - ============================================================
2026-05-14 17:55:09,824 - INFO - [joint_finetune_illness_gC] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:55:09,902 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:55:09,902 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:55:09,903 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:55:09,972 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:55:09,972 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:55:09,972 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:55:09,974 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:55:09,975 - INFO - DiffusionTSF initialized:
2026-05-14 17:55:09,975 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:55:09,975 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:55:09,975 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:55:09,997 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:55:12,545 - INFO - [warmup] epoch 1/10 | train loss 0.8321 (noise 0.0000, aux 0.8321) | val 0.3606 | lr d=1.16e-04 i=4.35e-04 | 1.3s
2026-05-14 17:55:25,290 - INFO - [joint] epoch 2/10 | train loss 0.6873 (noise 0.0390, aux 0.5917) | val 0.4634 | lr d=1.07e-04 i=4.04e-04 | 12.7s
2026-05-14 17:55:37,348 - INFO - [joint] epoch 3/10 | train loss 0.5262 (noise 0.0252, aux 0.4592) | val 0.7864 | lr d=9.43e-05 i=3.55e-04 | 12.0s
2026-05-14 17:55:49,283 - INFO - [joint] epoch 4/10 | train loss 0.4285 (noise 0.0206, aux 0.3688) | val 0.4988 | lr d=7.79e-05 i=2.93e-04 | 11.9s
2026-05-14 17:56:01,217 - INFO - [joint] epoch 5/10 | train loss 0.3453 (noise 0.0174, aux 0.2895) | val 0.6808 | lr d=5.98e-05 i=2.24e-04 | 11.9s
2026-05-14 17:56:13,142 - INFO - [joint] epoch 6/10 | train loss 0.2735 (noise 0.0158, aux 0.2203) | val 0.5895 | lr d=4.17e-05 i=1.55e-04 | 11.9s
2026-05-14 17:56:25,076 - INFO - [joint] epoch 7/10 | train loss 0.2348 (noise 0.0150, aux 0.1849) | val 0.6565 | lr d=2.54e-05 i=9.29e-05 | 11.9s
2026-05-14 17:56:25,077 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:56:25,348 - INFO - [joint_finetune_illness_gC] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 17:56:25,404 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:56:25,404 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:56:25,404 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:56:25,479 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:56:25,480 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:56:25,480 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:56:25,482 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:56:25,482 - INFO - DiffusionTSF initialized:
2026-05-14 17:56:25,482 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:56:25,482 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:56:25,482 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:56:25,506 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:56:26,236 - INFO - [warmup] epoch 1/10 | train loss 0.7705 (noise 0.0000, aux 0.7705) | val 0.3762 | lr d=2.63e-04 i=1.94e-04 | 0.7s
2026-05-14 17:56:38,188 - INFO - [joint] epoch 2/10 | train loss 0.6772 (noise 0.0789, aux 0.5397) | val 0.6102 | lr d=2.44e-04 i=1.80e-04 | 11.9s
2026-05-14 17:56:50,211 - INFO - [joint] epoch 3/10 | train loss 0.5176 (noise 0.0341, aux 0.4427) | val 0.6798 | lr d=2.15e-04 i=1.58e-04 | 11.9s
2026-05-14 17:57:02,151 - INFO - [joint] epoch 4/10 | train loss 0.4557 (noise 0.0234, aux 0.3914) | val 0.6026 | lr d=1.77e-04 i=1.31e-04 | 11.9s
2026-05-14 17:57:14,195 - INFO - [joint] epoch 5/10 | train loss 0.3978 (noise 0.0190, aux 0.3377) | val 0.6179 | lr d=1.36e-04 i=1.00e-04 | 12.0s
2026-05-14 17:57:26,157 - INFO - [joint] epoch 6/10 | train loss 0.3562 (noise 0.0203, aux 0.2998) | val 0.7367 | lr d=9.45e-05 i=6.99e-05 | 11.9s
2026-05-14 17:57:38,120 - INFO - [joint] epoch 7/10 | train loss 0.3328 (noise 0.0241, aux 0.2739) | val 0.6058 | lr d=5.72e-05 i=4.25e-05 | 11.9s
2026-05-14 17:57:50,080 - INFO - [joint] epoch 8/10 | train loss 0.3026 (noise 0.0163, aux 0.2493) | val 0.6547 | lr d=2.76e-05 i=2.07e-05 | 11.9s
2026-05-14 17:58:02,032 - INFO - [joint] epoch 9/10 | train loss 0.2838 (noise 0.0134, aux 0.2341) | val 0.6277 | lr d=8.54e-06 i=6.79e-06 | 11.9s
2026-05-14 17:58:02,033 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 17:58:02,323 - INFO - [joint_finetune_illness_gC] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 17:58:02,375 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:58:02,375 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:58:02,375 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:58:02,439 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:58:02,439 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:58:02,439 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:58:02,441 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:58:02,441 - INFO - DiffusionTSF initialized:
2026-05-14 17:58:02,442 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:58:02,442 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:58:02,442 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:58:02,463 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:58:03,214 - INFO - [warmup] epoch 1/10 | train loss 0.8651 (noise 0.0000, aux 0.8651) | val 0.4453 | lr d=6.99e-05 i=6.99e-05 | 0.7s
2026-05-14 17:58:15,296 - INFO - [joint] epoch 2/10 | train loss 0.6746 (noise 0.0209, aux 0.6074) | val 0.5355 | lr d=6.48e-05 i=6.48e-05 | 12.1s
2026-05-14 17:58:27,301 - INFO - [joint] epoch 3/10 | train loss 0.5827 (noise 0.0169, aux 0.5290) | val 0.9338 | lr d=5.70e-05 i=5.70e-05 | 11.9s
2026-05-14 17:58:39,259 - INFO - [joint] epoch 4/10 | train loss 0.5385 (noise 0.0197, aux 0.4846) | val 0.6173 | lr d=4.71e-05 i=4.71e-05 | 11.9s
2026-05-14 17:58:51,210 - INFO - [joint] epoch 5/10 | train loss 0.4977 (noise 0.0182, aux 0.4420) | val 0.5665 | lr d=3.62e-05 i=3.62e-05 | 11.9s
2026-05-14 17:59:03,166 - INFO - [joint] epoch 6/10 | train loss 0.4708 (noise 0.0150, aux 0.4163) | val 0.6282 | lr d=2.52e-05 i=2.52e-05 | 11.9s
2026-05-14 17:59:15,092 - INFO - [joint] epoch 7/10 | train loss 0.4522 (noise 0.0139, aux 0.4014) | val 0.5906 | lr d=1.53e-05 i=1.53e-05 | 11.9s
2026-05-14 17:59:15,093 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 17:59:15,409 - INFO - Joint diffusion model: backbone_in_channels=2 (ghost=C)
2026-05-14 17:59:15,410 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:59:15,410 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:59:15,473 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:59:15,473 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:59:15,473 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:59:15,475 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:59:15,475 - INFO - DiffusionTSF initialized:
2026-05-14 17:59:15,475 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:59:15,475 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:59:15,475 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:59:15,606 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/illness_joint_finetuned_gC.pt (val=0.4634, epoch=2)
Done (worker).
```
