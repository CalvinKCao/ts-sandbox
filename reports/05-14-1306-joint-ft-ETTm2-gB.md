# Run report: 05-14-1306-joint-ft-ETTm2-gB

- Job ID: 14
- Log: results/logs/05-14-1306-joint-ft-ETTm2-gB.log
- Duration: 3h 40m 53s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3581306  task=finetune  ghost=gB  log: ./results/logs/05-14-1306-joint-ft-ETTm2-gB.log
Node: kn079  started: 2026-05-14T17:39:20-04:00
==========================================
The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) CCconfig        6)  ucx/1.14.1         11) flexiblas/3.3.1
  2) gentoo/2023     7)  libfabric/1.18.0   12) imkl/2023.2.0
  3) gcccore/.12.3   8)  pmix/4.2.4         13) StdEnv/2023
  4) gcc/12.3        9)  ucc/1.2.0
  5) hwloc/2.9.1     10) openmpi/4.1.5
Activated cluster venv: /project/6101823/ccao87/diffusion-tsf/venv
========== joint finetune dataset=ETTm2 n_variates=7 ==========
Using legacy checkpoints_7var; new runs use checkpoints_multivariate — rename or migrate when ready.
Using legacy results_7var; new runs use results_multivariate — rename or migrate when ready.
2026-05-14 17:39:33,869 - INFO - traffic.csv already exists
2026-05-14 17:39:33,962 - INFO - ============================================================
2026-05-14 17:39:33,962 - INFO - Joint finetune (e2e): subset=ETTm2, dim=7, epochs=10, trials=3
2026-05-14 17:39:33,962 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained_gB.pt
2026-05-14 17:39:33,962 - INFO - ============================================================
2026-05-14 17:39:34,571 - INFO - [joint_finetune_ETTm2_gB] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 17:39:34,785 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 17:39:34,786 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 17:39:34,841 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 17:39:34,940 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 17:39:34,943 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 17:39:34,943 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 17:39:34,970 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 17:39:34,972 - INFO - DiffusionTSF initialized:
2026-05-14 17:39:34,972 - INFO -   Variables: 7 (multivariate)
2026-05-14 17:39:34,972 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 17:39:34,972 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 17:39:35,005 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 17:41:05,321 - INFO - [warmup] epoch 1/10 | train loss 1.4160 (noise 0.0000, aux 1.4160) | val 1.5450 | lr d=1.16e-04 i=4.35e-04 | 83.9s
2026-05-14 17:56:34,225 - INFO - [joint] epoch 2/10 | train loss 1.4045 (noise 0.0165, aux 1.3529) | val 1.5490 | lr d=1.07e-04 i=4.04e-04 | 847.3s
2026-05-14 18:12:02,718 - INFO - [joint] epoch 3/10 | train loss 1.3698 (noise 0.0113, aux 1.3302) | val 1.5198 | lr d=9.43e-05 i=3.55e-04 | 846.9s
2026-05-14 18:27:31,373 - INFO - [joint] epoch 4/10 | train loss 1.3496 (noise 0.0101, aux 1.3144) | val 1.5253 | lr d=7.79e-05 i=2.93e-04 | 847.1s
2026-05-14 18:42:59,784 - INFO - [joint] epoch 5/10 | train loss 1.3165 (noise 0.0090, aux 1.2846) | val 1.5301 | lr d=5.98e-05 i=2.24e-04 | 847.0s
2026-05-14 18:58:28,161 - INFO - [joint] epoch 6/10 | train loss 1.2689 (noise 0.0080, aux 1.2401) | val 1.5317 | lr d=4.17e-05 i=1.55e-04 | 847.0s
2026-05-14 19:13:56,663 - INFO - [joint] epoch 7/10 | train loss 1.2080 (noise 0.0074, aux 1.1818) | val 1.5511 | lr d=2.54e-05 i=9.29e-05 | 847.1s
2026-05-14 19:29:25,406 - INFO - [joint] epoch 8/10 | train loss 1.1330 (noise 0.0069, aux 1.1095) | val 1.5801 | lr d=1.24e-05 i=4.37e-05 | 847.2s
2026-05-14 19:29:25,407 - INFO - Early stopping at epoch 8 (patience=5)
2026-05-14 19:29:25,665 - INFO - [joint_finetune_ETTm2_gB] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 19:29:25,717 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 19:29:25,717 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 19:29:25,718 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 19:29:25,782 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 19:29:25,783 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 19:29:25,783 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 19:29:25,784 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 19:29:25,785 - INFO - DiffusionTSF initialized:
2026-05-14 19:29:25,785 - INFO -   Variables: 7 (multivariate)
2026-05-14 19:29:25,785 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 19:29:25,785 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 19:29:25,806 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 19:30:49,416 - INFO - [warmup] epoch 1/10 | train loss 1.3593 (noise 0.0000, aux 1.3593) | val 1.4966 | lr d=2.63e-04 i=1.94e-04 | 80.1s
2026-05-14 19:46:17,829 - INFO - [joint] epoch 2/10 | train loss 1.2768 (noise 0.0173, aux 1.2248) | val 1.5521 | lr d=2.44e-04 i=1.80e-04 | 846.8s
2026-05-14 20:01:46,383 - INFO - [joint] epoch 3/10 | train loss 1.1421 (noise 0.0105, aux 1.1038) | val 1.5885 | lr d=2.15e-04 i=1.58e-04 | 847.0s
2026-05-14 20:17:14,578 - INFO - [joint] epoch 4/10 | train loss 1.0140 (noise 0.0089, aux 0.9805) | val 1.6068 | lr d=1.77e-04 i=1.31e-04 | 846.7s
2026-05-14 20:32:42,690 - INFO - [joint] epoch 5/10 | train loss 0.8934 (noise 0.0076, aux 0.8641) | val 1.6005 | lr d=1.36e-04 i=1.00e-04 | 846.7s
2026-05-14 20:48:10,886 - INFO - [joint] epoch 6/10 | train loss 0.7977 (noise 0.0072, aux 0.7711) | val 1.6261 | lr d=9.45e-05 i=6.99e-05 | 846.7s
2026-05-14 21:03:39,089 - INFO - [joint] epoch 7/10 | train loss 0.7205 (noise 0.0068, aux 0.6963) | val 1.6090 | lr d=5.72e-05 i=4.25e-05 | 846.7s
2026-05-14 21:03:39,090 - INFO - Early stopping at epoch 7 (patience=5)
2026-05-14 21:03:39,345 - INFO - [joint_finetune_ETTm2_gB] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 21:03:39,395 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 21:03:39,395 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:03:39,396 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:03:39,460 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:03:39,460 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:03:39,460 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:03:39,462 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:03:39,462 - INFO - DiffusionTSF initialized:
2026-05-14 21:03:39,462 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:03:39,462 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:03:39,462 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:03:39,483 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 21:04:58,955 - INFO - [warmup] epoch 1/10 | train loss 1.3587 (noise 0.0000, aux 1.3587) | val 1.4802 | lr d=6.99e-05 i=6.99e-05 | 76.0s
2026-05-14 21:20:26,431 - INFO - [joint] epoch 2/10 | train loss 1.2874 (noise 0.0167, aux 1.2358) | val 1.5709 | lr d=6.48e-05 i=6.48e-05 | 846.0s
2026-05-14 21:20:26,614 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 21:20:26,614 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 21:20:26,614 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 21:20:26,678 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 21:20:26,679 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 21:20:26,679 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 21:20:26,681 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 21:20:26,681 - INFO - DiffusionTSF initialized:
2026-05-14 21:20:26,681 - INFO -   Variables: 7 (multivariate)
2026-05-14 21:20:26,681 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 21:20:26,681 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 21:20:26,809 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTm2_joint_finetuned_gB.pt (val=1.5198, epoch=3)
Done (worker).
```
