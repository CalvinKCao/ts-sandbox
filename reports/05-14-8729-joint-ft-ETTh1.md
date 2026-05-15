# Run report: 05-14-8729-joint-ft-ETTh1

- Job ID: 14
- Log: results/logs/05-14-8729-joint-ft-ETTh1.log
- Duration: 1h 16m 47s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3578729  task=finetune  log: ./results/logs/05-14-8729-joint-ft-ETTh1.log
Node: kn056  started: 2026-05-14T12:37:36-04:00
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
2026-05-14 12:37:44,180 - INFO - traffic.csv already exists
2026-05-14 12:37:44,407 - INFO - ============================================================
2026-05-14 12:37:44,407 - INFO - Joint finetune (e2e): subset=ETTh1, dim=7, epochs=10, trials=3
2026-05-14 12:37:44,407 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained.pt
2026-05-14 12:37:44,407 - INFO - ============================================================
2026-05-14 12:37:44,730 - INFO - [joint_finetune_ETTh1] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 12:37:44,804 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 12:37:44,805 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 12:37:44,806 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 12:37:44,875 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 12:37:44,875 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 12:37:44,875 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 12:37:44,877 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 12:37:44,878 - INFO - DiffusionTSF initialized:
2026-05-14 12:37:44,878 - INFO -   Variables: 7 (multivariate)
2026-05-14 12:37:44,878 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 12:37:44,878 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 12:37:44,899 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 12:38:06,413 - INFO - [warmup] epoch 1/10 | train loss 1.5000 (noise 0.0000, aux 1.5000) | val 266682459729429.8125 | lr d=1.16e-04 i=4.35e-04 | 19.4s
2026-05-14 12:41:54,976 - INFO - [joint] epoch 2/10 | train loss 1.4820 (noise 0.0272, aux 1.4123) | val 266681464217367.4062 | lr d=1.07e-04 i=4.04e-04 | 208.8s
2026-05-14 12:45:42,795 - INFO - [joint] epoch 3/10 | train loss 1.3936 (noise 0.0195, aux 1.3364) | val 266665460862439.0312 | lr d=9.43e-05 i=3.55e-04 | 207.9s
2026-05-14 12:49:30,590 - INFO - [joint] epoch 4/10 | train loss 1.2906 (noise 0.0173, aux 1.2398) | val 266671799007034.4375 | lr d=7.79e-05 i=2.93e-04 | 207.9s
2026-05-14 12:53:17,852 - INFO - [joint] epoch 5/10 | train loss 1.1693 (noise 0.0149, aux 1.1227) | val 266648366125642.4688 | lr d=5.98e-05 i=2.24e-04 | 207.5s
2026-05-14 12:57:05,009 - INFO - [joint] epoch 6/10 | train loss 1.0306 (noise 0.0140, aux 0.9876) | val 266659176404934.8438 | lr d=4.17e-05 i=1.55e-04 | 207.3s
2026-05-14 13:00:52,051 - INFO - [joint] epoch 7/10 | train loss 0.9075 (noise 0.0129, aux 0.8675) | val 266659263838197.6562 | lr d=2.54e-05 i=9.29e-05 | 207.3s
2026-05-14 13:04:39,060 - INFO - [joint] epoch 8/10 | train loss 0.8102 (noise 0.0127, aux 0.7724) | val 266653601383992.9688 | lr d=1.24e-05 i=4.37e-05 | 207.3s
2026-05-14 13:08:26,401 - INFO - [joint] epoch 9/10 | train loss 0.7404 (noise 0.0118, aux 0.7054) | val 266655891521911.8125 | lr d=4.05e-06 i=1.21e-05 | 207.5s
2026-05-14 13:12:13,269 - INFO - [joint] epoch 10/10 | train loss 0.7094 (noise 0.0116, aux 0.6751) | val 266653904332578.9688 | lr d=1.18e-06 i=1.18e-06 | 207.1s
2026-05-14 13:12:13,279 - INFO - Early stopping at epoch 10 (patience=5)
2026-05-14 13:12:13,795 - INFO - [joint_finetune_ETTh1] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 13:12:13,952 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:12:13,952 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:12:13,952 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:12:14,019 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:12:14,019 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:12:14,019 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:12:14,022 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:12:14,022 - INFO - DiffusionTSF initialized:
2026-05-14 13:12:14,022 - INFO -   Variables: 7 (multivariate)
2026-05-14 13:12:14,022 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:12:14,022 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:12:14,043 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 13:12:32,924 - INFO - [warmup] epoch 1/10 | train loss 1.4742 (noise 0.0000, aux 1.4742) | val 266693780036088.5938 | lr d=2.63e-04 i=1.94e-04 | 18.0s
2026-05-14 13:16:19,535 - INFO - [joint] epoch 2/10 | train loss 1.3631 (noise 0.0311, aux 1.2879) | val 266676021113278.0938 | lr d=2.44e-04 i=1.80e-04 | 206.9s
2026-05-14 13:20:06,408 - INFO - [joint] epoch 3/10 | train loss 1.1759 (noise 0.0178, aux 1.1222) | val 266676832555313.6562 | lr d=2.15e-04 i=1.58e-04 | 207.1s
2026-05-14 13:23:53,181 - INFO - [joint] epoch 4/10 | train loss 1.0417 (noise 0.0155, aux 0.9945) | val 266666821446721.7812 | lr d=1.77e-04 i=1.31e-04 | 207.0s
2026-05-14 13:27:39,992 - INFO - [joint] epoch 5/10 | train loss 0.9294 (noise 0.0133, aux 0.8868) | val 266661652913756.0625 | lr d=1.36e-04 i=1.00e-04 | 207.0s
2026-05-14 13:31:26,913 - INFO - [joint] epoch 6/10 | train loss 0.8383 (noise 0.0129, aux 0.7987) | val 266658603486975.9062 | lr d=9.45e-05 i=6.99e-05 | 207.1s
2026-05-14 13:35:13,938 - INFO - [joint] epoch 7/10 | train loss 0.7654 (noise 0.0120, aux 0.7286) | val 266659978643469.1250 | lr d=5.72e-05 i=4.25e-05 | 207.2s
2026-05-14 13:39:01,122 - INFO - [joint] epoch 8/10 | train loss 0.7128 (noise 0.0112, aux 0.6788) | val 266660420411533.8438 | lr d=2.76e-05 i=2.07e-05 | 207.4s
2026-05-14 13:42:48,382 - INFO - [joint] epoch 9/10 | train loss 0.6805 (noise 0.0110, aux 0.6491) | val 266662270315304.9062 | lr d=8.54e-06 i=6.79e-06 | 207.5s
2026-05-14 13:46:35,781 - INFO - [joint] epoch 10/10 | train loss 0.6619 (noise 0.0105, aux 0.6322) | val 266660174217872.8125 | lr d=1.98e-06 i=1.98e-06 | 207.6s
2026-05-14 13:46:36,042 - INFO - [joint_finetune_ETTh1] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 13:46:36,095 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:46:36,095 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:46:36,096 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:46:36,162 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:46:36,162 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:46:36,162 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:46:36,164 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:46:36,164 - INFO - DiffusionTSF initialized:
2026-05-14 13:46:36,164 - INFO -   Variables: 7 (multivariate)
2026-05-14 13:46:36,164 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:46:36,164 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:46:36,185 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 13:46:55,198 - INFO - [warmup] epoch 1/10 | train loss 1.4866 (noise 0.0000, aux 1.4866) | val 266692749243937.5000 | lr d=6.99e-05 i=6.99e-05 | 18.1s
2026-05-14 13:50:42,943 - INFO - [joint] epoch 2/10 | train loss 1.4100 (noise 0.0265, aux 1.3423) | val 266677189190991.0000 | lr d=6.48e-05 i=6.48e-05 | 207.9s
2026-05-14 13:54:30,693 - INFO - [joint] epoch 3/10 | train loss 1.2708 (noise 0.0198, aux 1.2145) | val 266672424078167.8125 | lr d=5.70e-05 i=5.70e-05 | 207.9s
2026-05-14 13:54:30,878 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:54:30,879 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:54:30,879 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:54:30,948 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:54:30,948 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:54:30,948 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:54:30,950 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:54:30,950 - INFO - DiffusionTSF initialized:
2026-05-14 13:54:30,950 - INFO -   Variables: 7 (multivariate)
2026-05-14 13:54:30,950 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:54:30,950 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:54:31,091 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTh1_joint_finetuned.pt (val=266648366125642.4688, epoch=5)
Done (worker).
```
