# Run report: 05-14-8730-joint-ft-ETTh2

- Job ID: 14
- Log: results/logs/05-14-8730-joint-ft-ETTh2.log
- Duration: 1h 1m 38s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3578730  task=finetune  log: ./results/logs/05-14-8730-joint-ft-ETTh2.log
Node: kn070  started: 2026-05-14T12:40:16-04:00
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
2026-05-14 12:40:30,262 - INFO - traffic.csv already exists
2026-05-14 12:40:30,332 - INFO - ============================================================
2026-05-14 12:40:30,332 - INFO - Joint finetune (e2e): subset=ETTh2, dim=7, epochs=10, trials=3
2026-05-14 12:40:30,332 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained.pt
2026-05-14 12:40:30,332 - INFO - ============================================================
2026-05-14 12:40:30,754 - INFO - [joint_finetune_ETTh2] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 12:40:30,935 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 12:40:30,936 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 12:40:30,987 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 12:40:31,080 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 12:40:31,080 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 12:40:31,080 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 12:40:31,103 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 12:40:31,105 - INFO - DiffusionTSF initialized:
2026-05-14 12:40:31,105 - INFO -   Variables: 7 (multivariate)
2026-05-14 12:40:31,105 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 12:40:31,106 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 12:40:31,134 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 12:40:55,023 - INFO - [warmup] epoch 1/10 | train loss 263334036170609.9688 (noise 0.0000, aux 263334036170609.9688) | val 654570140656.9449 | lr d=1.16e-04 i=4.35e-04 | 20.4s
2026-05-14 12:44:43,161 - INFO - [joint] epoch 2/10 | train loss 262947691635630.9375 (noise 0.0276, aux 262947691635630.9375) | val 654775421878.1790 | lr d=1.07e-04 i=4.04e-04 | 208.4s
2026-05-14 12:48:30,125 - INFO - [joint] epoch 3/10 | train loss 262945805432937.9375 (noise 0.0188, aux 262945805432937.9375) | val 654221956502.1514 | lr d=9.43e-05 i=3.55e-04 | 207.1s
2026-05-14 12:52:17,119 - INFO - [joint] epoch 4/10 | train loss 262947097767303.5625 (noise 0.0162, aux 262947097767303.5625) | val 654855751787.5460 | lr d=7.79e-05 i=2.93e-04 | 207.2s
2026-05-14 12:56:04,009 - INFO - [joint] epoch 5/10 | train loss 262934928023528.7812 (noise 0.0144, aux 262934928023528.7812) | val 654627950149.5464 | lr d=5.98e-05 i=2.24e-04 | 207.1s
2026-05-14 12:59:50,903 - INFO - [joint] epoch 6/10 | train loss 268544825866672.4375 (noise 0.0130, aux 268544825866672.4375) | val 654863259591.7621 | lr d=4.17e-05 i=1.55e-04 | 207.1s
2026-05-14 13:03:37,777 - INFO - [joint] epoch 7/10 | train loss 262933173773405.0938 (noise 0.0121, aux 262933173773405.0938) | val 654761874276.2858 | lr d=2.54e-05 i=9.29e-05 | 207.1s
2026-05-14 13:07:24,657 - INFO - [joint] epoch 8/10 | train loss 262934739700321.3438 (noise 0.0119, aux 262934739700321.3438) | val 654698797935.9713 | lr d=1.24e-05 i=4.37e-05 | 207.1s
2026-05-14 13:07:24,658 - INFO - Early stopping at epoch 8 (patience=5)
2026-05-14 13:07:24,911 - INFO - [joint_finetune_ETTh2] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 13:07:24,960 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:07:24,961 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:07:24,961 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:07:25,025 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:07:25,025 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:07:25,025 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:07:25,027 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:07:25,027 - INFO - DiffusionTSF initialized:
2026-05-14 13:07:25,027 - INFO -   Variables: 7 (multivariate)
2026-05-14 13:07:25,027 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:07:25,027 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:07:25,048 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 13:07:44,217 - INFO - [warmup] epoch 1/10 | train loss 262696882328935.0312 (noise 0.0000, aux 262696882328935.0312) | val 655027954930.2848 | lr d=2.63e-04 i=1.94e-04 | 18.3s
2026-05-14 13:11:31,178 - INFO - [joint] epoch 2/10 | train loss 262413154080138.7188 (noise 0.0311, aux 262413154080138.7188) | val 653697659514.6071 | lr d=2.44e-04 i=1.80e-04 | 207.2s
2026-05-14 13:15:18,156 - INFO - [joint] epoch 3/10 | train loss 262250254530120.8125 (noise 0.0183, aux 262250254530120.8125) | val 654429667427.6791 | lr d=2.15e-04 i=1.58e-04 | 207.1s
2026-05-14 13:19:05,029 - INFO - [joint] epoch 4/10 | train loss 261460980639445.5000 (noise 0.0153, aux 261460980639445.5000) | val 652135095543.4790 | lr d=1.77e-04 i=1.31e-04 | 207.1s
2026-05-14 13:22:51,930 - INFO - [joint] epoch 5/10 | train loss 259375114816071.0000 (noise 0.0136, aux 259375114816071.0000) | val 659451411751.7173 | lr d=1.36e-04 i=1.00e-04 | 207.1s
2026-05-14 13:26:38,884 - INFO - [joint] epoch 6/10 | train loss 256332631030669.3125 (noise 0.0120, aux 256332631030669.3125) | val 655745411616.7576 | lr d=9.45e-05 i=6.99e-05 | 207.2s
2026-05-14 13:30:25,707 - INFO - [joint] epoch 7/10 | train loss 252782034632296.7500 (noise 0.0110, aux 252782034632296.7500) | val 655505134919.0282 | lr d=5.72e-05 i=4.25e-05 | 207.1s
2026-05-14 13:34:12,684 - INFO - [joint] epoch 8/10 | train loss 248643059019975.5625 (noise 0.0104, aux 248643059019975.5625) | val 655375641801.7827 | lr d=2.76e-05 i=2.07e-05 | 207.2s
2026-05-14 13:37:59,527 - INFO - [joint] epoch 9/10 | train loss 246928838956976.9688 (noise 0.0100, aux 246928838956976.9688) | val 661491967656.3510 | lr d=8.54e-06 i=6.79e-06 | 207.1s
2026-05-14 13:37:59,528 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 13:37:59,783 - INFO - [joint_finetune_ETTh2] Trial 3/3: diffusion_lr=7.16e-05, itrans_lr=7.16e-05
2026-05-14 13:37:59,834 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:37:59,834 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:37:59,834 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:37:59,898 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:37:59,898 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:37:59,898 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:37:59,901 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:37:59,901 - INFO - DiffusionTSF initialized:
2026-05-14 13:37:59,901 - INFO -   Variables: 7 (multivariate)
2026-05-14 13:37:59,901 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:37:59,901 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:37:59,921 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 13:38:20,581 - INFO - [warmup] epoch 1/10 | train loss 262487768538047.4062 (noise 0.0000, aux 262487768538047.4062) | val 657090027569.7908 | lr d=6.99e-05 i=6.99e-05 | 19.8s
2026-05-14 13:42:07,673 - INFO - [joint] epoch 2/10 | train loss 259625090605440.3438 (noise 0.0271, aux 259625090605440.3438) | val 655726387442.6602 | lr d=6.48e-05 i=6.48e-05 | 207.3s
2026-05-14 13:42:07,849 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 13:42:07,849 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 13:42:07,850 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 13:42:07,913 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 13:42:07,913 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 13:42:07,913 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 13:42:07,915 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 13:42:07,915 - INFO - DiffusionTSF initialized:
2026-05-14 13:42:07,915 - INFO -   Variables: 7 (multivariate)
2026-05-14 13:42:07,915 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 13:42:07,915 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 13:42:08,043 - INFO - Joint finetune checkpoint saved: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/ETTh2_joint_finetuned.pt (val=652135095543.4790, epoch=4)
Done (worker).
```
