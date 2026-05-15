# Run report: 05-14-8732-joint-ft-ETTm2

- Job ID: 14
- Log: results/logs/05-14-8732-joint-ft-ETTm2.log
- Duration: 3h 22m 32s

No final dataset metrics parsed from the log.

## Last 100 lines (stats and errors)

```==========================================
Job ID: 3578732  task=finetune  log: ./results/logs/05-14-8732-joint-ft-ETTm2.log
Node: kn047  started: 2026-05-14T12:45:27-04:00
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
2026-05-14 12:45:35,617 - INFO - traffic.csv already exists
2026-05-14 12:45:35,701 - INFO - ============================================================
2026-05-14 12:45:35,701 - INFO - Joint finetune (e2e): subset=ETTm2, dim=7, epochs=10, trials=3
2026-05-14 12:45:35,701 - INFO - Loading joint pretrain checkpoint: /scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_7var/pretrained_dim7/joint_pretrained.pt
2026-05-14 12:45:35,701 - INFO - ============================================================
2026-05-14 12:45:36,102 - INFO - [joint_finetune_ETTm2] Trial 1/3: diffusion_lr=1.18e-04, itrans_lr=4.46e-04
2026-05-14 12:45:36,177 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 12:45:36,178 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 12:45:36,178 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 12:45:36,249 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 12:45:36,249 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 12:45:36,249 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 12:45:36,251 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 12:45:36,251 - INFO - DiffusionTSF initialized:
2026-05-14 12:45:36,251 - INFO -   Variables: 7 (multivariate)
2026-05-14 12:45:36,251 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 12:45:36,251 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 12:45:36,272 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 12:46:59,477 - INFO - [warmup] epoch 1/10 | train loss 108884593170629.5469 (noise 0.0000, aux 108884593170629.5469) | val 3333793734656.4932 | lr d=1.16e-04 i=4.35e-04 | 78.6s
2026-05-14 13:02:23,963 - INFO - [joint] epoch 2/10 | train loss 108886293393160.5938 (noise 0.0177, aux 108886293393160.5938) | val 3333461721639.5894 | lr d=1.07e-04 i=4.04e-04 | 843.4s
2026-05-14 13:17:46,009 - INFO - [joint] epoch 3/10 | train loss 108886874044669.0625 (noise 0.0113, aux 108886874044669.0625) | val 3333423270537.1914 | lr d=9.43e-05 i=3.55e-04 | 841.1s
2026-05-14 13:33:07,561 - INFO - [joint] epoch 4/10 | train loss 108886915256771.3438 (noise 0.0101, aux 108886915256771.3281) | val 3333390050852.7422 | lr d=7.79e-05 i=2.93e-04 | 840.7s
2026-05-14 13:48:28,080 - INFO - [joint] epoch 5/10 | train loss 108886974741144.5625 (noise 0.0088, aux 108886974741144.5625) | val 3333416911653.8555 | lr d=5.98e-05 i=2.24e-04 | 839.8s
2026-05-14 14:03:47,786 - INFO - [joint] epoch 6/10 | train loss 108886991874804.1250 (noise 0.0081, aux 108886991874804.1250) | val 3333406901925.0693 | lr d=4.17e-05 i=1.55e-04 | 839.1s
2026-05-14 14:19:08,010 - INFO - [joint] epoch 7/10 | train loss 108887003315266.6094 (noise 0.0074, aux 108887003315266.6094) | val 3333401645477.2041 | lr d=2.54e-05 i=9.29e-05 | 839.5s
2026-05-14 14:34:29,903 - INFO - [joint] epoch 8/10 | train loss 108887018057328.4062 (noise 0.0070, aux 108887018057328.4062) | val 3333397233895.1313 | lr d=1.24e-05 i=4.37e-05 | 840.9s
2026-05-14 14:49:52,480 - INFO - [joint] epoch 9/10 | train loss 108887018125971.2500 (noise 0.0066, aux 108887018125971.2344) | val 3333397054047.4629 | lr d=4.05e-06 i=1.21e-05 | 841.8s
2026-05-14 14:49:52,481 - INFO - Early stopping at epoch 9 (patience=5)
2026-05-14 14:49:52,734 - INFO - [joint_finetune_ETTm2] Trial 2/3: diffusion_lr=2.70e-04, itrans_lr=1.98e-04
2026-05-14 14:49:52,788 - INFO - Joint diffusion model: backbone_in_channels=3 (ghost=B)
2026-05-14 14:49:52,789 - INFO - TimeSeriesTo2D initialized: H=64, MS=3.5
2026-05-14 14:49:52,789 - INFO - VerticalGaussianBlur initialized: kernel_size=31, sigma=1.0
2026-05-14 14:49:52,854 - INFO - ConditionalUNet2D initialized with channels=[64, 128, 256], kernel_size=(3, 3)
2026-05-14 14:49:52,854 - INFO -   Visual concat: 1 past image channels directly concatenated
2026-05-14 14:49:52,854 - INFO -   attention_levels=[2], context_dim=256
2026-05-14 14:49:52,856 - INFO - DiffusionScheduler initialized: T=1000, schedule=linear
2026-05-14 14:49:52,856 - INFO - DiffusionTSF initialized:
2026-05-14 14:49:52,856 - INFO -   Variables: 7 (multivariate)
2026-05-14 14:49:52,856 - INFO -   Lookback: 96, Forecast: 104
2026-05-14 14:49:52,856 - INFO -   Image size: 64 x 104 (H x W; denoised future canvas)
2026-05-14 14:49:52,877 - INFO - Joint training param groups: iTrans=6.67M, diffusion=10.55M
2026-05-14 14:51:16,550 - INFO - [warmup] epoch 1/10 | train loss 108885414994392.3281 (noise 0.0000, aux 108885414994392.3281) | val 3333646093139.7085 | lr d=2.63e-04 i=1.94e-04 | 80.2s
2026-05-14 15:06:38,236 - INFO - [joint] epoch 2/10 | train loss 108886052082181.9844 (noise 0.0176, aux 108886052082181.9844) | val 3333412555980.3960 | lr d=2.44e-04 i=1.80e-04 | 840.9s
2026-05-14 15:21:59,716 - INFO - [joint] epoch 3/10 | train loss 108886526870048.7344 (noise 0.0107, aux 108886526870048.7344) | val 3333448059445.4365 | lr d=2.15e-04 i=1.58e-04 | 840.6s
2026-05-14 15:37:22,262 - INFO - [joint] epoch 4/10 | train loss 108886744839263.4844 (noise 0.0087, aux 108886744839263.4531) | val 3333378548451.3384 | lr d=1.77e-04 i=1.31e-04 | 841.7s
2026-05-14 15:52:44,899 - INFO - [joint] epoch 5/10 | train loss 108886812198952.9062 (noise 0.0080, aux 108886812198952.8906) | val 3333401772179.0649 | lr d=1.36e-04 i=1.00e-04 | 841.6s
2026-05-14 16:08:07,291 - INFO - [joint] epoch 6/10 | train loss 108886809974517.3125 (noise 0.0073, aux 108886809974517.3125) | val 3333385455560.6909 | lr d=9.45e-05 i=6.99e-05 | 841.5s
```
