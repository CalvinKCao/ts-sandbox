# Oracle-coarse ordinal patch refinement kill test

## Geometry
1. Encode ordinal ranks → hi-res CDF `(H, W=16)` and coarse `(16, W)`.
2. Vertical-only NN upsample coarse → `(H, W)`.
3. Overlapping **8×8** crops with coarse transition edge at local row 4; skip canvas-OOB.

## Refiner (binary diffusion)
`FactorizedDiT` trained with XOR bit-flip noise (linear schedule, T=1000, min-SNR), dual-head BCE (x0+zt), conditioned on `[naive_upscale, past_hist]`. Inference: iterative `quad_t` sample (20 steps).

## Discriminator
1D ordinal-rank **refined vs GT** only (`InvertedSliceDiscriminator`).

```bash
python -m experiments.ordinal_patch_refinement_killtest.smoke --steps 30 --resolution 256
./submit_ordinal_patch_refinement_full_narval.sh --dataset ETTh1 --resolution 256
```
