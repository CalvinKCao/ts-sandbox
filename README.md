# ts-sandbox

Probabilistic multivariate time series forecasting via **binary staged diffusion** on hard CDF maps. Future values are encoded as 2D binary images, denoised with XOR bit-flip diffusion (BCE loss), and decoded back to 1D forecasts.

The model is strong on discontinuities, flat segments, step functions, and geometric patterns that Gaussian MSE baselines tend to blur.

## Documentation

**[Architecture overview →](architecture.md)**

That doc covers:

- Training phases (synthetic pretrain → iTransformer → coarse/fine finetune → eval)
- End-to-end data flow with diagrams
- Binary CDF representation (BDPM-inspired) and dual-scale decomposition
- MMPD-style anchor mechanism (binary flat adaptation)
- Strengths, tradeoffs, and analysis
- Technical details for developers (config defaults, file map, pipeline phases)

## Current experiment (Killarney)

Live campaign leaf: `configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10.yaml`

- Canvas **128**, coarse **H=16**, refine patch **64×6** (8 coarse bins tall), stride 5
- Window-norm + `max_scale` encoding (not ordinal)
- Full train windows and all variate crops (`*_fraction: 1.0`)
- 10 Optuna trials × 20 epochs per diffusion stage (patience 8); next stage loads the best trial checkpoint; no refit
- Search: donor LR ±10×, effective univariate batch ±1 power of 2, EMA default ± one neighbor; donor trial enqueued
- Subsets: ETTh1 uses the allv-randwin window caps in that YAML; traffic uses `train_max_windows: 266`

Ordinal match-up (same geometry, strides, HP policy) lives on branch `exp/ordinal-c128-p64x6` with leaf `configs/binary_ordinal_patch_refine_canvas128_p64x6_allv_randwin_lr10.yaml`.

```bash
cd "$SCRATCH/ts-sandbox"
source .venv/bin/activate
git checkout main
git pull

./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10 \
    --datasets ETTh1,traffic --time 3-00:00:00
# ./submit_binary.sh --gpu h100 --configs binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10 \
#   --datasets ETTh1,traffic --time 3-00:00:00

# Ordinal comparison (after checking out exp/ordinal-c128-p64x6):
# git checkout exp/ordinal-c128-p64x6
# git pull
# ./submit_binary.sh --configs binary_ordinal_patch_refine_canvas128_p64x6_allv_randwin_lr10 \
#   --datasets ETTh1,traffic --time 3-00:00:00
```

Local debug (one dataset, no Slurm):

```bash
source .venv/bin/activate
python models/diffusion_tsf/train_multivariate_pipeline.py \
  --config configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10.yaml \
  --dataset ETTh1
```

Leaf configs under `configs/` extend `configs/base/binary_staged.yaml`. See [Technical Details](architecture.md#technical-details) for production defaults and pitfalls.

## References

- [BDPM — Binary Diffusion Probabilistic Models](reference/BDPM_ref.md)
- [MMPD — Multi-Mode Patch Diffusion Loss](reference/MMPD_methods_reference.md)
