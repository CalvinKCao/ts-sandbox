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

## Quick start

```bash
source .venv/bin/activate
python models/diffusion_tsf/train_multivariate_pipeline.py --config configs/binary_anchor_stationary_flat_subsets_ema099.yaml
```

Leaf configs under `configs/` extend `configs/base/binary_staged.yaml`. See [Technical Details](architecture.md#technical-details) for production defaults and pitfalls.

## References

- [BDPM — Binary Diffusion Probabilistic Models](reference/BDPM_ref.md)
- [MMPD — Multi-Mode Patch Diffusion Loss](reference/MMPD_methods_reference.md)
