## Diffusion TSF - Occupancy (CDF) Mode Notes

- **Toggle**: `representation_mode` lives in `DiffusionTSFConfig` (model) and `DatasetConfig` (dataset helper). Values: `pdf` (stripe/one-hot) or `cdf` (occupancy map). Defaults remain `pdf`.
- **Encoding (cdf)**:
  - Normalize with existing standardizer, clamp to `[-max_scale, max_scale]`.
  - Map each time step to an integer bin `y` in `[0, H-1]`.
  - Fill all pixels `0..y` as 1.0 (occupancy), rest 0.0.
  - Apply the same vertical Gaussian blur to soften the step into a smooth boundary.
  - For diffusion, the blurred occupancy is clamped to `[0, 1]` and shifted to `[-1, 1]` (no extra gain factor).
- **Decoding (cdf)**:
  - Bring diffusion output back to `[0, 1]` via `(x + 1)/2`, clamp non-negative.
  - Optionally apply decode smoothing if enabled.
  - Sum each column; clamp the sum to `[0, H]`, normalize by `H`, then map back to the normalized value range `[-max_scale, max_scale]`.
  - Final denormalization to the original scale still happens after this step in the model pipeline.
- **Decoding (pdf)**: unchanged — softmax + expectation over bin centers.
- **Why**: Occupancy/CDF mode makes the mass cumulative per column; blur turns the hard edge into a differentiable sigmoid-like transition while preserving the original stripe path for comparison.

