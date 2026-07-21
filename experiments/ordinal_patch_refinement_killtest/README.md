# Oracle-coarse ordinal patch-refinement kill test

`smoke.py` is the gated first stage: it loads one ETTh1 test window through the repository dataset path, creates a ground-truth 16-bin CDF oracle, enlarges it to 256x256 by nearest neighbour, and trains the existing `FactorizedDiT` locally on its sixteen input-coordinate patches.

The high-resolution target is independently encoded from the ordinal future at 256 value bins; it is never interpolated from the coarse map. GT, naïve, and refined decodes all use the same global ordinal ladder and snapping behavior as the discriminator tooling. Results are saved under `results/ordinal_patch_refinement_killtest/smoke`.

Run locally:

```powershell
.venv\Scripts\python.exe experiments\ordinal_patch_refinement_killtest\smoke.py
```

This stage deliberately does not generate a held-out pack or run the discriminator. Those are post-smoke work, pending inspection of the saved arrays and figures.
