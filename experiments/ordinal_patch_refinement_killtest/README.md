# Oracle-coarse ordinal patch refinement kill test

Vertical-only geometry (no horizontal stretch):

1. Encode ordinal ranks to a hi-res CDF `(H, W=horizon)` with `H ∈ {256, 512}`.
2. Encode the same ranks to GT coarse `(16, W)`.
3. Nearest-upsample coarse **vertically only** to `(H, W)`.
4. Train `FactorizedDiT` on in-bounds overlapping **8×8** crops (column stride 2); skip crops that would pad past the `(H, W)` canvas edge.

Train forecast windows use pack `train_stride=2` (overlapping). Val/test use non-overlapping futures. All splits apply the same 8×8 OOB filter.

```bash
# Local smoke (GPU if available)
python -m experiments.ordinal_patch_refinement_killtest.smoke --steps 5 --resolution 256 \
  --output results/ordinal_patch_refinement_killtest/vert8_smoke_256
python -m experiments.ordinal_patch_refinement_killtest.smoke --steps 5 --resolution 512 \
  --output results/ordinal_patch_refinement_killtest/vert8_smoke_512

# Full Narval jobs
./submit_ordinal_patch_refinement_full_narval.sh --dataset ETTh1 --resolution 256
./submit_ordinal_patch_refinement_full_narval.sh --dataset ETTh1 --resolution 512
./submit_ordinal_patch_refinement_full_narval.sh --dataset exchange_rate --resolution 256
./submit_ordinal_patch_refinement_full_narval.sh --dataset exchange_rate --resolution 512
```
