# Legacy submit wrappers → `submit_binary.sh` / `submit_mmpd.sh`

Thin `submit_*.sh` wrappers were removed. Training/eval campaigns use only:

- **`./submit_binary.sh`** — binary / patch-decoder diffusion pipeline (`slurm_worker.sh`)
- **`./submit_mmpd.sh`** — MMPD train + gaussian-anchor eval

Bare config stems resolve under `configs/` (e.g. `--configs foo` → `configs/foo.yaml`).
With `WANDB_API_KEY` set, `submit_binary.sh` defaults `--wandb-project ts-sandbox-leaderboard`.

Diagnostic util launchers (`submit_diagnose_*`, `submit_probe_*`, `submit_binary_mmpd_staged_diag_*`) are separate; they do not train via these entrypoints.

## Binary / patch-decoder

| Old wrapper | Equivalent |
|-------------|------------|
| `test_submit.sh` / `submit_grid.sh` | `./submit_binary.sh` |
| `submit_patch_decoder_lb336_hz720_fixed_killarney.sh` | `./submit_binary.sh --configs binary_anchor_ar_patch_decoder_ctx_lb336_hz720_fixed --datasets ETTh1,electricity,exchange_rate --time 24:00:00` |
| `submit_patch_decoder_lb336_hz720_ordinal_norm_killarney.sh` | `./submit_binary.sh --configs binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm --datasets ETTh1,traffic,weather,dynamic,exchange_rate --time 10:00:00` |
| `submit_patch_decoder_healthy_norm_reduced_hp_killarney.sh` | Removed with its leaf YAML; no submit equivalent. |
| `submit_patch_decoder_lb336_hz720_ord_unc_bs_hp_killarney.sh --tier small` | `./submit_binary.sh --configs binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_small --datasets ETTh1,exchange_rate,traffic --time 1-00:00:00` |
| `… --tier mid` | `…_uncompressed_bs_mid` |
| `… --tier xlarge` | `…_uncompressed_bs_xlarge` |
| `submit_noise_sched_ablation_elec_unc_killarney.sh` | `./submit_binary.sh --configs binary_noise_sched_ablation_elec_unc_g1p0,binary_noise_sched_ablation_elec_unc_g1p5,binary_noise_sched_ablation_elec_unc_g3p0 --datasets electricity --time 3:00:00` |
| `submit_noise_sched_crps_grid_killarney.sh` | `./submit_binary.sh --configs 'configs/binary_noise_sched_ablation_elec_unc_g*.yaml' --datasets ETTh1,traffic,exchange_rate,electricity --time 3:00:00` (refine/confirm seeds: pick stems from that config family) |
| `submit_noise_sched_past_native_crps_grid_killarney.sh` | `./submit_binary.sh --configs 'configs/binary_noise_sched_ablation_past_native_g*.yaml' --datasets …` |
| `submit_uncompressed_crps_g_killarney.sh` | `./submit_binary.sh --configs binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_g3p0` (or `_g7p0`) `--datasets … --time 1-00:00:00` |
| `submit_past_native_crps_g_full_killarney.sh` | `./submit_binary.sh --configs binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native` (or `_past_native_g*`) `--datasets … --time 1-00:00:00` |

## MMPD

| Old wrapper | Equivalent |
|-------------|------------|
| `submit_mmpd_sweep_subset.sh` | `./submit_mmpd.sh` |
| `submit_mmpd_decoder_flat_subsets_paper_lb336_hz720.sh` | `./submit_mmpd.sh --mmpd-run-config mmpd_decoder_flat_subsets_paper_lb336_hz720 --output-dir results/datasets/$(date +%m-%d)-mmpd-decoder-paper-lb336-hz720-subset --time 24:00:00 --mmpd-tune-trials 0` |
| `submit_mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.sh` | `./submit_mmpd.sh --mmpd-run-config mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm --output-dir results/datasets/$(date +%m-%d)-mmpd-decoder-ordinal-norm-lb336-hz720 --time 24:00:00 --mmpd-tune-trials 0 --no-mmpd-instance-norm` |
| `submit_mmpd_decoder_flat_subsets_paper_lb336_hz96.sh` | `./submit_mmpd.sh --mmpd-run-config mmpd_decoder_flat_subsets_paper_lb336_hz96 --output-dir results/datasets/$(date +%m-%d)-mmpd-decoder-paper-lb336-hz96-subset --mmpd-tune-trials 0` |
| `submit_mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.sh` | `./submit_mmpd.sh --mmpd-run-config mmpd_decoder_flat_subsets_grad_accum_200_lr_lo --output-dir results/datasets/$(date +%m-%d)-mmpd-decoder-grad-accum-200-lr-lo-subset --time 3:00:00 --mmpd-tune-trials 7` |

## Configs

Leaf YAMLs under `configs/` were **not** renamed. Use the same stems as before with `--configs` / `--mmpd-run-config` (bare stem or `configs/<stem>.yaml`). Geometry and HPs stay in YAML — new experiment = new leaf config, not a new `.sh`.
