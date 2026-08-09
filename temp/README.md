# temp/

Scratch for one-off cluster helpers and artifacts.

- **Scripts:** `temp/scripts/` — all temp `.sh` / `.py` (diag, viz, migrate, ablation, etc.).
- **Other artifacts:** data dumps, logs, viz outputs, clones (e.g. `temp/MMPD`) live in other `temp/` subdirs — not repo root / `./tmp`.
- Compat symlinks at `temp/<name>.sh|.py` → `scripts/<name>` keep old Killarney paths working.

Examples under `scripts/`:
- `submit_migrate_reused_checkpoints_killarney.sh` — binary + optional MMPD reuse migration
- `submit_migrate_mmpd_lb336_hz96_killarney.sh` — shorthand for lb336/hz96 MMPD paper subset (12 datasets)
- `submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh` — binary grid 4208596–4208599 + MMPD ordinal hz720 (4 datasets)
- `submit_binary_mmpd_diag_lb336_hz720_ordinal_four_killarney.sh` — compare those four after migrate
