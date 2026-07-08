# Project Instructions
never tell me to export WANDB_API_KEY = ... , i dont need the reminder.
# BEFORE EVERY SINGLE NEW MESSAGE/PROMPT
Run `git branch --show-current` to see what branch you're on. IF THE BRANCH HAS CHANGED SINCE YOUR LAST MESSAGE AND YOU (THE AGENT) DID NOT CHANGE THE BRANCH, SWITCH BACK TO THE ORIGINAL BRANCH, UNLESS THE USER HAS SPECIFICALLY SAID OTHERWISE.

## First thing in every new conversation
1. If Linux, always `alias rm='trash-put'`.
2. Python env: use **`.venv` at repo root** (`source .venv/bin/activate`). Cluster Slurm jobs build or reuse venvs under `$SCRATCH` / `$SLURM_TMPDIR`, not this naming.
3. This is a WSL folder; activate the root venv before running commands.

## Security
- Outside-project reads/edits are allowed only when either:
  - the file/path is explicitly attached as context, or
  - the user explicitly grants permission in chat.
- NEVER NEVER run destructive commands (for example: `rm -rf`, `git reset --hard`, force pushes, or destructive DB/file operations) without asking first and getting explicit approval.
- ALWAYS dry run risky glob/regex expansion commands that might accidentally permanently modify or delete things they're not supposed to.

## Alliance Canada / Slurm
For anything Slurm/Alliance Canada related, ALWAYS use the `/alliancecan` skill first.
If `/alliancecan` does not resolve the question, check `wiki_docs/` for cluster-specific details.

When giving **pull / prep / submit** commands for Killarney (or any cluster), follow the alliancecan skill: **one copy-pastable bash block**, with optional or alternate lines **commented out** (`#`), not split across prose.

## Git hygiene
ALWAYS use the /git skill before using git operations.

## ML smoke tests (not unit-test TDD)
Full test suites are usually impractical for ML. Prefer **smoke tests**: the smallest run that still exercises the real training/eval path end-to-end.

**Goal:** catch dumb failures fast (wrong hyperparams, tensor shape mismatches, broken dataloading, device issues). **Not** to prove the model learns.

**Bare minimum:** **one sample, one epoch** on a **consumer GPU**, finishing in **seconds**. If even one epoch is too heavy for the real loop, **one sample, one batch, one optimizer step** is fine as long as it hits the same forward/backward/optimizer path. Do **not** use “small % of data for multiple epochs” as a stand-in — that can still take hours.

**Design for smoke runs:** add a dedicated smoke flag and/or CLI knobs that slash workload (`--smoke-test`, `--max-samples`, `--max-steps`, batch size 1, etc.). Thread those limits through helpers (datamodule, train step, eval) via parameters or a small smoke config — avoid hard-coding full-scale-only paths. Any non-trivial train/eval entrypoint should reach a **few-second** run when smoke mode is on.

**When to run:** after any change beyond a trivial tweak you are **certain** cannot affect shapes, batching, or the train loop, run the smoke path locally before handing work back.

## Weights & Biases (wandb)
Use `wandb` for training/eval runs tied to this repo.

- **Leaderboard project:** pipeline runs that should appear in the sweep table must use **`ts-sandbox-leaderboard`** (`wandb.project` in YAML, or `configs/base/binary_staged.yaml` default). Do not point Slurm submit scripts at ad-hoc project names unless the user explicitly asks.
- `test_submit.sh` enables wandb when `WANDB_API_KEY` is set but **does not override** `wandb.project` from the config unless you pass `--wandb-project …` on the submit command.
- Staged eval rows: `job_type=staged_eval`, metrics `eval/staged_*` (e.g. `eval/staged_prob_mse`, `eval/staged_anchor_mse`, `eval/staged_crps`). Set `config_nickname` / tags so leaderboard scripts can filter runs.
- **MMPD eval → leaderboard:** campaigns driven by `--mmpd-run-config` should set `mmpd.leaderboard: true` in that YAML. After each dataset worker finishes eval, `utils/eval_mmpd_gaussian_anchor.py` auto-creates an `mmpd_eval` run in `ts-sandbox-leaderboard` (group `MM-DD-{SLURM_JOB_ID}-{dataset}-{config_stem}`, metrics `eval/staged_*`). Requires `WANDB_API_KEY` on the compute node. Skip with `--no-mmpd-leaderboard`; backfill old runs with `python utils/sync_leaderboard_wandb_metadata.py --decoder-grad-accum-only` (or add a matching stub creator for the campaign).
- Use a stable project name via `wandb.init(...)`.
- Standardize the API key env var as `WANDB_API_KEY` (do not depend on key files in this repo).
- Set `name=` to describe what changed vs prior runs; avoid generic names.
- Log all training-relevant hyperparameters in `wandb.config`.
- When resuming from checkpoint, reuse the same run with saved `run_id` and `resume="allow"` or `resume="must"`.
- After evaluation, `wandb.log` final metrics with clear names (example: `eval/mae`, `eval/mse`).
- Attach visualization artifacts to the same run (for example with `wandb.Image`), and compress/downscale images before upload.
- At run end, log the ENTIRE stdout and stderr output files to wandb as artifacts/files.

## Reports & Visualizations
- From now on, put visualizations in the same directory as their corresponding reports (e.g. within `reports/`).
- Create a subfolder with the folder name equal to the report markdown filename (without the `.md` extension).

## General style
- Keep writing terse and detailed; avoid context pollution.
- Comments should sound natural, not AI-generated.
- Do not use flashy/fancy formatting in comments (for example `# ---HEADER---` or `// == PIPELINE PT A ==`).
- Flashy formatting in print statements is okay for clarity, but only use separators when denoting major log subdivisions.
- Never leave wording that makes AI authorship obvious.

## General code style
Always fail fast over adding a million compatability/fallback paths.
