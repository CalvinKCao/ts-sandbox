# Project Instructions

## First thing in every new conversation
1. If Linux, always `alias rm='trash-put'`.
2. Python env: use **`.venv` at repo root** (`source .venv/bin/activate`). If both `.venv/` and `venv/` exist, prefer `.venv`; the loose `venv/` copy is legacy/duplicate. Cluster Slurm jobs build or reuse venvs under `$SCRATCH` / `$SLURM_TMPDIR`, not this naming.
3. This is a WSL folder; activate the root venv before running commands.
4. Read this file before doing substantial work.

## Security
- Stay inside this project by default.
- Outside-project reads/edits are allowed only when either:
  - the file/path is explicitly attached as context, or
  - the user explicitly grants permission in chat.
- NEVER NEVER run destructive commands (for example: `rm -rf`, `git reset --hard`, force pushes, or destructive DB/file operations) without asking first and getting explicit approval.

## Alliance Canada / Slurm
For anything Slurm/Alliance Canada related, ALWAYS ALWAYS use the `/alliancecan` skill first (`.ai/skills/alliancecan/SKILL.md`).
If `/alliancecan` does not resolve the question, check `wiki_docs/` for cluster-specific details.

## Git hygiene
If you generate obvious junk, oversized artifacts, or useless throwaway files, **add ignore patterns to `.gitignore` in the same change** (or delete them) so they do not get committed.

## Wrap-up
After multi-step work, use the **`/git`** skill (`.ai/skills/git/SKILL.md`) to produce semantic commits and **push** to the tracked branch.

## ML smoke tests (not unit-test TDD)
Full test suites are usually impractical for ML. Prefer **smoke tests**: the smallest run that still exercises the real training/eval path end-to-end.

**Goal:** catch dumb failures fast (wrong hyperparams, tensor shape mismatches, broken dataloading, device issues). **Not** to prove the model learns.

**Bare minimum:** **one sample, one epoch** on a **consumer GPU**, finishing in **seconds**. If even one epoch is too heavy for the real loop, **one sample, one batch, one optimizer step** is fine as long as it hits the same forward/backward/optimizer path. Do **not** use “small % of data for multiple epochs” as a stand-in — that can still take hours.

**Design for smoke runs:** add a dedicated smoke flag and/or CLI knobs that slash workload (`--smoke-test`, `--max-samples`, `--max-steps`, batch size 1, etc.). Thread those limits through helpers (datamodule, train step, eval) via parameters or a small smoke config — avoid hard-coding full-scale-only paths. Any non-trivial train/eval entrypoint should reach a **few-second** run when smoke mode is on.

**When to run:** after any change beyond a trivial tweak you are **certain** cannot affect shapes, batching, or the train loop, run the smoke path locally before handing work back.

## Weights & Biases (wandb)
Use `wandb` for training/eval runs tied to this repo.

- Use a stable project name via `wandb.init(...)`.
- Standardize the API key env var as `WANDB_API_KEY` (do not depend on key files in this repo).
- Set `name=` to describe what changed vs prior runs; avoid generic names.
- Log all training-relevant hyperparameters in `wandb.config`.
- When resuming from checkpoint, reuse the same run with saved `run_id` and `resume="allow"` or `resume="must"`.
- After evaluation, `wandb.log` final metrics with clear names (example: `eval/mae`, `eval/mse`).
- Attach visualization artifacts to the same run (for example with `wandb.Image`), and compress/downscale images before upload.
- At run end, log the ENTIRE stdout and stderr output files to wandb as artifacts/files.

Implement these behaviors in training, eval, and visualization scripts whenever experiment code changes.

## General style
- Keep writing terse and detailed; avoid context pollution.
- Comments should sound natural, not AI-generated.
- Do not use flashy/fancy formatting in comments (for example `# ---HEADER---` or `// == PIPELINE PT A ==`).
- Use print statements sparingly; only use separators when splitting very large log sections.
- Never leave wording that makes AI authorship obvious.

# Notes space for agents

**Boundary:** Do not move, copy, merge, or relocate any content from **above** this `# Notes space for agents` heading into this section. Keep structural instructions in their existing sections upstream; use this block only for **new** session-specific notes you add (scratch reminders, one-off context).

## ts-sandbox (repo context)

ML / time-series experiments (diffusion and related). Slurm entrypoints live at repo root (`slurm_*.sh`); training code under `models/`. Alliance-specific paths: `.ai/cluster-paths.md`; use the **`/alliancecan`** skill for cluster work.

After tasks: semantic commit and push via **`/git`** — see **Wrap-up** above. Do not commit scratch outputs, huge logs, checkpoints, or throwaway scripts — see **Git hygiene** above; extend `.gitignore` in the same change when you add that kind of artifact.