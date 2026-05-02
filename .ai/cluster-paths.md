# Alliance / Killarney — path and Slurm rules (for AI + humans)

This file exists so assistants and `.lnai-manifest.json` stay aligned: **one source of truth** for cluster copy-paste, separate from code.

## `$SCRATCH` on Killarney (critical)

On many Alliance systems, **`$SCRATCH` is already your per-user scratch**, e.g. `/scratch/ccao87`. It is **not** `/scratch` with a separate `$USER` segment you must add.

- **Wrong:** `cd $SCRATCH/$USER` → often becomes `/scratch/ccao87/ccao87` → **No such file or directory**
- **Right:** `cd $SCRATCH` then `git clone … drc-sokoban-ma` → repo at **`$SCRATCH/drc-sokoban-ma`**

Match the pattern used in `ts-sandbox` Slurm scripts: prefer **`$SCRATCH/<repo-name>`** for the working tree, with fallback to `$HOME/<repo-name>` only when scratch is unavailable.

## Persistent artifacts (`$PROJECT`)

Checkpoints, venv, wandb, large datasets belong under **`$PROJECT/$USER/<app>/`**, not under `$HOME` and not at the bare group root. Resolve `$PROJECT` with:

```bash
ls ~/projects/
export PROJECT=$(readlink -f ~/projects/aip-YOURGROUP)   # example
```

## Slurm stdout / stderr (where are my logs?)

- Prefer **one combined log** per job (stdout + stderr): early in the batch script, `cd "$SLURM_SUBMIT_DIR"`, `mkdir -p results/logs`, set `LOG=results/logs/$(date +%m-%d)-${SLURM_JOB_ID: -4}-<slug>.log`, then `exec >>"$LOG" 2>&1`, and use `#SBATCH --output=/dev/null --error=/dev/null`, or set identical `-o`/`-e` paths under `results/logs/`.
- Put that log and all run artifacts under **`./results/{logs,ckpts,datasets}/`** relative to **`SLURM_SUBMIT_DIR`** (the directory you were in when you ran `sbatch` if you also pass `--chdir` to that same path). Do not scatter `*.out` / `*.err` in the repo root or under `$SCRATCH/...` unless the user overrides.
- **Find a finished job’s paths:** `sacct -j <JOBID> --format=JobID,JobName,WorkDir,StdOut,State,ExitCode`

## Slurm `--account` (critical)

- The value is the **CCDB Group Name** (Resource Allocation Project), not a free-text label.
- **Never** run `sed` to replace `aip-boyuwang` with the literal string `your-ccdb-group` — Slurm will reject the job (`Invalid account`).
- If the job fails with “Please specify one of the following accounts”, use **exactly** one of the listed names (e.g. `aip-boyuwang`), or pass at submit time:  
  `sbatch --account=aip-boyuwang script.sh`

## Syncing `.ai` skills to tools

Skills live under **`.ai/skills/<name>/`**. The file **`.ai/.lnai-manifest.json`** defines symlinks into `.cursor/`, `.gemini/`, `.windsurf/`, etc. After editing a skill under `.ai/skills/`, re-run your LN sync command if your workflow requires it; the repo already points Cursor at `.ai/skills/*` via the manifest.
