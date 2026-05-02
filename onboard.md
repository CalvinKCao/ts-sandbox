# ts-sandbox — agent onboard

ML / time-series experiments (diffusion and related). Slurm entrypoints live at repo root (`slurm_*.sh`); training code under `models/`. Alliance-specific paths: `.ai/cluster-paths.md` and `/alliancecan`.

## After you finish a task

Use the **`/git`** skill (`.ai/skills/git/SKILL.md`): semantic commit message, split commits if needed, **commit and push** to the current branch’s upstream.

## Artifacts

Do not commit scratch outputs, huge logs, checkpoints, or throwaway scripts. Extend `.gitignore` in the same change if you create that kind of file.
