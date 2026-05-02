---
name: lnai
description: Manage unified .ai config with lnai; default sync scope is gemini, cursor, copilot.
---

# LNAI Workflow

Use `lnai` as the source-of-truth workflow for AI tooling config.

## Defaults
- Default tool scope: `gemini cursor copilot`
- Default sync command: `lnai sync --tools gemini cursor copilot`
- Validate before sync when config changed: `lnai validate --tools gemini cursor copilot`

## Core commands
- Validate: `lnai validate --tools gemini cursor copilot`
- Preview sync: `lnai sync --dry-run --tools gemini cursor copilot`
- Apply sync: `lnai sync --tools gemini cursor copilot`

## Rules
- Keep `.ai/` committed; treat it as source of truth.
- Prefer `--dry-run` before writing files when changing rules/settings.
- Use `--verbose` when diagnosing unexpected generated output.
- Do not sync unrelated tools unless user asks.

## Notes
- Gemini output includes `.gemini/settings.json`, `.gemini/skills/*`, and generated `GEMINI.md` files.
- Cursor and Copilot outputs are generated from the same `.ai/` source.
