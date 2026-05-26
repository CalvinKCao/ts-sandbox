---
name: lnai-master-sync
description: Terse flow to update master .ai, copy to repo, and sync with lnai.
---

# Master .ai Sync (Terse)

Permission grant for this workflow:
- The agent is explicitly allowed to edit `/home/cao/.ai` for this task.

Do this sequence:
1. Edit master config in `/home/cao/.ai`.
2. Copy master `.ai` into current repo root.
3. Run `lnai validate --tools gemini cursor copilot`.
4. Run `lnai sync --tools gemini cursor copilot`.

Constraints:
- Keep it token-light and direct.
- Only target `gemini`, `cursor`, and `copilot` unless user overrides.
