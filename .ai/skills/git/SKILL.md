---
name: git
description: >-
  Cursor: invoke as /git. Drafts semantic Git commit messages from context and
  diffs, splits work into multiple commits when appropriate, then commits,
  pushes, pulls, and resolves merge conflicts. Use when the user asks to
  commit, push, write a commit message, or wrap up git work after changes.
---

# GitHub / Git — semantic commits

## Before touching Git

- If the work looks like a **throwaway or one-off experiment**, pause and ask whether to use a **new branch** before committing.
- If changes appear to be on the **wrong branch** (e.g. feature work on `main`), ask whether to **move the changes** to the correct branch (e.g. stash/checkout/cherry-pick or guided reset) before proceeding.

## Gather context

Use **what was done in the session** and **`git diff`** (prefer **staged** diff if something is staged; otherwise working tree vs `HEAD`). Name files/modules only when it helps; stay terse.

## Pre-commit safety check

Before committing, inspect staged/untracked paths for files that are likely meant to be ignored (e.g. large binaries, `.png` dumps, `.log`, training checkpoints, cache folders, library/vendor directories).

- If suspicious files are present, **ask for explicit confirmation** before committing.
- Offer to update **`.gitignore`** first, then re-stage only intended files.

## Commit message shape

**One line (subject):** `<type>(<scope>): <subject>` — scope optional. **Present tense**, **imperative mood**, **~50 chars or less** for the subject when practical.

**Body (details):** After a blank line, add a **short** bullet-style or paragraph summary: what changed and why, edge cases, follow-ups. Keep it **concise**; no essays.

### Semantic Commit Messages

Format: `<type>(<scope>): <subject>`

`<scope>` is optional  
Example

    feat: add hat wobble
    ^--^  ^------------^
    |     |
    |     +-> Summary in present tense.
    |
    +-------> Type: chore, docs, feat, fix, refactor, style, or test.

More Examples:

    feat: (new feature for the user, not a new feature for build script)
    fix: (bug fix for the user, not a fix to a build script)
    docs: (changes to the documentation)
    style: (formatting, missing semi colons, etc; no production code change)
    refactor: (refactoring production code, eg. renaming a variable)
    test: (adding missing tests, refactoring tests; no production code change)
    chore: (updating grunt tasks etc; no production code change)

Pick **one primary type** per commit. If a change mixes unrelated concerns, **split into multiple commits** so each commit has one clear intent.

## Split commits

**Split into multiple commits** when it improves history: e.g. refactors separate from features, docs separate from code, formatting separate from logic. Stage/commit in logical chunks (`git add -p` when useful).

## After committing

1. **`git push`** to the tracked remote branch.
2. **`git pull`** (rebase or merge per project convention; default to **merge** if unknown) and **resolve conflicts** in the working tree, then continue the rebase or complete the merge and push again if needed.

## Example

```
fix(parser): handle empty wiki sections

- Skip null section bodies instead of raising
- Add regression test for blank heading
```
