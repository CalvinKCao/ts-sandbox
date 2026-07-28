---
name: git
description: >-
  rafts custom-scheme Git commit messages from context and
  diffs, splits work into multiple commits when appropriate, then commits,
  pushes, pulls, and resolves merge conflicts. Use when the user asks to
  commit, push, write a commit message, or wrap up git work after changes.
---

# GitHub / Git — custom commit scheme

## Before touching Git
- Your goal is to rigorously defend the below guidelines from new developers on the codebase who aren't familiar with best practises. Always
prefer to interject before commiting changes to ask user whether to branch, cherry pick, etc. based on below guides.
- If the work looks like a **novel experiment** that is not yet verified to improve upon current design, pause and ask user whether to use a **new experimental branch** (e.g. `exp/<short-desc>`) before committing. This should basically ALWAYS be the case unless the user explicity says
they've already tested and found the change successful. You may need to stop and ask the user to branch more often than not.
    - Branch naming convention: use exp/<short-desc> for exploratory or uncertain work you're not yet committed to keeping; feat/<short-desc> for features you intend to merge into main once validated; fix/<short-desc> for standalone, general-purpose fixes pulled out of other branches (e.g. via cherry-pick); graveyard/<short-desc> for abandoned branches you're keeping around for reference rather than deleting; and paused/<short-desc> as a tag (not a branch) for marking a specific commit you want to return to later without keeping its branch alive. This keeps git branch --list and git tag -l readable at a glance and makes it obvious what state each line of work is in.
- If changes appear to be on the **wrong branch** (e.g. feature work on `main`), ask whether to **move the changes** to the correct branch (stash/checkout/cherry-pick or guided reset) before proceeding.
- If a commit is about to be made on an experiment branch and the change is actually **general-purpose / orthogonal to the experiment** (e.g. a logging fix, dataloader bug fix, util function), ask whether to **cherry-pick it onto its own branch off `main`** (e.g. `fix/<short-desc>`) instead of, or in addition to, committing it on the exp branch. Don't do this for changes whose value depends on the experiment's outcome — those wait until validated.
- If you summarize the results of an experiment and they seem successful, offer to merge the changes into main. Don't let the newbie developer
work forever on a side branch without commiting successes to main.
- If the commit has already been tied to an experiment run (i.e. its hash will be logged externally — W&B, MLflow, a results file, etc.), treat it as **immutable once made**: don't offer to amend/reword/rebase it afterward. If commit history is messy (e.g. tons of bugfixes in individual commits) ask the user if they've tied any of the intermediate commits to important runs, then suggest tagging it instead (e.g. `git tag exp/<short-desc>-<date>`) if the user wants a durable reference beyond the branch tip.

## Split commits

**Split into multiple commits** when it improves history: e.g. architecture changes separate from engineering changes, fixes separate from chores. Stage/commit in logical chunks (`git add -p` when useful).

## Gather context

Use **what was done in the session** and **`git diff`** (prefer **staged** diff if something is staged; otherwise working tree vs `HEAD`). Name files/modules only when it helps; stay terse.

## Pre-commit safety check

Before committing, inspect staged/untracked paths for files that are likely meant to be ignored (e.g. large binaries, `.png` dumps, `.log`, training checkpoints, cache folders, library/vendor directories).

- If suspicious files are present, **ask for explicit confirmation** before committing.
- Offer to update **`.gitignore`** first, then re-stage only intended files.

## Commit message shape

**One line (subject):** `<type>: <subject>` — **present tense**, **imperative mood**, **~40 chars or less**.

**Body (details):** After a blank line, bullet points:
- One or more bullets describing the change in detail, including *reasoning* (why, not just what)
- A bullet briefly listing the modules/classes/major files touched and what was done to them
- Additional bullets for edge cases, follow-ups, etc. as needed

Keep it **concise**; no essays.

### Custom commit types

Format: `<type>: <subject>`

    A: {< 40 char main commit msg}

    * Bullet point describing change in more detail, including reasoning
    * Briefly list the modules/classes/major files touched and what you did to them
    * ...

Types:

    A = ML architecture change (model/layer/objective changes)
    E = Engineering change (data pipelines, hyperparam config/tuning, infra, tooling)
    F = Fix (logical/correctness bug — NOT swapping in a better hyperparam/config)
    C = Chore (docs, deleting old files, reorganizing directories, no logic change, or other misc. commits)

Pick **one primary type** per commit. If a change mixes unrelated concerns, **split into multiple commits** so each commit has one clear intent.

## After committing

1. **`git push`** to the tracked remote branch.
2. **`git pull`** (rebase or merge per project convention; default to **merge** if unknown) and **resolve conflicts** in the working tree, then continue the rebase or complete the merge and push again if needed.
3. Always give the user the correct commands to checkout the exact right branch, pull new changes, and submit.

## Example

```
F: empty wiki section crash

* Skip null section bodies instead of raising, since upstream API
  can return sections with no body on drafts
* parser.py: added null check in `parse_section()`
* tests/test_parser.py: added regression test for blank heading
```