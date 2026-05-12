#!/usr/bin/env bash
# Recursively find iTransformer checkpoint / HP cache files under the repo.
#
# Usage (from repo root):
#   ./utils/clean_itrans_checkpoints.sh           # dry-run: print paths only
#   ./utils/clean_itrans_checkpoints.sh --delete  # actually remove (trash-put if available)
#
# Skips directory trees: .git, .venv, venv (by name, anywhere under ROOT)

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

DELETE=0
if [[ "${1:-}" == "--delete" ]] || [[ "${1:-}" == "-f" ]]; then
  DELETE=1
fi

matches=()
while IFS= read -r -d '' f; do
  matches+=("$f")
done < <(
  find "$ROOT" \
    \( -name '.git' -o -name '.venv' -o -name 'venv' \) -prune -o \
    \( \
      -name 'itransformer.pt' -o \
      -name 'pretrained_itransformer.pt' -o \
      -name 'itrans_hp_best.pt' -o \
      -name 'itrans_hp.json' -o \
      -name '*_itrans_ft_hp_best.pt' -o \
      -name '*_itrans_ft_hp.json' -o \
      -name '*_itransformer_finetuned.pt' \
    \) -type f -print0 2>/dev/null
)

n=${#matches[@]}
if [[ "$n" -eq 0 ]]; then
  echo "No matching iTrans checkpoint/cache files under $ROOT"
  exit 0
fi

if [[ "$DELETE" -eq 0 ]]; then
  echo "Dry run ($n file(s)). Pass --delete to remove:"
  printf '  %s\n' "${matches[@]}"
  exit 0
fi

for f in "${matches[@]}"; do
  if command -v trash-put >/dev/null 2>&1; then
    trash-put -- "$f" && echo "trashed: $f" || echo "FAILED: $f" >&2
  else
    rm -f -- "$f" && echo "deleted: $f" || echo "FAILED: $f" >&2
  fi
done

echo "Done. Removed $n file(s)."
