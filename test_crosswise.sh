#!/usr/bin/env bash
# test_crosswise.sh — run client_console.py over all (source, reference) pairs
# Usage:
#   ./test_crosswise.sh [SOURCE_DIR] [REFERENCE_DIR] [-- extra click options...]
# Example:
#   ./test_crosswise.sh examples/Source_Demo examples/Reference_Demo \
#       -- --diffusion-steps 50 --length-adjust 1.0 --inference-cfg-rate 0.7 --auto-f0-adjust

set -euo pipefail

SRC_DIR="${1:-examples/Source_Demo}"
REF_DIR="${2:-examples/Reference_Demo}"

# Everything after a literal "--" is passed through to the Python CLI as-is.
EXTRA_ARGS=()
if [[ "${3-}" == "--" ]]; then
  shift 2
  EXTRA_ARGS=("$@")
else
  # If no "--" provided but more args exist, still forward them safely.
  shift $(($# > 2 ? 2 : $#))
  EXTRA_ARGS=("$@")
fi

PY="${PYTHON:-python}"
CLIENT="${CLIENT:-client_console.py}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: Source dir not found: \"$SRC_DIR\"" >&2
  exit 1
fi
if [[ ! -d "$REF_DIR" ]]; then
  echo "ERROR: Reference dir not found: \"$REF_DIR\"" >&2
  exit 1
fi

# Collect audio files (quoted to handle spaces). Extend the pattern list if needed.
mapfile -d '' SOURCES < <(find "$SRC_DIR" -maxdepth 1 -type f \
  \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' \) -print0)

mapfile -d '' REFS < <(find "$REF_DIR" -maxdepth 1 -type f \
  \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' \) -print0)

if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "ERROR: No audio files in \"$SRC_DIR\"" >&2
  exit 1
fi
if [[ ${#REFS[@]} -eq 0 ]]; then
  echo "ERROR: No audio files in \"$REF_DIR\"" >&2
  exit 1
fi

mkdir -p runs

for s in "${SOURCES[@]}"; do
  for r in "${REFS[@]}"; do
    printf '\n=== %s  ×  %s ===\n' "$(basename "$s")" "$(basename "$r")"
    ts="$(date +%Y%m%d-%H%M%S)"
    s_base="$(basename "$s")"
    r_base="$(basename "$r")"
    log="runs/${s_base%.*}__${r_base%.*}__${ts}.log"

    # All inputs are passed as strings and quoted to preserve spaces.
    "$PY" "$CLIENT" \
      --source-uri "$s" \
      --target-uri "$r" \
      "${EXTRA_ARGS[@]}" | tee "$log"
  done
done

echo -e "\nAll pairs finished. Logs saved in ./runs/"
