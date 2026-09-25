#!/bin/bash
# Judge the five 4B pairs that were generated at the 2048 cap for the truncation diagnostic but
# never judged head-to-head. No GPU needed — this runs on the login node against the OpenAI API.
#
# The 1024-cap judgments already exist (h2h_scale4b/*); these are the same policies and the same
# 500 prompts, only the generation cap differs, so the pair answers "does the canonical-vs-Amari
# result survive doubling the cap" at 4B. The 8B counterpart runs separately.
set -uo pipefail
cd "$(dirname "$0")"
OUT=results/bench/arena_v01/h2h_scale4b
mkdir -p "$OUT"
for d in adiv chi2 hel js kl; do
  a="scale4b_${d}_amari_bdpo_mn2048"
  b="scale4b_${d}_canon_bdpo_mn2048"
  o="$OUT/h2h_scale4b_${d}_amari_vs_canon_mn2048.json"
  [ -f "$o" ] && { echo "=== skip $d (already judged) ==="; continue; }
  echo "=== judging $d | $(date) ==="
  python pairwise_h2h.py --a "$a" --b "$b" --out "$o"
done
echo "=== ALL DONE | $(date) ==="
