#!/bin/bash
# Judge the four complete pairs a collaborator generated (branch lora-8b-seeds), pulled into
# results/bench/arena_v01/. LoRA only.
#
# Comparable with our own runs: the training recipe is identical (r=alpha=128, dropout 0.05,
# beta 0.1, lr 5e-6 linear + 0.05 warmup, accum 64, max-len 1024, clamp 15, init-noise 0.01,
# same UltraFeedback pairs) and the generation flags match gen_lora4b.slrm exactly, cap included
# (arena_hard_v01, greedy, rep-penalty 1.05, max-new 1024, same system prompt).
#
# 1.7B seed 1 is NOT here: its amari arm was still generating. 8B divs seeds 0/1 likewise.
set -uo pipefail
cd "$(dirname "$0")"
PAIRS=(
  "lora8b_kl_amari_bdpo_s1|lora8b_kl_canon_bdpo_s1|h2h_lora8b/h2h_lora8b_kl_amari_vs_canon_s1"
  "lora8b_kl_amari_bdpo_s2|lora8b_kl_canon_bdpo_s2|h2h_lora8b/h2h_lora8b_kl_amari_vs_canon_s2"
  "lora1p7b_kl_amari_bdpo_s0|lora1p7b_kl_canon_bdpo_s0|h2h_lora1p7b/h2h_lora1p7b_kl_amari_vs_canon_s0"
  "lora1p7b_kl_amari_bdpo_s2|lora1p7b_kl_canon_bdpo_s2|h2h_lora1p7b/h2h_lora1p7b_kl_amari_vs_canon_s2"
)
A=results/bench/arena_v01
for p in "${PAIRS[@]}"; do
  a="${p%%|*}"; b="$(echo "$p" | cut -d'|' -f2)"; o="$(echo "$p" | cut -d'|' -f3)"
  mkdir -p "$A/$(dirname "$o")"
  [ -f "$A/${o}.json" ] && { echo "=== skip $a (judged) ==="; continue; }
  echo "=== judging $a vs $b | $(date) ==="
  python pairwise_h2h.py --a "$a" --b "$b" --out "$A/${o}.json"
done
echo "=== ALL DONE | $(date) ==="
