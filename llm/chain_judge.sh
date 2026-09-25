#!/bin/bash
# Wait for the six generations of 2026-09-20 to land, then judge the three pairs.
#
# The 4B LoRA arms were resumed to the full 1910 steps (arrays 5552333/5552514), so seed 0 supersedes
# the step-1000 policies the published 58.5% came from; its old answers are kept beside the new ones
# as *_step1000.jsonl. Gemma seed 0 is judged at the 2048 cap the cross-family block uses.
#
# Judging runs here on the login node: it needs the internet and OPENAI_API_KEY, which no compute node
# has. Each pair reports AMARI's win rate, so canonical is 100 minus what pairwise_h2h.py prints.
#
#   nohup bash chain_judge.sh > logs/chain_judge_20260920.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
A=results/bench/arena_v01
PAIRS=(
  "lora4b_kl_amari_bdpo_s0|lora4b_kl_canon_bdpo_s0|h2h_lora4b/h2h_lora4b_s0_1910"
  "lora4b_kl_amari_bdpo_s2|lora4b_kl_canon_bdpo_s2|h2h_lora4b/h2h_lora4b_s2_1910"
  "loragemma1b_kl_amari_bdpo_s0_mn2048|loragemma1b_kl_canon_bdpo_s0_mn2048|h2h_lorafam/h2h_loragemma1b_s0"
)

echo "=== waiting for 6 generations | $(date) ==="
while :; do
  n=0
  for p in "${PAIRS[@]}"; do
    for t in "${p%%|*}" "$(echo "$p" | cut -d'|' -f2)"; do
      [ -f "$A/$t.jsonl" ] && [ "$(wc -l < "$A/$t.jsonl")" -ge 500 ] && n=$((n+1))
    done
  done
  echo "  ready: $n/6 | $(date)"
  [ "$n" -ge 6 ] && break
  sleep 300
done

for p in "${PAIRS[@]}"; do
  a="${p%%|*}"; b="$(echo "$p" | cut -d'|' -f2)"; out="$(echo "$p" | cut -d'|' -f3)"
  mkdir -p "$A/$(dirname "$out")"
  echo "=== judging $a vs $b | $(date) ==="
  python pairwise_h2h.py --a "$a" --b "$b" --out "$A/${out}.json"
done
echo "=== ALL DONE | $(date) ==="
