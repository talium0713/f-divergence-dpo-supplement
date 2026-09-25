#!/bin/bash
# Waits for the two Gemma-1B LoRA seed-0 generations (job 5566503_[6,7], the cross-family block at the
# 2048 cap) and judges the pair. Same judge as the other 44 head-to-heads.
set -uo pipefail
cd "$(dirname "$0")"
source "$HOME/arena_judge_env/bin/activate"
export H2H_JUDGE=gpt-4.1-mini-2025-04-14
export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
A=results/bench/arena_v01
a=loragemma1b_kl_amari_bdpo_s0_mn2048; b=loragemma1b_kl_canon_bdpo_s0_mn2048
while :; do
  n=0
  for t in $a $b; do [ -f $A/$t.jsonl ] && [ $(wc -l < $A/$t.jsonl) -ge 500 ] && n=$((n+1)); done
  echo "  ready $n/2 | $(date)"; [ $n -ge 2 ] && break; sleep 300
done
mkdir -p $A/h2h_lorafam
python pairwise_h2h.py --a $a --b $b --out $A/h2h_lorafam/h2h_loragemma1b_s0.json
echo "=== GEMMA JUDGED | $(date) ==="
