#!/bin/bash
# Three head-to-heads launched in parallel (judging is API-bound on the login node, no GPU):
#   1.7B full-FT seed 2 FKL and Hellinger (group C1, trained on the rented box), and 8B LoRA seed 2 FKL.
# Same judge as the other 39 head-to-heads. Each process keeps --parallel 8, so 24 calls in flight.
set -uo pipefail
cd "$(dirname "$0")"
source "$HOME/arena_judge_env/bin/activate"
export H2H_JUDGE=gpt-4.1-mini-2025-04-14
export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
A=results/bench/arena_v01
python pairwise_h2h.py --a rkl_amari_bdpo_s2 --b rkl_canon_bdpo_s2 \
  --out $A/h2h_seeds17b/h2h_rkl_amari_vs_canon_s2_mini2.json > logs/judge_s2_17b_fkl.log 2>&1 &
python pairwise_h2h.py --a hel_amari_bdpo_s2 --b hel_canon_bdpo_s2 \
  --out $A/h2h_seeds17b/h2h_hel_amari_vs_canon_s2_mini2.json > logs/judge_s2_17b_hel.log 2>&1 &
python pairwise_h2h.py --a lora8b_rkl_amari_bdpo_s2 --b lora8b_rkl_canon_bdpo_s2 \
  --out $A/h2h_lora8b/h2h_lora8b_rkl_amari_vs_canon_s2.json > logs/judge_s2_8b_fkl.log 2>&1 &
wait
echo "=== ALL THREE DONE | $(date) ==="
