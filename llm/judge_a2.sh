#!/bin/bash
# A2 (2026-09-23): the three pairs completed on the rented 4x L40S box.
#   1.7B full-FT seed 2 — FKL (code key rkl) and Hellinger
#   8B LoRA   seed 2 — FKL
# Judge pinned to gpt-4.1-mini, the model 39 of the 43 existing judgments used, so these pool with them.
set -uo pipefail
source "$HOME/arena_judge_env/bin/activate"   # the only env with the openai client
cd "$(dirname "$0")"
export H2H_JUDGE=gpt-4.1-mini-2025-04-14
export OPENAI_API_KEY="$(tr -d "[:space:]" < openai_key.txt)"
A=results/bench/arena_v01; mkdir -p $A/h2h_seeds17b $A/h2h_lora8b

echo "=== 1.7B full-FT FKL seed 2 | $(date) ==="
python pairwise_h2h.py --a rkl_amari_bdpo_s2 --b rkl_canon_bdpo_s2 \
  --out $A/h2h_seeds17b/h2h_rkl_amari_vs_canon_s2_mini2.json

echo "=== 1.7B full-FT Hellinger seed 2 | $(date) ==="
python pairwise_h2h.py --a hel_amari_bdpo_s2 --b hel_canon_bdpo_s2 \
  --out $A/h2h_seeds17b/h2h_hel_amari_vs_canon_s2_mini2.json

echo "=== 8B LoRA FKL seed 2 | $(date) ==="
python pairwise_h2h.py --a lora8b_rkl_amari_bdpo_s2 --b lora8b_rkl_canon_bdpo_s2 \
  --out $A/h2h_lora8b/h2h_lora8b_rkl_amari_vs_canon_s2_mini2.json

echo "=== A2 JUDGING DONE | $(date) ==="
