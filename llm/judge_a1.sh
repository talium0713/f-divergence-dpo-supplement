#!/bin/bash
# A1: the 4B LoRA pairs regenerated on vast.ai at the full 1910 steps (seeds 0 and 2).
# Judge is pinned to gpt-4.1-mini, the model 39 of the 43 existing judgments used — including the
# step-1000 4B seed-0 number (Amari 41.5%) this one supersedes, so the two are comparable.
set -uo pipefail
source "$HOME/arena_judge_env/bin/activate"   # the only env with the openai client
cd "$(dirname "$0")"
export H2H_JUDGE=gpt-4.1-mini-2025-04-14
export OPENAI_API_KEY="$(tr -d "[:space:]" < openai_key.txt)"
A=results/bench/arena_v01; mkdir -p $A/h2h_lora4b
for S in 0 2; do
  echo "=== 4B LoRA seed $S @1910 | $(date) ==="
  python pairwise_h2h.py --a lora4b_kl_amari_bdpo_s$S --b lora4b_kl_canon_bdpo_s$S \
    --out $A/h2h_lora4b/h2h_lora4b_kl_amari_vs_canon_s${S}_1910.json
done
echo "=== A1 JUDGING DONE | $(date) ==="
