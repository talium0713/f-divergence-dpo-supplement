#!/bin/bash
# AlpacaEval 2.0 LC, canonical vs Amari, 4B LoRA RKL. Mirrors a collaborator's
# jobs/div_sweep_1.7b/alpaca_lc/run_h2h_canon_vs_amari.sh: the Amari arm of the SAME seed is the
# reference, not the gpt-4-turbo leaderboard baseline, which floors a small model and cannot resolve
# the two arms. Annotator is his weighted_alpaca_eval_gpt41mini (stock prompt + logprob weighting,
# model_name gpt-4.1-mini-2025-04-14, since gpt-4-1106-preview is retired).
# Output dirs follow his make_ae_table.py pattern: ae_h2h_<size>_<div>_s<seed>.
set -uo pipefail

module load StdEnv/2023 python/3.11 gcc arrow/21.0.0 scipy-stack
source ~/alpaca_judge_env/bin/activate
cd "$(dirname "$0")"
export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
export OPENBLAS_NUM_THREADS=4
D=jobs/alpaca_lc
for k in "$@"; do
  echo "=== $(date +%H:%M:%S) 4B RKL seed $k — canon vs amari ==="
  alpaca_eval --model_outputs "$D/alpaca_lora4b_kl_canon_s${k}.json" \
    --reference_outputs "$D/alpaca_lora4b_kl_amari_s${k}.json" \
    --annotators_config weighted_alpaca_eval_gpt41mini \
    --output_path "results/bench/ae_h2h_4b_kl_s${k}" 2>&1 | tail -8
done
echo "=== DONE $(date) ==="
