#!/bin/bash
# AlpacaEval 2.0 LC and WR, canonical versus the Amari arm of the SAME seed, for any LoRA scale.
#   bash run_ae_h2h.sh <size> <seed> [seed ...]        e.g.  run_ae_h2h.sh 8b 0 1 2
#
# Generalises run_ae_h2h_4b.sh, which hard-coded lora4b. Everything else is unchanged so the numbers
# pool with the ones already in the table: the reference is the Amari policy of that seed rather than
# the gpt-4-turbo leaderboard baseline (which floors a small model and cannot resolve the two arms),
# and the annotator is a collaborator's weighted_alpaca_eval_gpt41mini (stock prompt + logprob weighting,
# model gpt-4.1-mini-2025-04-14, since gpt-4-1106-preview is retired).
#
# Output dirs follow his make_ae_table.py pattern: ae_h2h_<size>_kl_s<seed>. A seed whose two 805-row
# generations are not both present is skipped with a message rather than half-judged, and a seed that
# already has a result directory is left alone - judging twice costs API budget and overwrites a
# number someone may already have copied into the paper.
set -uo pipefail
SIZE="${1:?usage: run_ae_h2h.sh <size> <seed> [seed ...]}"; shift

module load StdEnv/2023 python/3.11 gcc arrow/21.0.0 scipy-stack
source ~/alpaca_judge_env/bin/activate
cd "$(dirname "$0")"
export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
export OPENBLAS_NUM_THREADS=4
D=jobs/alpaca_lc

for k in "$@"; do
  C="$D/alpaca_lora${SIZE}_kl_canon_s${k}.json"
  A="$D/alpaca_lora${SIZE}_kl_amari_s${k}.json"
  O="results/bench/ae_h2h_${SIZE}_kl_s${k}"
  if [ ! -f "$C" ] || [ ! -f "$A" ]; then
    echo "[skip] ${SIZE} seed $k - no generations (need $(basename "$C") and $(basename "$A"))"
    continue
  fi
  if [ -d "$O" ]; then echo "[skip] ${SIZE} seed $k - $O already judged"; continue; fi
  echo "=== $(date +%H:%M:%S) ${SIZE} RKL seed $k - canon vs amari ==="
  alpaca_eval --model_outputs "$C" --reference_outputs "$A" \
    --annotators_config weighted_alpaca_eval_gpt41mini \
    --output_path "$O" 2>&1 | tail -8
done
echo "=== DONE $(date) ==="
