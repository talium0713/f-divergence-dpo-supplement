# Code supplement — canonical vs Amari generators for f-divergence preference optimization

Anonymized source for the paper's two experimental halves: the tabular study on a layered MDP
(`python/`) and the LLM study on Arena-Hard and AlpacaEval (`llm/`).

**Anonymization.** Cluster paths, the allocation account and the repository URL are replaced by
placeholders (`$PROJECT`, `$SCRATCH`, `ACCOUNT`, `cluster`, `ANONYMIZED`), the git history is not
included, and the figure sidecars no longer record the commit that produced them. Nothing else is
altered: every script is the one that produced the numbers in the paper.

---

## Layout

| Path | What it holds |
|---|---|
| `python/` | The tabular experiments and every tabular figure. Cached run outputs in `python/data/tabular/` (23 runs) mean the figures redraw without retraining. |
| `llm/` | f-DPO training (`stage_b_train.py`), generation (`gen_bench.py`, `gen_alpaca.py`), judging (`pairwise_h2h.py`), and 76 SLURM wrappers under `jobs/`. |
| `figure4overleaf/` | The exported figures as they appear in the paper, each with a JSON sidecar naming the source script and the source data. |
| `cluster/` | Environment setup for the machine the LLM runs used (module list, pinned requirements). |

---

## Reproducing the tabular figures

```bash
pip install numpy matplotlib          # that is the whole dependency list for these
cd python
python fig_tabular.py                 # Figure 1
python fig_paper_layered_v2.py        # Figures 2, 10, 11, 12 (peaks 0.7 / 0.6 / 0.8 / 0.9) and 13
python fig_figure3.py                 # Figure 3
python fig_figure4.py                 # Figure 4
python fig_paper_layered.py           # Figure 6
python fig_mechanism.py               # Figures 7, 8
python verify.py                      # 7 numerical checks; all must print PASS
```

Figures land in `python/figs/`. `python ../export_for_paper.py` copies them to
`figure4overleaf/` under their paper names; open a sidecar there to see which script and which run
directory produced a given figure.

Figures 6-9 and 14 are typeset with real LaTeX (`text.usetex`), so they need a TeX installation with
`amsmath` and `amssymb`. The others use matplotlib's own mathtext and need nothing extra.

To regenerate the cached runs rather than redraw from them, `python/run_canon_final.py`,
`run_tabular.py`, `run_adiv.py` and `run_t06_alpha_nmc.py` are the entry points. A full 100-MDP
sweep is a few CPU-hours; the shipped caches let a reviewer skip that.

**Configuration.** Every tabular figure asserts the environment it was drawn from against the run
manifest before drawing (`fig_paper_layered.check_env`), so a figure cannot silently be drawn from a
different MDP: depth 4, 3 states per layer, 3 actions, ε = 0.2, γ = 0.9, rewards Uniform(−0.8, 0.8),
uniform π_ref, Bradley–Terry preferences, and a regularization weight calibrated per (divergence,
target peak) by bisection rather than a fixed β.

---

## Reproducing the LLM experiments

```bash
pip install torch transformers peft accelerate datasets safetensors numpy
```

Three stages, each a plain Python invocation; the `.slrm` files under `llm/jobs/` are SLURM wrappers
around exactly these commands.

**1. Train.** Both arms of a pair differ only in `--norm`:

```bash
python stage_b_train.py \
  --ref Qwen/Qwen3-1.7B-Base --policy Qwen/Qwen3-1.7B-Base --init-noise 0.01 \
  --data data/uf_pairs_train_full.jsonl --eval-data data/uf_pairs_test.jsonl \
  --div kl --norm canon --inner sample --step-mode token \
  --beta 0.1 --lr 5e-6 --lr-schedule linear --warmup-ratio 0.05 \
  --epochs 2 --grad-accum 64 --max-len 1024 --clamp 15 --seed 0 --save-policy \
  --out results/bench/stageB_kl_canon_s0
```

`--norm amari` gives `f'(1)=0`, `--norm canon` gives `f'(1)=f''(1)`. Divergence keys: `kl` is
**reverse** KL and `rkl` is **forward** KL; the others are `adiv`, `js`, `hel`, `chi2`. Adding
`--lora-r 128 --lora-alpha 128 --lora-dropout 0.05` switches a run from full fine-tuning to LoRA.
`--init-noise` is required for the Amari arm: at u = 1 an Amari generator has zero gradient, so a
policy initialized at the reference would never move.

`python stage_b_train.py --selftest` checks the score math on synthetic logits — no GPU, no
downloads — and verifies that RKL's canonical arm reproduces standard DPO exactly.

**2. Generate.** 500 Arena-Hard v0.1 prompts, greedy:

```bash
python gen_bench.py --model <policy dir> --generator <tag> --prompts data/arena_hard_v01.jsonl \
  --format arenav2 --think default --system "You are a helpful assistant." \
  --rep-penalty 1.05 --max-new 1024 --temperature 0.0 --out results/bench/arena_v01/<tag>.jsonl
```

For AlpacaEval, `gen_alpaca.py` with `--prompts data/alpaca_eval.jsonl --max-new 512` (805 prompts).

**3. Judge** (needs an OpenAI key in `openai_key.txt` or `OPENAI_API_KEY`):

```bash
python pairwise_h2h.py --a <amari tag> --b <canonical tag> --out <result>.json
```

Two position-swapped games per prompt, judge `gpt-4.1-mini-2025-04-14`, prompt-level bootstrap CIs.
The reported number is the **canonical** win rate, i.e. 100 minus what this script prints.
AlpacaEval is judged with the `alpaca_eval` CLI, passing the Amari arm as `--reference_outputs` so
the comparison is canonical-vs-Amari rather than against a fixed leaderboard baseline.

**Cost.** One run is 1,910 optimizer steps (2 epochs, effective batch 64). Measured on one 48 GB
L40S: 1.7B LoRA ≈ 19 h (20 GB), 4B LoRA ≈ 40 h (18 GB), 8B LoRA ≈ 44 h (28 GB), 1.7B full
fine-tuning ≈ 21 h (26 GB). Generation is ≈ 0.6 h per policy at 4B.

**Data.** `data/*.jsonl` is not shipped (UltraFeedback pairs are ~240 MB). `make_pairs_jsonl.py`
rebuilds the training and evaluation splits from the public HuggingFace dataset, and
`make_alpaca_jsonl.py` the AlpacaEval prompts; `data/arena_hard_v01.jsonl` is the public Arena-Hard
v0.1 set.

---

## What is not included

Model checkpoints and generated answers (hundreds of GB), the training data files (regenerable with
the scripts above), the git history, and the paper PDF. `llm/results/` keeps the judged head-to-head
summaries and the Stage-A measurement arrays, which are what the figures read.

Also left out: the scripts that moved files between our cluster and a rented GPU box, and an
interactive browser demo of the theory. Neither produced a number in the paper. Every script that
did is here.
