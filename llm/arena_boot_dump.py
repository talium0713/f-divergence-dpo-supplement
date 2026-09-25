"""arena_boot_dump.py — recover the bootstrap replicates behind fL3's top panel.

fL3's top panel (Arena-Hard v0.1 win rate against the fixed gpt-4-0314 baseline) was transcribed by
hand from show_result.py's console output, so the figure only ever had a point estimate and two CI
half-widths — enough for an error bar, not for a box. show_result.py does build the replicates
(`bootstraps`, 100 rounds) and then discards everything but the mean and two quantiles. This script
rebuilds them from the judgment files already on disk, so no re-judging and no API cost.

It writes BOTH resampling schemes, because the two panels of fL3 do not currently agree on one:

  game   — what arena-hard-auto does: resample the exploded game rows within each model. Two games of
           the same prompt (the position swap) can be drawn independently, which treats them as two
           independent observations when they are not.
  prompt — what llm/pairwise_h2h.py does for the bottom panel: resample prompts, carrying both of a
           prompt's games together. Correlated games stay together, so the interval is wider and the
           two panels of fL3 become comparable.

1000 replicates either way, matching pairwise_h2h.py, against arena's 100.

Run from llm/ with the arena judge env active:
    python arena_boot_dump.py
Out: results/bench/arena_v01/topboot_normcmp.json
"""
import json, os, sys
import numpy as np

ARENA = os.path.expanduser("~/arena-hard-auto")
sys.path.insert(0, ARENA)
os.chdir(ARENA)                       # load_judgments resolves data/ relative to cwd

from show_result import load_judgments   # noqa: E402

JUDGE = "gpt-4.1-mini-2025-04-14"
BENCH = "arena-hard-v0.1"
CATEGORY = "arena-hard-v0.1"
NBOOT = 1000
MODELS = {                              # figure label -> model tag in the arena repo
    "RKL":   ("kl_amari_bdpo",   "kl_bdpo"),
    "α-div": ("adiv_bdpo",       "adiv_canon_bdpo"),
    "FKL":   ("rkl_amari_bdpo",  "rkl_canon_bdpo"),
    "JS":    ("js_bdpo",         "js_canon_bdpo"),
    "Hel":   ("hel_bdpo",        "hel_canon_bdpo"),
    "χ²":    ("chi2_bdpo",       "chi2_canon_bdpo"),
}

battles = load_judgments([JUDGE], BENCH)
battles = battles[battles.category == CATEGORY] if "category" in battles else battles
print(f"[load] {len(battles)} game rows · {battles.model.nunique()} models", flush=True)

rng = np.random.default_rng(12345)
out = {"judge": JUDGE, "benchmark": BENCH, "n_boot": NBOOT, "arms": {}}

for label, arms in MODELS.items():
    for arm, tag in zip(("amari", "canon"), arms):
        sub = battles[battles.model.map(lambda m: m.split("/")[-1]) == tag]
        if sub.empty:
            print(f"  [miss] {label} {arm}: no rows for {tag}"); continue
        scores = sub.scores.astype(float).to_numpy()
        uids = sub.uid.to_numpy()

        game = rng.choice(scores, size=(NBOOT, len(scores)), replace=True).mean(axis=1)

        uniq, inv = np.unique(uids, return_inverse=True)          # prompt-level: draw prompts, keep
        by_prompt = [scores[inv == i] for i in range(len(uniq))]  # each prompt's games together
        draws = rng.integers(0, len(uniq), size=(NBOOT, len(uniq)))
        prompt = np.array([np.concatenate([by_prompt[j] for j in row]).mean() for row in draws])

        out["arms"][f"{label}|{arm}"] = {
            "model": tag, "n_games": int(len(scores)), "n_prompts": int(len(uniq)),
            "point": float(scores.mean() * 100),
            "boot_game": [round(float(v) * 100, 4) for v in np.sort(game)],
            "boot_prompt": [round(float(v) * 100, 4) for v in np.sort(prompt)],
        }
        g, p = np.sort(game) * 100, np.sort(prompt) * 100
        print(f"  {label:6s} {arm:5s} {tag:20s} point {scores.mean()*100:5.1f}  "
              f"game[2.5,97.5] {g[24]:5.1f}-{g[975]:5.1f}  prompt {p[24]:5.1f}-{p[975]:5.1f}", flush=True)

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".",
                   "results/bench/arena_v01/topboot_normcmp.json")
dst = "$PROJECT/bregman-lab/llm/results/bench/arena_v01/topboot_normcmp.json"
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w"))
print(f"[saved] {dst}  ({len(out['arms'])} arms)")
