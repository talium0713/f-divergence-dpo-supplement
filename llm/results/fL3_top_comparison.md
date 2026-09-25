# fL3 top panel — three options

Generated from `results/bench/arena_v01/topboot_normcmp.json` (1000 replicates per arm,
recovered from the existing gpt-4.1-mini judgments; no re-judging).

## Intervals per divergence

| div | option | Amari | canonical | overlap |
|---|---|---|---|---|
| RKL | published: game-level 5–95 | 7.3–9.4 | 13.0–15.7 | -3.6 |
| RKL | game-level 2.5–97.5 | 7.1–9.6 | 12.8–16.1 | -3.2 |
| RKL | prompt-level 2.5–97.5 | 6.5–10.4 | 12.0–17.4 | -1.5 |
| α-div | published: game-level 5–95 | 8.6–10.7 | 11.7–14.4 | -1.0 |
| α-div | game-level 2.5–97.5 | 8.4–11.2 | 11.6–14.6 | -0.4 |
| α-div | prompt-level 2.5–97.5 | 7.9–12.1 | 10.9–15.6 | **+1.2** |
| FKL | published: game-level 5–95 | 8.9–11.2 | 12.2–14.6 | -1.0 |
| FKL | game-level 2.5–97.5 | 8.9–11.5 | 12.0–15.1 | -0.5 |
| FKL | prompt-level 2.5–97.5 | 8.1–12.5 | 11.2–16.0 | **+1.3** |
| JS | published: game-level 5–95 | 9.2–11.6 | 12.4–15.1 | -0.8 |
| JS | game-level 2.5–97.5 | 9.0–11.9 | 12.2–15.5 | -0.3 |
| JS | prompt-level 2.5–97.5 | 8.2–12.7 | 11.3–16.6 | **+1.5** |
| Hel | published: game-level 5–95 | 8.6–10.8 | 11.3–13.6 | -0.5 |
| Hel | game-level 2.5–97.5 | 8.6–11.2 | 10.9–14.1 | **+0.3** |
| Hel | prompt-level 2.5–97.5 | 7.8–11.9 | 10.4–15.1 | **+1.5** |
| χ² | published: game-level 5–95 | 6.2–8.1 | 12.2–14.7 | -4.1 |
| χ² | game-level 2.5–97.5 | 6.1–8.4 | 11.7–14.9 | -3.4 |
| χ² | prompt-level 2.5–97.5 | 5.4–8.8 | 10.9–15.9 | -2.1 |

## How many of the six overlap

| option | overlapping | reading |
|---|---|---|
| published: game-level 5–95 | **0 / 6** | what the paper shows now — every pair looks cleanly separated |
| game-level 2.5–97.5 | **1 / 6** | same resampling as arena-hard-auto, honest 95% |
| prompt-level 2.5–97.5 | **4 / 6** | same resampling AND percentiles as the bottom panel |

## What the overlap does and does not mean

The top panel compares each arm to a common external baseline, so the two intervals are *unpaired*.
Two unpaired 95% intervals can overlap while the paired difference is still decisive, and that is
exactly the case here: the bottom panel is the paired head-to-head on the same 500 prompts, and its
boxes sit at 57.2–64.4% with every whisker clear of 50%. Widening the top panel therefore does not
weaken the claim — it stops the top panel from overstating the strength of the weaker comparison.

## Files

| file | top panel |
|---|---|
| `results/stageB_normcmp_wr_h2h.{png,pdf}` | published: point + 5–95 game-level error bar |
| `results/stageB_normcmp_wr_h2h_boxtop_game.{png,pdf}` | box, 1000 game-level replicates, 2.5–97.5 |
| `results/stageB_normcmp_wr_h2h_boxtop_prompt.{png,pdf}` | box, 1000 prompt-level replicates, 2.5–97.5 |

Regenerate with `python fig_normcmp.py [--top box] [--top-boot prompt|game]`.
