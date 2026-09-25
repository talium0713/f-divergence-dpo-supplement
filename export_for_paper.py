"""export_for_paper.py (B13) — copy the paper-ready figures into figure4overleaf/ under the fNN_* naming
the Overleaf repo uses, each with a JSON sidecar recording (source figure, source data, git commit,
export time) so "which run produced this figure" is always answerable for the Reproducibility Statement.

Does NOT touch ~/overleaf. Run from the repo root:  python export_for_paper.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

OUT = "figure4overleaf"

# The layered-MDP configuration the paper quotes verbatim. Asserted against
# data/tabular/run_*/results.json by fig_paper_layered.check_env() before any t0x figure is drawn.
LAYERED_CONFIG = {
    "env": {"depth": 4, "SN": 3, "NA": 3, "eps": 0.2, "gamma": 0.9,
            "reward": "Uniform(-0.8,0.8) on every (layer, state, action)",
            "transitions": "depend on the action only; s'=a w.p. 1-eps, else uniform over the rest"},
    "start": "uniform over the three layer-0 states",
    "pi_ref": "uniform at every state",
    "oracle": "Bradley-Terry on gamma-discounted trajectory return differences, temperature 1",
    "runs": "100 independently drawn MDPs, one training seed each",
    "weight": "calibrated per (regularizer, target peak) by bisection, not a fixed beta",
    "adiv": "alpha = 0.5 (DEFAULT_ADIV_A)",
    "roster": ["RKL", "alpha-div", "FKL", "JS", "squared Hellinger", "Pearson chi^2",
               "squared Euclidean"],
}

# paper name -> (source stem without extension, source-data provenance note)
FIGS = {
    "f01_toy_scaling":             ("python/figs/toy_Asize_ablation",
                                    "fig_toy_ablation.py — analytic Ψ(u) (left) + toy off-policy std sweep (right)"),
    "f02_tabular_recovery":        ("python/figs/tabular_headline_p70",
                                    "fig_tabular.py · python/data/tabular/run_headline_p70/results.json "
                                    "(peak 0.7, 2730 cells over 30 MDPs, root seed 20260629). "
                                    "This note said 100 MDPs until 2026-09-20; the run has always "
                                    "had 30. Euclidean is dropped from the paper figures."),
    "f03_adiv_normalization":      ("python/figs/adiv_compare_p80",
                                    "fig_adiv_compare.py · run_adiv_p80{,_kln}/results.json (T5, peak 0.8)"),
    "f03_adiv_normalization_paper":("python/figs/adiv_compare_p80_paper",
                                    "fig_adiv_compare.py --paper profile (text-width)"),
    "f10_adiv_morph":              ("python/figs/adiv_morph_p80",
                                    "fig_adiv.py · run_adiv_p80/results.json (T10, peak 0.8)"),
    "f11_adiv_invariant":          ("python/figs/adiv_invariant",
                                    "fig_adiv_invariant.py — normalization-invariant permissibility metric"),
    "f12_permissibility_bias":     ("python/figs/permissibility_bias",
                                    "fig_permissibility_bias.py — off-policy single-sample inner-term |bias| "
                                    "vs drift (canonical generators); RKL≡0 uniquely permissible, others fan out"),
    "f13_policy_3x7_canonical":    ("python/figs/part3_policy_3x7_p9",
                                    "run_canon_final.py run_2x7/fig_policy_3x7 — π* recovery under the exact "
                                    "inner term | off-policy Amari f'(1)=0 | off-policy canonical f'(1)=f''(1), "
                                    "peak 0.9 (high drift), 100 MDPs, pure off-policy (single logged a'). "
                                    "RKL has no Amari cell (u·log u is already canonical) and euc no canonical "
                                    "cell (not an f-divergence). RKL's canonical arm equals its exact arm to "
                                    "machine precision — the sample-free property."),
    # ── layered-MDP regeneration of the paper's legacy tree figures (round 0916-B, option c) ──
    "Figure9":                     ("python/figs/figure9_tabular_recovery",
                                    "fig_figure9.py — peak 0.7, two panels: Delta_pi vs the MC budget "
                                    "on-policy, and the exact-vs-single-sample bars (Amari | Canonical | "
                                    "exact) per divergence. The f02 'resampled' middle panel is dropped "
                                    "and the text is rendered by LaTeX, as in Figures 7 and 8."),
    "Figure7":                     ("python/figs/t01_single_state_variance",
                                    "fig_mechanism.py — 2x2: rows are the generator normalization (Amari "
                                    "f'(1)=0 vs canonical f'(1)=f''(1)), columns the value and the spread of "
                                    "the Eq. 9 estimator Psi_hat^(n). Amari RKL has ordinary nonzero variance; "
                                    "only canonical RKL is Psi == 1."),
    "Figure8":                     ("python/figs/t02_trajectory_compounding",
                                    "fig_mechanism.py — the same 2x2 split for the trajectory sum; "
                                    "non-admissible spread grows as sqrt(H-1) in both rows, canonical RKL "
                                    "stays exactly 0."),
    "Figure10":                    ("python/figs/t04_state_anatomy_p6",
                                    "fig_paper_layered_v2.py — the same 6x2 as Figure 2 at peak 0.6 "
                                    "(mildest drift)."),
    "t04_state_anatomy_p7":        ("python/figs/t04_state_anatomy_p7",
                                    "fig_paper_layered_v2.py — 6x2 at peak 0.7: columns are the six "
                                    "f-divergences, rows Amari vs canonical; pi* (dashed, recomputed "
                                    "with solve_dp) against the recovered policy over the twelve cells. "
                                    "Euclidean is dropped from the paper figures (no canonical arm)."),
    "Figure11":                    ("python/figs/t04_state_anatomy_p8",
                                    "fig_paper_layered_v2.py — the same 6x2 as Figure 2 at peak 0.8."),
    "Figure12":                    ("python/figs/t04_state_anatomy_p9",
                                    "fig_paper_layered_v2.py — the same 6x2 as Figure 2 at peak 0.9 "
                                    "(high drift); only RKL's canonical arm still matches its exact arm here."),
    "t06_alpha_sweep_p6":          ("python/figs/t06_alpha_sweep_p6",
                                    "fig_paper_layered_v2.py — alpha family, Amari vs canonical, peak 0.6."),
    "t06_alpha_sweep_p7":          ("python/figs/t06_alpha_sweep_p7",
                                    "fig_paper_layered_v2.py — alpha family, Amari vs canonical, peak 0.7."),
    "t06_alpha_sweep_p8":          ("python/figs/t06_alpha_sweep_p8",
                                    "fig_paper_layered_v2.py — alpha family, Amari vs canonical, peak 0.8."),
    "t06_alpha_sweep_p9":          ("python/figs/t06_alpha_sweep_p9",
                                    "fig_paper_layered_v2.py — alpha family, Amari vs canonical, peak 0.9."),
    "Figure14":                    ("python/figs/traincurves_1p7b",
                                    "fig_traincurves.py — 1.7B training curves from the W&B CSV exports "
                                    "in llm/results/curves: gradient norm on a log axis (left) and the "
                                    "held-out reward margin (right), mean with the export's min-max band. "
                                    "Amari vs canonical."),
    "Figure13":                    ("python/figs/t06_alpha_sweep",
                                    "fig_paper_layered_v2.py — the three peaks side by side, for the "
                                    "main body. Same data as the three standalone files."),
    "Figure6":                     ("python/figs/t07_calibration",
                                    "fig_paper_layered.py — policy peak vs the regularization weight, with "
                                    "the calibrated anchors at all three target peaks (MDP 0). Typeset like "
                                    "Figures 7-8: real LaTeX, axis labels 15pt, ticks and annotations 12pt, "
                                    "legend 11pt."),
    "Figure1":                     ("python/figs/tabular_bars_p70",
                                    "fig_tabular.py fig_bars_only — the exact-vs-off-policy panel of "
                                    "f02 on its own, at PEAK 0.7 over run_headline_p70 (2730 cells, "
                                    "30 MDPs). Three bars per divergence: exact "
                                    "closed-form inner term over all a' (hatched ///), then the "
                                    "single-logged-action estimator under the Amari normalization "
                                    "f'(1)=0 and under the canonical f'(1)=f''(1), with 95% CIs. "
                                    "Euclidean is dropped from all paper figures — it is a Bregman, "
                                    "not an f-divergence, so it has no canonical bar at all. Built at "
                                    "7.6x5.0 in with 17pt labels for placement at about half width"),
    "Figure2":                     ("python/figs/t04_state_anatomy_p7",
                                    "fig_paper_layered_v2.py fig_anatomy — the state-by-state anatomy "
                                    "at PEAK 0.7, 6 columns (the f-divergences) x 2 rows (Amari "
                                    "f'(1)=0 above, canonical f'(1)=f''(1) below). Dashed grey is pi*, "
                                    "the REGULARIZED optimum recomputed per cell with solve_dp from the "
                                    "cached reward draw and the calibrated weight — not any arm's "
                                    "recovered policy. Solid is the recovered policy; the per-cell "
                                    "Delta_pi is its mean TV from pi*. x ticks are the four layers of "
                                    "the layered MDP (depth 4, SN 3, NA 3, eps 0.2, gamma 0.9). "
                                    "Euclidean is dropped from the paper figures: not an f-divergence, "
                                    "so its canonical cell never existed and used to render as a "
                                    "hatched placeholder"),
    "Figure4":                     ("python/figs/figure4_regularity_gap",
                                    "fig_figure4.py — the composed Figure 4, 40/60 on one gridspec "
                                    "row so the two halves share an axes height. (left) tabular: the "
                                    "alpha family at PEAK 0.8 under both normalizations, mean over "
                                    "MDPs with a 95% CI band; alpha=1 is the same divergence on both "
                                    "curves, so the vertical distance there is the regularity gap "
                                    "itself (was figs/t06_alpha_sweep_p8, whose 'peak 0.8' title is "
                                    "dropped — the caption has to state the peak). (right) Arena-Hard "
                                    "v0.1 stacked: win rate against the fixed gpt-4-0314 baseline "
                                    "above, the direct canonical-vs-Amari head-to-head below, every "
                                    "box 1000 prompt-level bootstrap replicates at 2.5/25/50/75/97.5 "
                                    "(was fL3). Judge gpt-4.1-mini-2025-04-14, 500 prompts, two "
                                    "position-swapped games each — also caption material, the panels "
                                    "no longer say it. Data: python/data/tabular/run_adiv_p80{,_kln}/"
                                    "results.json, llm/results/bench/arena_v01/topboot_normcmp.json "
                                    "and h2h_normcmp/*_mini2*.json"),
    "Figure3":                     ("python/figs/figure3_noise_scaling",
                                    "fig_figure3.py — the composed Figure 3, 50/25/25 on one gridspec "
                                    "row so all three panels share an axes height. (left) toy: "
                                    "std_{a~pi_ref}[Psi(u_a)] vs action-set size, median + IQR over 10 "
                                    "seeds, pi_ref uniform and pi = softmax of Gaussian logits, broken "
                                    "y-axis with RKL's exact 0 below the break — mechanism only, the "
                                    "magnitude is toy-dependent. (middle) measured: per-token Psi(u_a) "
                                    "per divergence, box with 1-99% whiskers on symlog. (right) "
                                    "measured: histogram of log10(pi_theta/pi_ref) per logged token, "
                                    "min u = 1.5e-16. Both measured panels come from "
                                    "llm/results/stageA_qwen3_1p7b_raw.npz: pi_theta = Qwen3-1.7B, "
                                    "pi_ref = Qwen3-1.7B-Base, 133153 logged tokens of "
                                    "llm/data/uf_test_prefs.jsonl, forward pass only, no training. "
                                    "Built at 13x3.9 in with 18pt labels so it survives being scaled "
                                    "to about half width on the page; the panels carry no titles, so "
                                    "the caption must state the models, the whisker convention and "
                                    "min u"),
    "f01_toy_scaling_asize":       ("python/figs/toy_Asize_ablation_right",
                                    "fig_toy_ablation.py --panel right — the |A|-scaling panel of "
                                    "f01_toy_scaling on its own, for composing beside the Stage A "
                                    "panels: off-policy inner-term noise std_{a~pi_ref}[Psi(u_a)] vs "
                                    "action-set size, median with IQR band over 10 seeds, broken "
                                    "y-axis with RKL's exact 0 in the lower strip. Toy: pi_ref "
                                    "uniform, pi = softmax of Gaussian logits (SCALE 3.0), so the "
                                    "mechanism is the claim and the magnitude is not — Stage A "
                                    "(fL0) measures the real one"),
    "fL0_stagea_inner_term":       ("llm/results/stageA_qwen3_1p7b",
                                    "stage_a_measure.py --replot — Stage A, measured with NO training: "
                                    "pi_theta = Qwen3-1.7B, pi_ref = Qwen3-1.7B-Base, 133153 logged "
                                    "tokens from llm/data/uf_test_prefs.jsonl. Left: per-token inner "
                                    "term Phi(u) per divergence, box with 1-99% whiskers on a symlog "
                                    "axis — RKL is the flat line at Phi=1. Right: histogram of the "
                                    "per-token log10 policy/reference ratio, min u = 1.5e-16. The "
                                    "figure carries no model names or token count of its own (they "
                                    "were dropped from the title on 2026-09-20), so a caption must "
                                    "state them; arrays in llm/results/stageA_qwen3_1p7b_raw.npz"),
    "fL1_arena_winrate":           ("llm/results/stageB_divergence_wr",
                                    "fig_permissibility_wr.py · llm/results/bench/arena_v01/divergence_wr.json"),
    "fL2_head_to_head":            ("llm/results/stageB_divergence_h2h",
                                    "fig_permissibility_h2h.py · llm/results/bench/arena_v01/h2h/*.json"),
    "fL3_normcmp_wr_h2h":          ("llm/results/stageB_normcmp_wr_h2h",
                                    "fig_normcmp.py — canonical(f'=f'') vs Amari(f'=0) on Arena-Hard v0.1, "
                                    "judge gpt-4.1-mini-2025-04-14, 500 prompts. Both panels are boxes on 1000 "
                                    "prompt-level bootstrap replicates at the 2.5/25/50/75/97.5 percentiles: "
                                    "top (vs the gpt-4-0314 baseline) from llm/results/bench/arena_v01/"
                                    "topboot_normcmp.json (arena_boot_dump.py, recovered 2026-09-19 from the "
                                    "existing judgments), bottom (head-to-head) from h2h_normcmp/*_mini2_raw.json "
                                    "(pairwise_h2h.py, re-judged 2026-09-05 to keep the replicates). Before "
                                    "2026-09-19 the top panel was a transcribed point + 5-95 GAME-level error "
                                    "bar, i.e. a 90% interval on independently-drawn swapped games"),
    "fL3new_normcmp_wr_prompts":   ("llm/results/stageB_normcmp_wr_prompts",
                                    "fig_normcmp.py --bottom mix — same top panel as fL3; bottom is how "
                                    "the 500 Arena-Hard prompts actually split once each is judged twice "
                                    "with the positions swapped (canonical both / no consistent winner / "
                                    "amari both), from h2h_normcmp/*_mini2_raw.json"),
    "fL4_vs_base_normcmp":         ("llm/results/stageB_vs_base",
                                    "fig_vs_base.py — each form against the model it started from "
                                    "(Qwen3-1.7B-Base answers = Base_repo) over the canonical-vs-amari "
                                    "boxes; llm/results/bench/arena_v01/h2h_vs_base/*.json + "
                                    "h2h_normcmp/*_mini2_raw.json, judge gpt-4.1-mini-2025-04-14"),
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(OUT, exist_ok=True)
    commit = git_commit()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    done, skipped = [], []
    for name, (stem, note) in FIGS.items():
        exts = [e for e in ("png", "pdf") if os.path.exists(f"{stem}.{e}")]
        if not exts:
            skipped.append((name, stem)); continue
        for e in exts:
            shutil.copyfile(f"{stem}.{e}", f"{OUT}/{name}.{e}")
        side = {"paper_name": name, "source_figure": stem, "source_data": note,
                "formats": exts, "git_commit": commit, "exported_at": now}
        # layered regeneration: record the full config. Figures 6-8 are t07/t01/t02 under their
        # paper numbers, so they are named here rather than caught by the "t0" prefix.
        if name.startswith("t0") or name in ("Figure6", "Figure7", "Figure8",
                                             "Figure10", "Figure11", "Figure12"):
            side["config"] = LAYERED_CONFIG
        json.dump(side, open(f"{OUT}/{name}.json", "w"), indent=2)
        done.append((name, "+".join(exts)))
    print(f"exported {len(done)} figures -> {OUT}/  (commit {commit[:8]})")
    for n, e in done:
        print(f"  {n}.{{{e}}}")
    if skipped:
        print("skipped (not generated yet):")
        for n, s in skipped:
            print(f"  {n}  <- {s}.{{png,pdf}}")


if __name__ == "__main__":
    main()
