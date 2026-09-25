"""
fig_figure9.py — Figure 9: the tabular recovery panel pair, at peak 0.7.

Two panels, not the three of f02: the "resampled" middle panel is dropped, leaving
  left   Delta_pi vs the MC budget, on-policy (fresh rollouts)
  right  exact vs single-sample bars, per divergence, Amari | Canonical | exact

Separate from fig_tabular.py on purpose: this one turns on text.usetex, which is a global rcParam,
and f02 and the rest of that module are not meant to change. Reads the same cached results.json.

Run from python/:  python fig_figure9.py [results.json ...]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# real LaTeX, matching Figure 7/8; every label here is ASCII + $...$
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": "\n".join([r"\usepackage{amsmath}",
                                                       r"\usepackage{amssymb}"])})

import fig_tabular as ft
from fig_tabular import REGKEYS, _COLORS as COLORS, _SHORT_TEX as SHORT_TEX, _kl_kw

AX_FS, TICK_FS, LEG_FS = 15, 12, 12


def figure9(agg_p, peak, nmc):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    axL, axR = axes
    ymax = 0.0

    for rk in REGKEYS:                       # left: on-policy only
        cs = np.array([agg_p["on"][rk][n][0] for n in nmc])
        ci = np.array([agg_p["on"][rk][n][1] for n in nmc])
        axL.plot(nmc, cs, marker="o", ms=4, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk))
        axL.fill_between(nmc, cs - ci, cs + ci, color=COLORS[rk], alpha=0.13, zorder=2)
        ymax = max(ymax, (cs + ci).max())
    axL.set_xscale("log", base=2); axL.set_xticks(nmc)
    axL.set_xticklabels([rf"${n}$" for n in nmc], fontsize=TICK_FS)
    axL.tick_params(axis="y", labelsize=TICK_FS)
    axL.set_xlabel(r"MC budget  $n$", fontsize=AX_FS)
    axL.set_ylabel(r"recovery gap $\Delta_\pi$ = mean TV$(\pi_\theta \,\|\, \pi^\star)$", fontsize=AX_FS)
    axL.grid(alpha=0.2)
    axL.legend(fontsize=LEG_FS, ncol=2, loc="upper right")

    ymax = max(ymax, ft.draw_bars(axR, agg_p, peak, leg_fs=LEG_FS, tick_fs=TICK_FS, title_fs=None))
    for ax in axes:
        ax.set_ylim(0, ymax * 1.30)          # headroom so neither legend sits on the data
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    stem = "figs/figure9_tabular_recovery"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close()
    return f"{stem}.png"


def main():
    man, results, _ = ft.load(sys.argv[1:] or None)
    agg, peaks, mdps = ft.aggregate(results)
    nmc = list(man["config"]["nmc"])
    peak = 0.7 if 0.7 in peaks else peaks[0]
    print(f"[data] {len(results)} cells · {len(mdps)} MDPs · peak {peak} · n_mc {nmc}")
    print("[saved]", figure9(agg[peak], peak, nmc))


if __name__ == "__main__":
    main()
