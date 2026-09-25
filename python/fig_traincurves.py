"""
fig_traincurves.py — 1.7B training curves, Amari vs Canonical, from the W&B CSV exports.

  figs/traincurves_1p7b   (left) gradient norm, log scale   (right) held-out reward margin
                          mean with the MIN-MAX band the export carries.

Colours are sampled from the paper's Amari/Canonical blocks: #b85327 and #68349a.
Typeset like Figures 7-9: real LaTeX, larger type.

Run from python/:  python fig_traincurves.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": "\n".join([r"\usepackage{amsmath}",
                                                       r"\usepackage{amssymb}"])})

CSV_DIR = "../llm/results/curves"
C_AMARI, C_CANON = "#b85327", "#68349a"          # sampled from the Amari / Canonical blocks
AX_FS, TICK_FS, LEG_FS = 16, 13, 14
ARMS = (("amari_1.7", "Amari form", C_AMARI),
        ("canon_1.7", "Canonical form (ours)", C_CANON))


def series(path, metric):
    """(step, mean, lo, hi) per arm, from a W&B multi-run CSV export."""
    rows = list(csv.DictReader(open(path)))
    step = np.array([float(r["Step"]) for r in rows])
    out = {}
    for key, _, _ in ARMS:
        col = f"Group: {key} - {metric}"
        m = np.array([float(r[col]) for r in rows])
        lo = np.array([float(r[col + "__MIN"]) for r in rows])
        hi = np.array([float(r[col + "__MAX"]) for r in rows])
        out[key] = (m, lo, hi)
    return step, out


def panel(ax, path, metric, ylabel, logy=False):
    step, data = series(path, metric)
    for key, label, color in ARMS:
        m, lo, hi = data[key]
        ax.plot(step, m, color=color, lw=2.2, label=label, zorder=3)
        ax.fill_between(step, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"training step", fontsize=AX_FS)
    ax.set_ylabel(ylabel, fontsize=AX_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.2, which="both")
    return step


def main():
    os.makedirs("figs", exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    panel(axL, f"{CSV_DIR}/Gradient Norm.csv", "grad_norm",
          r"Gradient norm  $\|\nabla_\theta \widehat{\mathcal{L}_\theta}\|$", logy=True)
    panel(axR, f"{CSV_DIR}/Evaluation Reward Margin.csv", "eval_margin",
          r"Evaluation reward margin")
    # legend above the curves, in a band of its own, so it never sits on the data
    lo, hi = axR.get_ylim()
    axR.set_ylim(lo, hi + (hi - lo) * 0.22)
    axL.legend(fontsize=LEG_FS, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    stem = "figs/traincurves_1p7b"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close()
    print("[saved]", stem + ".png (+ .pdf)")


if __name__ == "__main__":
    main()
