"""
fig_figure4.py — Figure 4: the regularity gap in the tabular sweep and on Arena-Hard.

Two panels, 40/60, on one gridspec row so they share an axes height:

  left (40%)   tabular: the alpha family at peak 0.8 under both normalizations. alpha=1 is the SAME
               divergence on both curves — only the generator's normalization differs — so the
               vertical distance there is the regularity gap itself. Was figs/t06_alpha_sweep_p8.
  right (60%)  Arena-Hard v0.1, stacked: win rate against the fixed gpt-4-0314 baseline on top, the
               direct canonical-vs-Amari head-to-head below. Was llm/results/stageB_normcmp_wr_h2h
               (fL3); the right column's two panels together match the left panel's height.

Both halves use the same two colours for the two normalizations, so a reader carries the pairing
across the figure without a second legend.

Every box is 1000 bootstrap replicates at the 2.5/25/50/75/97.5 percentiles, prompt-level on the
Arena side (see llm/arena_boot_dump.py). Reads, all as data:
  python/data/tabular/run_adiv_p80{,_kln}/results.json      the tabular sweep
  llm/results/bench/arena_v01/topboot_normcmp.json          the baseline panel's replicates
  llm/results/bench/arena_v01/h2h_normcmp/*_mini2*.json     the head-to-head panel's

No titles beyond the left panel's, and no judge/whisker notes: the caption carries the judge
(gpt-4.1-mini-2025-04-14, 500 prompts, two position-swapped games each) and the peak.

Fonts assume the figure is placed at roughly half of --width on the page.

Run from python/:  python fig_figure4.py [--label-fs 17] [--tick-fs 14]
Out: figs/figure4_regularity_gap.{png,pdf}  ->  exported by export_for_paper.py as Figure4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from regularizers import COLORS
from fig_paper_layered_v2 import _agg_adiv

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figs")
ADIV = (os.path.join(HERE, "data/tabular/run_adiv_p80/results.json"),
        os.path.join(HERE, "data/tabular/run_adiv_p80_kln/results.json"))
ARENA = os.path.join(HERE, "..", "llm", "results", "bench", "arena_v01")
TOPBOOT = os.path.join(ARENA, "topboot_normcmp.json")
H2H = os.path.join(ARENA, "h2h_normcmp")

C_AM, C_CA = "#6b7280", "#b0224b"                 # Amari / canonical, shared with the tabular panel
# One group width for both Arena panels: the bottom draws a single box of GROUP_W, the top draws
# two side by side whose outer edges land on the same x, so the columns line up down the figure.
GROUP_W, PAIR_GAP = 0.52, 0.06
BOX_W = (GROUP_W - PAIR_GAP) / 2                  # 0.23
BOX_DX = (GROUP_W - BOX_W) / 2                    # 0.145 -> outer edge at GROUP_W/2
DIVS = ["RKL", "α-div", "FKL", "JS", "Hel", "χ²"]
BAR_KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]
H2H_KEY = {"RKL": "kl", "α-div": "adiv", "FKL": "fkl", "JS": "js", "Hel": "hel", "χ²": "chi2"}


def pct(xs, p):
    i = p * (len(xs) - 1); lo = int(i); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (i - lo)) + xs[hi] * (i - lo)


def h2h_box(div):
    """Canonical's win rate against Amari: the stored replicates are Amari's, so 1 - each."""
    summ = json.load(open(f"{H2H}/h2h_{H2H_KEY[div]}_amari_vs_canon_mini2.json"))
    boot = json.load(open(f"{H2H}/h2h_{H2H_KEY[div]}_amari_vs_canon_mini2_raw.json"))["boot"]
    c = sorted(100.0 * (1.0 - b) for b in boot)
    return {"med": round(100 - summ["win_rate"], 1), "q1": pct(c, .25), "q3": pct(c, .75),
            "whislo": pct(c, .025), "whishi": pct(c, .975), "label": div}


def top_box(div, arm):
    reps = sorted(json.load(open(TOPBOOT))["arms"][f"{div}|{arm}"]["boot_prompt"])
    return {"med": pct(reps, .5), "q1": pct(reps, .25), "q3": pct(reps, .75),
            "whislo": pct(reps, .025), "whishi": pct(reps, .975), "label": f"{div}|{arm}"}


def style(bp, col, n):
    for k in range(n):
        bp["boxes"][k].set(facecolor=col, alpha=0.26, edgecolor=col, linewidth=1.3)
        bp["medians"][k].set(color=col, linewidth=2.2)
        for art in (bp["whiskers"][2 * k], bp["whiskers"][2 * k + 1],
                    bp["caps"][2 * k], bp["caps"][2 * k + 1]):
            art.set(color=col, linewidth=1.2, alpha=0.9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-fs", type=float, default=17.0)
    ap.add_argument("--tick-fs", type=float, default=14.0)
    ap.add_argument("--legend-fs", type=float, default=13.0)
    ap.add_argument("--title-fs", type=float, default=16.0)
    ap.add_argument("--width", type=float, default=15.0)
    ap.add_argument("--height", type=float, default=5.4)
    ap.add_argument("--left", type=float, default=0.062)
    ap.add_argument("--bottom", type=float, default=0.115)
    ap.add_argument("--wspace", type=float, default=0.30)
    ap.add_argument("--legend-headroom", type=float, default=7.0,
                    help="win-rate points of empty axis above the tallest whisker, for the legend")
    a = ap.parse_args()
    LFS, TFS = a.label_fs, a.tick_fs

    os.makedirs(FIGDIR, exist_ok=True)
    fig = plt.figure(figsize=(a.width, a.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 6], wspace=a.wspace,
                          left=a.left, right=0.99, top=0.865, bottom=a.bottom)
    axT = fig.add_subplot(gs[0])                               # tabular alpha sweep
    gsR = gs[1].subgridspec(2, 1, hspace=0.07)
    axU, axL = fig.add_subplot(gsR[0]), fig.add_subplot(gsR[1])

    # ---------------- left: the alpha sweep at peak 0.8 ----------------
    _, A, am = _agg_adiv(ADIV[0])
    _, _, ca = _agg_adiv(ADIV[1])
    keep = [x for x in A if not (0.9 < x < 1.1 and abs(x - 1.0) > 1e-9)]   # drop the dense probes
    Ak = np.array(keep)
    for d, col, lab in ((am, C_AM, r"Amari  $f'(1)=0$"),
                        (ca, C_CA, r"Canonical  $f'(1)=f''(1)$")):
        m = np.array([d[x][0] for x in keep]); ci = np.array([d[x][1] for x in keep])
        axT.plot(Ak, m, marker="o", ms=3.5, lw=1.8, color=col, label=lab)
        axT.fill_between(Ak, m - ci, m + ci, color=col, alpha=0.15)
    axT.axvline(1.0, color=COLORS["kl"], ls=":", lw=1.1, alpha=0.7)

    y_am, y_ca, acc = am[1.0][0], ca[1.0][0], COLORS["kl"]
    axT.scatter([1.0, 1.0], [y_am, y_ca], s=170, facecolors="none", edgecolors=acc,
                linewidths=2.0, zorder=9)
    axT.annotate("", xy=(1.0, y_ca), xytext=(1.0, y_am), zorder=8,
                 arrowprops=dict(arrowstyle="<->", color=acc, lw=2.0))
    axT.text(0.96, (y_am + y_ca) / 2, "Regularity\ngap", ha="right", va="center",
             fontsize=LFS + 1, fontweight="bold", color=acc)
    # +0.055, not +0.022: the Amari curve keeps rising past alpha=1 to ~0.474, so the label sat on it
    axT.annotate(r"$f(u)=u\ln u-(u-1)$", xy=(1.0, y_am), xytext=(1.05, y_am + 0.055),
                 fontsize=LFS - 4, color=acc, ha="left", va="bottom")
    axT.annotate(r"$f(u)=u\ln u$", xy=(1.0, y_ca), xytext=(1.05, y_ca - 0.022),
                 fontsize=LFS - 4, color=acc, ha="left", va="top")

    axT.set_title(r"Divergence-family parameter  $\alpha$" "\n"
                  r"($\alpha\!\to\!0$: FKL   $\alpha\!=\!1$: RKL   $\alpha\!=\!2$: $\chi^2$)",
                  fontsize=a.title_fs)
    axT.set_xlabel(r"Parameter  $\alpha$", fontsize=LFS)
    axT.set_ylabel(r"Recovery gap  $\Delta_\pi = \mathbb{E}\,[\mathrm{TV}(\pi_\theta\,\|\,\pi^\star)]$",
                   fontsize=LFS)
    axT.set_ylim(0, max(axT.get_ylim()[1], y_am + 0.11))
    axT.grid(alpha=0.2)
    axT.legend(fontsize=a.legend_fs, loc="lower left", frameon=False)

    # ---------------- right: Arena-Hard, baseline above, head-to-head below ----------------
    x = list(range(len(DIVS)))
    for arm, col, dx, zo in (("amari", C_AM, -BOX_DX, 5), ("canon", C_CA, +BOX_DX, 6)):
        stats = [top_box(d, arm) for d in DIVS]
        style(axU.bxp(stats, positions=[i + dx for i in x], widths=BOX_W, patch_artist=True,
                      showfliers=False, zorder=zo), col, len(stats))
        for i, st in enumerate(stats):           # the numbers fL3 printed beside each arm
            axU.text(i + dx, st["whishi"] + 0.25, f"{st['med']:.1f}", color=col, fontsize=TFS - 3,
                     va="bottom", ha="center", fontweight="bold" if arm == "canon" else "normal")
    allb = [top_box(d, arm) for d in DIVS for arm in ("amari", "canon")]
    lo, hi = min(b["whislo"] for b in allb), max(b["whishi"] for b in allb)
    # Headroom on top: the legend now sits INSIDE the axes, above every box, and the medians are
    # printed over each whisker. hi is the tallest whisker (RKL canonical, 17.4).
    axU.set_ylim(lo - 0.9, hi + a.legend_headroom)
    axU.set_ylabel("vs gpt-4\nWin Rate (%)", fontsize=LFS)
    axU.set_title("Arena-Hard Results", fontsize=a.title_fs, pad=8)

    stats = [h2h_box(d) for d in DIVS]
    bpb = axL.bxp(stats, positions=x, widths=GROUP_W, patch_artist=True, showfliers=False, zorder=2)
    for k, key in enumerate(BAR_KEYS):                          # per-divergence colour, one box at a time
        bpb["boxes"][k].set(facecolor=COLORS[key], alpha=0.24, edgecolor=COLORS[key], linewidth=1.4)
        bpb["medians"][k].set(color=COLORS[key], linewidth=2.4)
        for art in (bpb["whiskers"][2 * k], bpb["whiskers"][2 * k + 1],
                    bpb["caps"][2 * k], bpb["caps"][2 * k + 1]):
            art.set(color=COLORS[key], linewidth=1.4, alpha=0.9)
    for k, (st, key) in enumerate(zip(stats, BAR_KEYS)):
        axL.text(k, st["whishi"] + 0.5, f"{st['med']:.1f}", color=COLORS[key], fontsize=TFS - 3,
                 va="bottom", ha="center", fontweight="bold")
    axL.set_ylim(52, 71.5); axL.set_yticks([55, 60, 65, 70])
    axL.set_ylabel("Canonical vs Amari\nWin Rate (%)", fontsize=LFS)
    axL.set_xticks(x); axL.set_xticklabels(DIVS, fontsize=TFS)
    axU.set_xlim(-0.6, len(DIVS) - 0.4)
    plt.setp(axU.get_xticklabels(), visible=False); axU.tick_params(axis="x", length=0)
    for ax in (axU, axL):
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.18)
        ax.yaxis.set_label_coords(-0.058, 0.5)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CA, ms=11,
                     label=r"Canonical (Ours)  $f'(1)=f''(1)$"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AM, ms=11,
                     label=r"Amari  $f'(1)=0$")]
    axU.legend(handles=legend, loc="upper center", ncol=2, fontsize=a.legend_fs, frameon=False,
               columnspacing=2.4, handletextpad=0.6, borderpad=0.55)

    for ax in (axT, axU, axL):
        ax.tick_params(labelsize=TFS)

    stem = os.path.join(FIGDIR, "figure4_regularity_gap")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    print(f"wrote {stem}.png/.pdf   ({a.width}x{a.height} in, label {LFS}pt, tick {TFS}pt)")


if __name__ == "__main__":
    main()
