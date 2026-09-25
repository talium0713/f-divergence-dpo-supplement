"""
fig_figure3.py — Figure 3: the single-sample noise grows with the action set.

Three panels on one row, composed from what were separate figures so the paper carries one figure
instead of three and the reader can put the toy prediction next to the LLM measurement:

  left  (40%)  toy: off-policy noise std_{a~pi_ref}[Psi(u_a)] against the action-set size |A|, median
               with an IQR band over N_SEEDS seeds. Broken y-axis — RKL is exactly 0 at every |A| and
               would otherwise be invisible under a log axis shared with the others. Same computation
               as fig_toy_ablation.py, whose offpolicy_phi_std is reused directly.
  middle (30%) measured: the per-token integrand Psi(u_a) per divergence on real logged tokens, box on
               a symlog axis, whiskers 1-99%.
  right  (30%) measured: how off-policy those tokens are — the histogram of log10(pi_theta/pi_ref).

The two measured panels read llm/results/stageA_qwen3_1p7b_raw.npz, the arrays stage_a_measure.py
saved. Nothing here needs torch or a GPU: llm/divergences.py mirrors python/regularizers.py for
KEYS, SHORT and COLORS, so the LLM panels are drawn from the numpy side.

The three panels sit in one gridspec row, so their axes have identical height however the widths
split — the point of the figure is to read the toy trend and the measured spread against each other.

Fonts are sized for a figure that will be scaled DOWN into the text: built at --width inches and
placed at roughly half that, so an 18pt label lands near 9pt on the page. --label-fs / --tick-fs
retune without editing anything.

Run from python/:  python fig_figure3.py [--label-fs 18] [--tick-fs 15] [--width 13]
Out: figs/figure3_noise_scaling.{png,pdf}  ->  exported by export_for_paper.py as Figure3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from regularizers import COLORS, SHORT, SHORT_TEX
from fig_toy_ablation import offpolicy_phi_std, A_GRID, N_SEEDS, KEYS as TOY_KEYS

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

FIGDIR = os.path.join(os.path.dirname(__file__), "figs")
NPZ = os.path.join(os.path.dirname(__file__), "..", "llm", "results", "stageA_qwen3_1p7b_raw.npz")
CACHE = os.path.join(FIGDIR, "_figure3_toycache.npz")
LLM_KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]      # euc excluded, as in the toy panel:
# it is measured and stored in the npz, just not shown — see fig_toy_ablation.KEYS for why.


def toy_curves():
    """median / q25 / q75 of the toy noise per divergence over A_GRID, cached (it is the slow part)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if (list(z["a_grid"]) == list(A_GRID) and int(z["n_seeds"]) == N_SEEDS
                and [str(k) for k in z["keys"]] == list(TOY_KEYS)):   # keys too: euc joined
            return {k: (z[f"{k}_med"], z[f"{k}_lo"], z[f"{k}_hi"]) for k in TOY_KEYS}
    out, flat = {}, {"a_grid": np.array(A_GRID), "n_seeds": N_SEEDS,
                     "keys": np.array(TOY_KEYS)}   # plain str array: np.load refuses object without allow_pickle
    for k in TOY_KEYS:
        med, lo, hi = [], [], []
        for na in A_GRID:
            v = np.array([offpolicy_phi_std(k, na, seed=s) for s in range(N_SEEDS)])
            med.append(np.median(v)); lo.append(np.quantile(v, .25)); hi.append(np.quantile(v, .75))
        out[k] = (np.array(med), np.array(lo), np.array(hi))
        flat[f"{k}_med"], flat[f"{k}_lo"], flat[f"{k}_hi"] = out[k]
    os.makedirs(FIGDIR, exist_ok=True)
    np.savez(CACHE, **flat)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-fs", type=float, default=18.0)
    ap.add_argument("--tick-fs", type=float, default=15.0)
    ap.add_argument("--legend-fs", type=float, default=11.0)
    ap.add_argument("--width", type=float, default=13.0, help="figure width in inches")
    ap.add_argument("--height", type=float, default=4.3)
    ap.add_argument("--left", type=float, default=0.088, help="left margin (room for the y-label)")
    ap.add_argument("--bottom", type=float, default=0.235,
                    help="bottom margin (the middle panel's 45-degree tick labels live here)")
    ap.add_argument("--wspace", type=float, default=0.36)
    ap.add_argument("--top", type=float, default=0.955)
    ap.add_argument("--ybot", type=float, default=2e4,
                    help="bottom of the left panel's log strip")
    ap.add_argument("--ytop", type=float, default=2e8,
                    help="top of the left panel's y-axis; the headroom the legend sits in")
    a = ap.parse_args()
    LFS, TFS = a.label_fs, a.tick_fs

    os.makedirs(FIGDIR, exist_ok=True)
    fig = plt.figure(figsize=(a.width, a.height))
    # 40 / 30 / 30. One row, so all three panels get the same axes height; the left panel's broken
    # pair splits that height 5:1 between the non-permissible band and RKL's exact zero.
    # Margins are hand-set rather than tight_layout: the wavy break is drawn in figure coordinates
    # after the layout is final, and a layout engine that re-fits afterwards would slide it. They are
    # sized for the defaults below — raising --label-fs much past 18 wants a larger --left too.
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 3, 3], wspace=a.wspace,
                          left=a.left, right=0.988, top=a.top, bottom=a.bottom)
    gsL = gs[0].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.10)
    axLt, axLb = fig.add_subplot(gsL[0]), None
    axLb = fig.add_subplot(gsL[1], sharex=axLt)
    axM, axR = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # ---------------- left: toy |A| scaling ----------------
    data = toy_curves()
    for k in TOY_KEYS:
        if k == "kl":
            continue
        med, lo, hi = data[k]
        axLt.plot(A_GRID, med, color=COLORS[k], lw=1.9, marker="o", ms=4)
        axLt.fill_between(A_GRID, lo, hi, color=COLORS[k], alpha=0.12)
    axLt.set_xscale("log"); axLt.set_yscale("log")
    # The strip has to span two very different bands now: the non-permissible f-divergences at
    # 1e5-1e7 rising with |A|, and euc FALLING from 1.6e4 to 0.33 over the same sweep. Headroom
    # Headroom above the FKL curve (max ~9.4e6) so the legend sits in empty axis: with only
    # the f-divergence band to cover, 3.7 decades keeps the IQR bands readable.
    axLt.set_ylim(a.ybot, a.ytop)
    handles = [Line2D([0], [0], color=COLORS[k], lw=3.0 if k == "kl" else 1.9, marker="o", ms=4,
                      label=SHORT_TEX[k]) for k in TOY_KEYS]
    axLt.legend(handles=handles, ncol=3, fontsize=a.legend_fs, framealpha=0.9,
                loc="upper left")
    axLt.set_ylabel(r"$\mathrm{std}_{a\sim\pi_{\mathrm{ref}}}[\Psi_a]$", fontsize=LFS)
    axLt.yaxis.set_label_coords(-0.145, 0.40)     # 0.40 centres it across the 5:1 broken pair;
                                              # -0.145 keeps the label clear of the 10^8 ticks

    axLb.plot(A_GRID, data["kl"][0], color=COLORS["kl"], lw=3.0, marker="o", ms=4)
    axLb.axhline(0.0, color="0.6", ls=":", lw=1)
    axLb.set_xscale("log"); axLb.set_ylim(-6e-16, 6e-16); axLb.set_yticks([0])
    # Dashed at the two reference scales, |A|=3 (tabular) and 152k (Qwen's vocab). Unlabelled — the
    # caption has to say which is which, or drop the lines too.
    for xa in (3, 152000):
        axLt.axvline(xa, color="0.8", ls="--", lw=1); axLb.axvline(xa, color="0.8", ls="--", lw=1)
    axLb.set_xlabel(r"Action-set size $|A|$", fontsize=LFS)
    axLt.spines["bottom"].set_visible(False)
    axLt.tick_params(which="both", bottom=False, labelbottom=False)
    axLb.spines["top"].set_visible(False); axLb.tick_params(which="both", top=False)

    # ---------------- middle + right: the measured panels ----------------
    npz = np.load(NPZ)
    phi = [np.asarray(npz[f"phi_{k}"], dtype=float) for k in LLM_KEYS]
    bp = axM.boxplot(phi, positions=np.arange(len(LLM_KEYS)), widths=0.62, whis=(1, 99),
                     showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=1.2))
    for patch, k in zip(bp["boxes"], LLM_KEYS):
        patch.set_facecolor(COLORS[k]); patch.set_alpha(0.85)
    axM.set_yscale("symlog", linthresh=1.0)
    # symlog labels every decade, which is ten labels over this range and unreadable at the size the
    # panel is placed. Keep the three that carry meaning — the RKL line at 1, the sign change at 0 —
    # and then every second decade down; the box edges are read against those, not off the ticks.
    axM.set_yticks([1e1, 1e0, 0, -1e2, -1e4, -1e6])
    axM.axhline(1.0, color="0.45", ls=":", lw=1); axM.axhline(0.0, color="0.75", lw=0.6)
    axM.set_xticks(np.arange(len(LLM_KEYS)))
    axM.set_xticklabels([SHORT[k] for k in LLM_KEYS], rotation=45, ha="right")
    axM.set_ylabel(r"$\Psi_a$   (symlog)", fontsize=LFS)

    lg = (npz["log_u"] / np.log(10)).astype(float)
    axR.hist(lg, bins=100, color="#4065E9", alpha=0.85)
    axR.set_yscale("log"); axR.axvline(0.0, color="0.5", ls="--", lw=1)
    axR.set_xlim(np.floor(lg.min()) - 0.5, np.ceil(lg.max()) + 0.5)
    axR.set_xlabel(r"$\log_{10}(\pi_\theta/\pi_{\mathrm{ref}})$", fontsize=LFS)
    axR.set_ylabel("Token count", fontsize=LFS)

    for ax in (axLt, axLb, axM, axR):
        ax.tick_params(labelsize=TFS)

    # wavy break across the left panel's gap, in figure coords so both squiggles keep the same
    # amplitude regardless of the 5:1 split. Drawn after layout — it needs the final positions.
    fig.canvas.draw()
    bt, bb = axLt.get_position(), axLb.get_position()
    xx = np.linspace(bt.x0, bt.x1, 600)
    yc = 0.5 * (bt.y0 + bb.y1)
    wave = 0.008 * np.sin(2 * np.pi * 22 * (xx - bt.x0) / (bt.x1 - bt.x0))
    for dy in (0.007, -0.007):
        fig.add_artist(Line2D(xx, yc + dy + wave, color="0.30", lw=1.6,
                              solid_capstyle="round", clip_on=False, zorder=60))

    stem = os.path.join(FIGDIR, "figure3_noise_scaling")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    print(f"wrote {stem}.png/.pdf   ({a.width}x{a.height} in, label {LFS}pt, tick {TFS}pt)")


if __name__ == "__main__":
    main()
