"""
make_figures.py -- report figures comparing the imputation methods.

Reads results/rq1/augmented_results.csv (and plain_results.csv for figure 6)
and saves PNGs to results/rq1/figures_report/.

Run from src/experiments:
    python make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2] / "results" / "rq1"
OUT = ROOT / "figures_report"
OUT.mkdir(parents=True, exist_ok=True)

SPARS = [10, 20, 30, 40]
METHODS = [("perfect", "Perfect (baseline)"), ("em", "EM"),
           ("mice", "MICE"), ("soft", "Soft-Impute")]
COLOUR = {"perfect": "#555555", "em": "#d95f02", "mice": "#1b9e77",
          "soft": "#7570b3", "listwise": "#999999"}
MARK = {"perfect": "s", "em": "o", "mice": "^", "soft": "D", "listwise": "x"}
SE_NOTE = "Mean over the 20 seeds; error bars show ±1 standard error."

df = pd.read_csv(ROOT / "augmented_results.csv")


def mean_se(frame, col, method):
    g = frame[frame.method == method].groupby("sparsity")[col]
    return g.mean().reindex(SPARS), (g.std() / np.sqrt(g.count())).reindex(SPARS)


def style_axes(ax, ylabel, title):
    ax.set_xticks(SPARS)
    ax.set_xticklabels([f"{s}%" for s in SPARS])
    ax.set_xlabel("Percentage of data missing")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


def save(fig, name, note=None):
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("Saved", OUT / name)


def line_figure(col, ylabel, title, name, legend_kw=None, top=None):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, label in METHODS:
        m, se = mean_se(df, col, key)
        ax.errorbar(SPARS, m, yerr=se, label=label, color=COLOUR[key],
                    marker=MARK[key], capsize=3, lw=2, ms=6)
    style_axes(ax, ylabel, title)
    if top is not None:
        ax.set_ylim(top=top)
    ax.legend(frameon=False, **(legend_kw or {}))
    save(fig, name, SE_NOTE)


# 1 and 2: KL and SHD against the amount of missing data
line_figure("kl_16", "KL-divergence (nats, 16 nodes)",
            "KL-divergence from the true network", "fig1_kl_by_sparsity.png")
line_figure("cpdag_shd_16", "CPDAG SHD (16 nodes)",
            "Structural error (SHD) against the true network", "fig2_shd_by_sparsity.png",
            legend_kw=dict(loc="upper left", ncol=2), top=27)

# 3: extra links between missingness indicators and variables
fig, ax = plt.subplots(figsize=(7, 4.6))
width = 0.2
for i, (key, label) in enumerate(METHODS):
    m, se = mean_se(df, "r_edges_extra", key)
    ax.bar(np.arange(4) + (i - 1.5) * width, m, width, yerr=se, label=label,
           color=COLOUR[key], capsize=2)
ax.set_xticks(range(4))
ax.set_xticklabels([f"{s}%" for s in SPARS])
ax.set_xlabel("Percentage of data missing")
ax.set_ylabel("Extra R-links per dataset")
ax.set_title("Links between missingness indicators and variables\n(true value: 0)", fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 10.5)
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.text(0.99, 0.97, "Perfect and MICE: 0 extra links\nin every dataset",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
save(fig, "fig3_extra_r_links.png", SE_NOTE)

# 4: SHD against KL, one point per dataset
fig, ax = plt.subplots(figsize=(7.5, 5))
for key, label in METHODS:
    sub = df[df.method == key]
    ax.scatter(sub.cpdag_shd_16, sub.kl_16, s=24, alpha=0.55, color=COLOUR[key],
               marker=MARK[key], label=label)
ax.set_xlabel("CPDAG SHD (16 nodes)")
ax.set_ylabel("KL-divergence (16 nodes)")
ax.set_title("SHD and KL for every dataset", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(frameon=False)
save(fig, "fig4_shd_vs_kl.png",
     "Each point is one dataset (seed, sparsity level, method); points near the origin overlap.")

# 5: the spread across seeds
fig, ax = plt.subplots(figsize=(9.5, 4.8))
width = 0.17
for i, (key, label) in enumerate(METHODS):
    data = [df[(df.method == key) & (df.sparsity == s)].kl_16 for s in SPARS]
    pos = np.arange(4) + (i - 1.5) * width * 1.15
    bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                    flierprops=dict(markersize=3), medianprops=dict(color="black"))
    for box in bp["boxes"]:
        box.set_facecolor(COLOUR[key])
        box.set_alpha(0.75)
ax.set_xticks(range(4))
ax.set_xticklabels([f"{s}%" for s in SPARS])
ax.set_xlabel("Percentage of data missing")
ax.set_ylabel("KL-divergence (16 nodes)")
ax.set_title("Spread of KL across the 20 seeds", fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.legend(handles=[Patch(facecolor=COLOUR[k], alpha=0.75, label=l) for k, l in METHODS],
          frameon=False)
save(fig, "fig5_kl_spread.png",
     "Boxes show the middle 50% of seeds; the line is the median; whiskers reach 1.5 × the box height; dots are outliers.")

# 6: original 11-variable evaluation next to the 16-node evaluation
plain_path = ROOT / "plain_results.csv"
if plain_path.exists():
    plain = pd.read_csv(plain_path, dtype={"sparsity": str})
    complete_kl = plain[plain.method == "complete"].kl.mean()
    plain = plain[plain.sparsity != "none"].copy()
    plain["sparsity"] = plain.sparsity.astype(int)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    OFFSET = {"em": -0.9, "mice": -0.3, "soft": 0.3, "listwise": 0.9}
    for key, label in [("em", "EM"), ("mice", "MICE"), ("soft", "Soft-Impute"),
                       ("listwise", "Listwise deletion")]:
        m, se = mean_se(plain, "kl", key)
        a1.errorbar(np.array(SPARS) + OFFSET[key], m, yerr=se, label=label,
                    color=COLOUR[key], marker=MARK[key], capsize=3, lw=2, ms=6,
                    ls="--" if key == "listwise" else "-")
    a1.axhline(complete_kl, color=COLOUR["perfect"], ls=":", lw=1.5, label="Complete data")
    style_axes(a1, "KL-divergence", "Original 11 variables only")
    a1.legend(frameon=False)
    for key, label in METHODS:
        m, se = mean_se(df, "kl_16", key)
        a2.errorbar(SPARS, m, yerr=se, label=label, color=COLOUR[key], marker=MARK[key],
                    capsize=3, lw=2, ms=6)
    style_axes(a2, "", "16 nodes (with missingness indicators)")
    a2.legend(frameon=False)
    save(fig, "fig6_original_vs_16node.png",
         SE_NOTE[:-1] + ". Markers are offset slightly for readability; KL in nats. "
         "Listwise deletion was not run in the 16-node version.")
else:
    print("plain_results.csv not found in results/rq1, so figure 6 was skipped.")


