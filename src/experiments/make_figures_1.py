"""
make_figures_1.py  -  three extra report figures for the 16-vs-16 evaluation.

  fig_gap_in_wobble_units.png   gap from the perfect baseline at 20%, in units of hash-seed wobble
  fig_paired_kl_per_seed.png    per-seed KL lines MICE -> EM -> Soft-Impute, one panel per sparsity
  fig_r_edge_recovery.png       true R-edges found (out of 5) by method and sparsity

Run from anywhere:   python src/experiments/make_figures_1.py   (paths are resolved from the script location)
Reads  results/rq1/augmented_results.csv and results/rq1/noise_floor.csv
Writes results/rq1/figures_report/   (same folder as make_figures.py)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG - if your CSV headers are named differently, change them here only.
# ----------------------------------------------------------------------------
# Script lives in <project>/src/experiments/, results live in <project>/results/ (outside src).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "rq1"
MAIN_CSV = RESULTS / "augmented_results.csv"
NOISE_CSV = RESULTS / "noise_floor.csv"
OUT_DIR = RESULTS / "figures_report"

COL_SEED = "seed"
COL_SPARSITY = "sparsity"        # 10/20/30/40 (or "10pct" - digits are extracted)
COL_METHOD = "method"            # perfect / em / mice / soft
COL_KL = "kl_16"                 # KL on the 16-node network (nats)
COL_SHD = "cpdag_shd_16"               # CPDAG SHD on the 16-node graph
COL_R_FOUND = "r_edges_found"    # true R-edges found, out of 5
COL_HASH = "hash_seed"           # noise_floor.csv only

NOISE_SPARSITY = 20              # noise floor was run at 20%
N_R_EDGES = 5
# ----------------------------------------------------------------------------

METHODS = ["perfect", "mice", "em", "soft"]
LABELS = {"perfect": "Perfect (baseline)", "mice": "MICE", "em": "EM", "soft": "Soft-Impute"}
COLORS = {"perfect": "#555555", "mice": "#1b9e77", "em": "#d95f02", "soft": "#7570b3"}   # same palette as make_figures.py


def load(path, needed, defaults=None):
    if not path.exists():
        sys.exit(f"Cannot find {path}. The script expects to sit in <project>/src/experiments/.")
    df = pd.read_csv(path)
    for col, val in (defaults or {}).items():
        if col not in df.columns:      # e.g. noise_floor.csv only holds the perfect baseline at 20%
            df[col] = val
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(
            f"{path.name} has no column(s) {missing}.\n"
            f"Columns it does have: {list(df.columns)}\n"
            f"Edit the CONFIG block at the top of this script to match."
        )
    df[COL_METHOD] = df[COL_METHOD].astype(str).str.strip().str.lower()
    df[COL_SPARSITY] = pd.to_numeric(
        df[COL_SPARSITY].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    return df


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    })


# ----------------------------------------------------------------------------
# Figure 1: gap from baseline in units of hash-seed wobble
# ----------------------------------------------------------------------------
def wobble(noise, col):
    """Typical spread caused by Hill Climbing's hash-seed alone: SD across the
    10 hash seeds within each dataset, averaged over the 20 datasets."""
    base = noise[(noise[COL_METHOD] == "perfect") & (noise[COL_SPARSITY] == NOISE_SPARSITY)]
    return base.groupby(COL_SEED)[col].std(ddof=1).mean()


def fig_gap(main, noise):
    at = main[main[COL_SPARSITY] == NOISE_SPARSITY]
    means = at.groupby(COL_METHOD)[[COL_KL, COL_SHD]].mean()
    base = means.loc["perfect"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, title in [(axes[0], COL_KL, "KL (16 nodes)"),
                           (axes[1], COL_SHD, "CPDAG SHD (16 nodes)")]:
        w = wobble(noise, col)
        ms = ["mice", "em", "soft"]
        gaps = [(means.loc[m, col] - base[col]) / w for m in ms]
        ax.axhspan(-1, 1, color="#9ca3af", alpha=0.25, zorder=0, label="within 1x wobble")
        bars = ax.bar([LABELS[m] for m in ms], gaps, color=[COLORS[m] for m in ms], width=0.6, zorder=2)
        ax.axhline(0, color="black", lw=0.8, zorder=3)
        for b, g in zip(bars, gaps):
            ax.annotate(f"{g:+.1f}x", (b.get_x() + b.get_width() / 2, g),
                        xytext=(0, 4 if g >= 0 else -4), textcoords="offset points",
                        ha="center", va="bottom" if g >= 0 else "top", fontsize=10)
        lo, hi = min(min(gaps), -1), max(max(gaps), 1)
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}\n(wobble = {w:.3f})" if col == COL_KL else f"{title}\n(wobble = {w:.2f})")
        ax.set_ylabel("Gap from perfect baseline\n(units of hash-seed wobble)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"How far each method sits from the baseline, at {NOISE_SPARSITY}% missingness", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_gap_in_wobble_units.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: paired KL per seed
# ----------------------------------------------------------------------------
def fig_paired(main):
    levels = sorted(main[COL_SPARSITY].dropna().unique())
    ms = ["mice", "em", "soft"]
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lv in zip(axes, levels):
        wide = (main[(main[COL_SPARSITY] == lv) & (main[COL_METHOD].isin(ms))]
                .pivot_table(index=COL_SEED, columns=COL_METHOD, values=COL_KL)[ms])
        x = np.arange(len(ms))
        for _, row in wide.iterrows():
            ax.plot(x, row.values, color="#9ca3af", alpha=0.5, lw=0.8, zorder=1)
        ax.plot(x, wide.mean().values, color="black", lw=2.2, marker="o", zorder=3, label="mean")
        for xi, m in zip(x, ms):
            ax.scatter(np.full(len(wide), xi), wide[m], color=COLORS[m], s=14, zorder=2)
        n_win = int(((wide["mice"] < wide["em"]) & (wide["mice"] < wide["soft"])).sum())
        ax.set_title(f"{int(lv)}% missing\nMICE lowest in {n_win}/{len(wide)} seeds")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in ms], rotation=20)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("KL (16 nodes, nats)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Each grey line is one seed: MICE -> EM -> Soft-Impute", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_paired_kl_per_seed.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: true R-edge recovery
# ----------------------------------------------------------------------------
def fig_r_edges(main):
    levels = sorted(main[COL_SPARSITY].dropna().unique())
    width = 0.2
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(METHODS):
        sub = main[main[COL_METHOD] == m].groupby(COL_SPARSITY)[COL_R_FOUND]
        mean = sub.mean().reindex(levels)
        se = (sub.std(ddof=1) / np.sqrt(sub.count())).reindex(levels)   # standard error, as in make_figures.py
        ax.bar(x + (i - 1.5) * width, mean, width, yerr=se, capsize=2,
               color=COLORS[m], label=LABELS[m], error_kw={"lw": 0.8, "alpha": 0.7})
    ax.axhline(N_R_EDGES, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(l)}%" for l in levels])
    ax.set_xlabel("Missingness level")
    ax.set_ylabel("True R-edges found (out of 5)")
    ax.set_ylim(0, N_R_EDGES + 0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("True missingness-mechanism edges recovered after imputation", pad=28)
    fig.text(0.5, -0.02, "Mean over the 20 seeds; error bars show ±1 standard error; dashed line = all 5 true R-edges.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_r_edge_recovery.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    base_cols = [COL_SEED, COL_SPARSITY, COL_METHOD, COL_KL, COL_SHD, COL_R_FOUND]
    main_df = load(MAIN_CSV, base_cols)
    noise_df = load(NOISE_CSV, [COL_SEED, COL_SPARSITY, COL_METHOD, COL_KL, COL_SHD, COL_HASH],
                    defaults={COL_METHOD: "perfect", COL_SPARSITY: NOISE_SPARSITY})

    fig_gap(main_df, noise_df)
    fig_paired(main_df)
    fig_r_edges(main_df)
    print(f"Saved 3 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()