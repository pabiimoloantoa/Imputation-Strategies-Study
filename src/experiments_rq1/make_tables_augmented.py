"""
make_tables_augmented.py -- tables for the 16-vs-16 evaluation, as images.

Reads results/rq1/augmented_results.csv and writes, to results/rq1/tables_16v16/:
  seed<N>_table.png   one table per seed
  summary_table.png   one table with mean +/- SD across all seeds

Run from src/experiments:
    python make_tables_augmented.py
"""

import textwrap
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "rq1" / "augmented_results.csv"
OUT_DIR = PROJECT_ROOT / "results" / "rq1" / "tables_16v16"

SPARSITIES = [10, 20, 30, 40]
METHODS = [("perfect", "Perfect (baseline)"), ("em", "EM"),
           ("mice", "MICE"), ("soft", "Soft-Impute")]
TRACE = "#f6d0d0"          # shading: at least one extra R-link (trace of the missingness)


def draw_table(text, colours, title, footnote, figsize, row_scale, wrap):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=text,
        cellColours=colours,
        rowLabels=[label for _, label in METHODS],
        colLabels=[f"{sp}% missing" for sp in SPARSITIES],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, row_scale)
    ax.set_title(title, fontsize=12, pad=14)
    fig.text(0.5, 0.02, textwrap.fill(footnote, wrap), ha="center", fontsize=8)
    return fig


# ---------------------------------------------------------------- one seed
def make_table(df_seed, seed):
    cell = {(r.method, r.sparsity): r for r in df_seed.itertuples()}

    text, colours = [], []
    for key, _ in METHODS:
        row_text, row_col = [], []
        for sp in SPARSITIES:
            r = cell.get((key, sp))
            if r is None:
                row_text.append("-")
                row_col.append("white")
            else:
                row_text.append(
                    f"SHD {int(r.cpdag_shd_16)}\n"
                    f"KL {r.kl_16:.3f}\n"
                    f"extra R-links {int(r.r_edges_extra)}"
                )
                row_col.append(TRACE if r.r_edges_extra > 0 else "white")
        text.append(row_text)
        colours.append(row_col)

    fb = df_seed["fallback_used"].iloc[0]
    return draw_table(
        text, colours,
        title=f"Seed {seed}  |  16-node evaluation  |  fallback seed: {'yes' if fb else 'no'}",
        footnote=("SHD = CPDAG SHD on all 16 nodes. "
                  "KL over 16 nodes. Lower is better for SHD and KL. "
                  "Shaded = at least one extra R-link (a trace of the missingness)."),
        figsize=(9.5, 6),
        row_scale=4.2,
        wrap=110,
    )
# ------------------------------------------------------- summary over seeds
def mean_sd(series, decimals):
    mean, sd = series.mean(), series.std()
    if pd.isna(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def make_summary(df):
    n_seeds = df["seed"].nunique()

    text, colours = [], []
    for key, _ in METHODS:
        row_text, row_col = [], []
        for sp in SPARSITIES:
            sub = df[(df.method == key) & (df.sparsity == sp)]
            row_text.append(
                f"SHD {mean_sd(sub['cpdag_shd_16'], 1)}\n"
                f"KL {mean_sd(sub['kl_16'], 3)}\n"
                f"extra R-links {sub['r_edges_extra'].mean():.1f} "
                f"(own {sub['r_own_links'].mean():.1f})"
            )
            row_col.append(TRACE if sub["r_edges_extra"].mean() > 0 else "white")
        text.append(row_text)
        colours.append(row_col)

    return draw_table(
        text, colours,
        title=f"Summary over {n_seeds} seeds, 16-node evaluation (mean ± SD)",
        footnote=("SHD = CPDAG SHD on all 16 nodes. "
                  "KL over 16 nodes. Lower is better for SHD and KL. "
                  "extra R-links = average number per dataset (own = linked to the R-node's own variable). "
                  "Shaded = extra R-links present."),
        figsize=(11.5, 7),
        row_scale=4.2,
        wrap=130,
    )


def main():
    df = pd.read_csv(RESULTS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in sorted(df["seed"].unique()):
        fig = make_table(df[df["seed"] == seed], int(seed))
        path = OUT_DIR / f"seed{int(seed)}_table.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved", path)

    fig = make_summary(df)
    path = OUT_DIR / "summary_table.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    main()









