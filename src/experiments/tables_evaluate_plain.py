"""
make_tables.py -- results tables as images.

Reads results/rq1/plain_results.csv and writes, to results/rq1/figures/:
  seed<N>_table.png   one table per seed
  summary_table.png   one table with mean +/- SD across all seeds

Run from src/experiments:
    python make_tables.py
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "rq1" / "plain_results.csv"
OUT_DIR = PROJECT_ROOT / "results" / "rq1" / "figures"

SPARSITIES = ["10", "20", "30", "40"]
METHODS = [("listwise", "Listwise deletion"), ("em", "EM"),
           ("mice", "MICE"), ("soft", "Soft-Impute")]
IMPUTERS = ["em", "mice", "soft"]
GREEN = "#cfe8cf"


def draw_table(text, colours, title, footnote, figsize):
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
    table.scale(1, 2.2)
    ax.set_title(title, fontsize=12, pad=14)
    fig.text(0.5, 0.03, footnote, ha="center", fontsize=8)
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
            else:
                row_text.append(f"SHD {int(r.cpdag_shd)}\nKL {r.kl:.3f}")
            row_col.append("white")
        text.append(row_text)
        colours.append(row_col)

    # shade the lowest KL among the imputation methods in each column
    for j, sp in enumerate(SPARSITIES):
        scores = {k: cell[(k, sp)].kl for k in IMPUTERS if (k, sp) in cell}
        if scores:
            best = min(scores, key=scores.get)
            colours[[k for k, _ in METHODS].index(best)][j] = GREEN

    comp = cell.get(("complete", "none"))
    ref = (f"Complete data: SHD {int(comp.cpdag_shd)}, KL {comp.kl:.3f}"
           if comp is not None else "Complete data: not available")
    return draw_table(
        text, colours,
        title=f"Seed {seed}   |   {ref}",
        footnote=("SHD = CPDAG SHD, KL in nats. Lower is better for both. "
                  "Green = lowest KL among EM, MICE, Soft-Impute."),
        figsize=(8, 4.2),
    )


# ------------------------------------------------------- summary over seeds
def mean_sd(series, decimals):
    """'mean +/- SD' text; just the mean if there is only one value."""
    mean, sd, n = series.mean(), series.std(), series.count()
    if n == 0:
        return "-"
    if n == 1 or pd.isna(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def make_summary(df):
    n_seeds = df["seed"].nunique()

    text, colours = [], []
    mean_kl = {}
    for key, _ in METHODS:
        row_text, row_col = [], []
        for sp in SPARSITIES:
            sub = df[(df.method == key) & (df.sparsity == sp)]
            row_text.append(
                f"SHD {mean_sd(sub['cpdag_shd'], 1)}\nKL {mean_sd(sub['kl'], 3)}"
            )
            row_col.append("white")
            mean_kl[(key, sp)] = sub["kl"].mean()
        text.append(row_text)
        colours.append(row_col)

    # shade the lowest MEAN KL among the imputation methods in each column
    for j, sp in enumerate(SPARSITIES):
        scores = {k: mean_kl[(k, sp)] for k in IMPUTERS if pd.notna(mean_kl[(k, sp)])}
        if scores:
            best = min(scores, key=scores.get)
            colours[[k for k, _ in METHODS].index(best)][j] = GREEN

    comp = df[df.method == "complete"]
    ref = (f"Complete data: SHD {mean_sd(comp['cpdag_shd'], 1)}, "
           f"KL {mean_sd(comp['kl'], 3)}")
    return draw_table(
        text, colours,
        title=f"Summary over {n_seeds} seeds (mean ± SD)   |   {ref}",
        footnote=("SHD = CPDAG SHD, KL in nats. Lower is better for both. "
                  "Green = lowest mean KL among EM, MICE, Soft-Impute."),
        figsize=(10, 4.4),
    )


def main():
    df = pd.read_csv(RESULTS)
    df["sparsity"] = df["sparsity"].astype(str)
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