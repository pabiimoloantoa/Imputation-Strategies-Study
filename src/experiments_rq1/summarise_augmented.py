"""
summarise_augmented.py -- quick summary of results/rq1/augmented_results.csv.

Run from src/experiments:
    python summarise_augmented.py
"""

from pathlib import Path

import pandas as pd

path = Path(__file__).resolve().parents[2] / "results" / "rq1" / "augmented_results.csv"
d = pd.read_csv(path)

METHODS = ["perfect", "em", "mice", "soft"]
SPARSITIES = [10, 20, 30, 40]
SEEDS = range(20)

# ---------------------------------------------------------- completeness check
print(f"Rows: {len(d)}  |  seeds: {d.seed.nunique()}  |  expected rows: 320")
have = set(zip(d.seed, d.sparsity, d.method))
missing = [(s, p, m) for s in SEEDS for p in SPARSITIES for m in METHODS
           if (s, p, m) not in have]
print("Missing datasets (seed, sparsity, method):", missing if missing else "none")
print("Duplicate rows:", int(d.duplicated(["seed", "sparsity", "method"]).sum()))
print()


# ---------------------------------------------------------- mean +/- SD tables
def mean_sd(col, decimals):
    g = d.groupby(["method", "sparsity"])[col]
    text = g.mean().round(decimals).astype(str) + " ± " + g.std().round(decimals).astype(str)
    return text.unstack("sparsity").loc[METHODS, SPARSITIES]


for col, decimals, title in [
    ("kl_16", 3, "KL over the 16 nodes (mean ± SD across seeds)"),
    ("cpdag_shd_16", 1, "CPDAG SHD, 16 nodes"),
    ("cpdag_shd_11", 1, "CPDAG SHD, the 11 original variables only"),
    ("r_edges_found", 2, "True R-edges found (out of 5)"),
    ("r_edges_extra", 2, "Extra R-links"),
    ("r_own_links", 2, "R-nodes linked to their own variable (out of 5)"),
]:
    print(title)
    print(mean_sd(col, decimals).to_string())
    print()

# ---------------------------------------------------------- footprint overview
print("Share of datasets with NO extra R-links at all:")
print(d.groupby("method").r_edges_extra.apply(lambda s: round(float((s == 0).mean()), 2)).to_string())
print()

# ---------------------------------------------------------- seed-by-seed comparisons
w = d.pivot_table(index=["seed", "sparsity"], columns="method", values="kl_16")
for other in ["em", "soft"]:
    share = {p: round(float((w.xs(p, level="sparsity")["mice"] < w.xs(p, level="sparsity")[other]).mean()), 2)
             for p in SPARSITIES}
    print(f"Share of seeds where MICE has lower KL than {other}: {share}")




