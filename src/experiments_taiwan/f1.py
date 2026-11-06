from pathlib import Path

import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parents[2] / "results" / "taiwan"
sc = pd.read_csv(RES / "scores.csv")
METHODS = ["EM", "MICE", "SoftImpute"]
LEVELS = [10, 20, 30, 40]

for metric in ["f1", "precision", "recall"]:
    s = sc.groupby(["level", "method"])[metric].agg(["mean", "std"])
    print(f"\n{metric.upper()}: mean +/- std over the 50 folds (level 0 = complete data)")
    print((s["mean"].round(3).astype(str) + " +/- " + s["std"].round(3).astype(str)).unstack())

print("\nSECONDARY paired comparison: method minus Median on the same folds (p from Wilcoxon)")
for metric in ["f1", "precision", "recall"]:
    for level in LEVELS:
        p = sc[sc.level == level].pivot_table(index=["seed", "fold"], columns="method", values=metric)
        parts = []
        for m in METHODS:
            d = p[m] - p["Median"]
            nz = d[d.abs() > 1e-12]
            pv = stats.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
            parts.append(f"{m} {d.mean():+.4f} (p={pv:.3f})")
        print(f"{metric:<9s} {level:>2}%   " + "   ".join(parts))