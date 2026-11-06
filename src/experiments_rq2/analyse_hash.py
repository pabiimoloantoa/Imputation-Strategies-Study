import re
from pathlib import Path

import pandas as pd
from scipy import stats

from load_pima import PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
METHODS = ["EM", "MICE", "SoftImpute"]


def tag_of(f):
    return f.stem.replace("fold_scores_repeated_hash", "")


files = sorted(RES.glob("fold_scores_repeated_hash*.csv"),
               key=lambda f: (0, int(tag_of(f))) if tag_of(f).isdigit() else (1, 0))
print("files found:", [f.name for f in files], "\n")

rows, median_auc = [], {}
for f in files:
    tag = tag_of(f)
    d = pd.read_csv(f)
    p = d.pivot_table(index=["seed", "fold"], columns="method", values="auc")
    median_auc[tag] = p["Median"]
    row = {"hash": tag, "rows": len(d), "Median": round(p["Median"].mean(), 4)}
    for m in METHODS:
        diff = p[m] - p["Median"]
        nz = diff[diff.abs() > 1e-12]
        pv = stats.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
        row[f"{m} diff"] = round(diff.mean(), 4)
        row[f"{m} p"] = round(pv, 3)
    rows.append(row)

print("Each method minus Median (AUC, 50 paired folds), under each hash seed (expect 200 rows per file):")
print(pd.DataFrame(rows).to_string(index=False))

if len(median_auc) > 1:
    all_med = pd.DataFrame(median_auc)
    spread = all_med.max(axis=1) - all_med.min(axis=1)
    print("\nMedian fill, per fold, across the hash seeds:")
    print("  folds where the AUC is not identical in every run:", int((spread > 1e-9).sum()), "of", len(spread))
    print("  average gap between the highest and lowest run:", round(float(spread.mean()), 4))
    print("  largest gap in any fold:", round(float(spread.max()), 4))