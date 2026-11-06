from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parents[2] / "results" / "taiwan"
sc = pd.read_csv(RES / "scores.csv")
ac = pd.read_csv(RES / "accuracy.csv")

# one accuracy number per run: the average over that seed's damaged columns
acc_run = ac.groupby(["seed", "level", "fold", "method"])["scaled_error"].mean().reset_index()

METHODS = ["EM", "MICE", "SoftImpute"]
LEVELS = sorted(int(l) for l in sc["level"].unique() if l > 0)

print("Rows per seed in scores.csv (expect 170):", sc.groupby("seed").size().tolist())
print("Runs per method and level (expect 50):")
print(sc[sc.level > 0].groupby(["level", "method"]).size().unstack())

print("\nAUC, mean and standard deviation over the 50 folds:")
s = sc.groupby(["level", "method"])["auc"].agg(["mean", "std"])
print((s["mean"].round(3).astype(str) + " +/- " + s["std"].round(3).astype(str)).unstack())


def paired(df, value, higher_is_better, title):
    for level in LEVELS:
        p = df[df.level == level].pivot_table(index=["seed", "fold"], columns="method", values=value)
        print(f"\n{title}, level {level}%: method minus Median on the same folds")
        for m in METHODS:
            d = p[m] - p["Median"]
            better = int((d > 1e-12).sum() if higher_is_better else (d < -1e-12).sum())
            worse = int((d < -1e-12).sum() if higher_is_better else (d > 1e-12).sum())
            ties = len(d) - better - worse
            nz = d[d.abs() > 1e-12]
            pv = stats.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
            per_seed = d.groupby(level="seed").mean().round(4).tolist()
            print(f"  {m:<11} mean diff {d.mean():+.4f}  better/tied/worse {better}/{ties}/{worse}  "
                  f"p={pv:.3f}  per-seed means {per_seed}")


paired(sc, "auc", True, "AUC (higher is better)")
paired(acc_run, "scaled_error", False, "Scaled error (lower is better)")

print("\nShare of the damage each method repairs (AUC): (method - Median) / (Complete - Median)")
ceiling = sc[sc.level == 0]["auc"].mean()
rows = {}
for level in LEVELS:
    m_auc = sc[sc.level == level].groupby("method")["auc"].mean()
    gap = ceiling - m_auc["Median"]
    rows[f"{level}%"] = {m: round((m_auc[m] - m_auc["Median"]) / gap, 2) if gap > 1e-6 else np.nan
                         for m in METHODS}
    rows[f"{level}%"]["Complete - Median"] = round(gap, 4)
print(pd.DataFrame(rows).T)

print("\nExploratory: scaled error by damaged column (all levels and seeds, lower is better):")
tab = ac.groupby(["column", "method"])["scaled_error"].mean().unstack().round(3)
tab["seeds_damaged"] = ac.groupby("column")["seed"].nunique()
print(tab)