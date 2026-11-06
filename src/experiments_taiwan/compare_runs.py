from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results"
old = pd.read_csv(RES / "taiwan_oldbins" / "scores.csv")
new = pd.read_csv(RES / "taiwan" / "scores.csv")

m = old.merge(new, on=["seed", "level", "fold", "method"], suffixes=("_old", "_new"))
print("rows: old", len(old), "| new", len(new), "| matched", len(m), "(expect 850 each)")
m["change"] = m["auc_new"] - m["auc_old"]

print("\nNew AUC minus old AUC, same method, level and fold:")
print(m.groupby(["method", "level"])["change"].agg(["mean", "std", "min", "max"]).round(4))

c = m[m.method == "Complete"]
print("\nComplete data (identical inputs in both runs): folds where the AUC differs:",
      int((c["change"].abs() > 1e-9).sum()), "of", len(c))