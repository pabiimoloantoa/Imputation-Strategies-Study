import pandas as pd

from load_pima import PROJECT_ROOT

old = pd.read_csv(PROJECT_ROOT / "results" / "rq2_old_softimpute" / "fold_scores_repeated.csv")
new = pd.read_csv(PROJECT_ROOT / "results" / "rq2" / "fold_scores_repeated.csv")

m = old.merge(new, on=["method", "seed", "fold"], suffixes=("_old", "_new"))
print("rows: old", len(old), "| new", len(new), "| matched", len(m), "(expect 200 each)")
m["change"] = m["auc_new"] - m["auc_old"]

print("\nNew AUC minus old AUC, same method and same fold:")
print(m.groupby("method")["change"].agg(["mean", "std", "min", "max"]).round(4))

print("\nMedian only, by seed:")
print(m[m.method == "Median"].groupby("seed")["change"].agg(["mean", "std"]).round(4))