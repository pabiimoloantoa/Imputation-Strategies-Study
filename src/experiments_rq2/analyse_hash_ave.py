import pandas as pd
from scipy import stats

from load_pima import PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
METHODS = ["EM", "MICE", "SoftImpute"]
METRICS = ["auc", "f1", "precision", "recall"]

d = pd.concat([pd.read_csv(RES / f"fold_scores_repeated_hash{h}.csv").assign(hash=h) for h in range(5)])

print("Mean AUC of each method under each hash seed (the range shows the run-to-run noise):")
print(d.groupby(["hash", "method"])["auc"].mean().unstack().round(4))

# average each method's score over the five hash seeds, fold by fold
avg = d.groupby(["method", "seed", "fold"])[METRICS].mean().reset_index()

print("\nMean +/- std over the 50 folds, after averaging each fold over the 5 hash seeds:")
s = avg.groupby("method")[METRICS].agg(["mean", "std"]).round(3)
print(s)

for metric in METRICS:
    p = avg.pivot_table(index=["seed", "fold"], columns="method", values=metric)
    print(f"\n{metric.upper()}: method minus Median on the same folds (hash-averaged)")
    for m in METHODS:
        diff = p[m] - p["Median"]
        wins, losses = int((diff > 1e-12).sum()), int((diff < -1e-12).sum())
        nz = diff[diff.abs() > 1e-12]
        pv = stats.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
        per_seed = diff.groupby(level="seed").mean().round(4).tolist()
        print(f"  {m:<11} mean diff {diff.mean():+.4f}  wins/ties/losses {wins}/{len(diff) - wins - losses}/{losses}"
              f"  p={pv:.3f}  per-split means {per_seed}")