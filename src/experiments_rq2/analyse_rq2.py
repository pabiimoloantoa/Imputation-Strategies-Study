import pandas as pd
from scipy import stats

from load_pima import PROJECT_ROOT

d = pd.read_csv(PROJECT_ROOT / "results" / "rq2" / "fold_scores_repeated.csv")

print(d.groupby("method")[["auc", "f1", "precision", "recall"]]
      .agg(["mean", "std"]).round(3))

for metric in ["auc", "f1"]:
    p = d.pivot_table(index=["seed", "fold"], columns="method", values=metric)
    print(f"\n{metric.upper()}: each method minus Median, on the same folds")
    for m in ["EM", "MICE", "SoftImpute"]:
        diff = p[m] - p["Median"]
        wins = (diff > 1e-12).sum()
        losses = (diff < -1e-12).sum()
        ties = len(diff) - wins - losses
        nz = diff[diff.abs() > 1e-12]
        pval = stats.wilcoxon(nz).pvalue if len(nz) else float("nan")
        per_seed = diff.groupby(level="seed").mean().round(4).tolist()
        print(f"{m:<11} mean diff {diff.mean():+.4f}  "
              f"wins/ties/losses {wins}/{ties}/{losses}  p={pval:.3f}  "
              f"per-seed means {per_seed}")