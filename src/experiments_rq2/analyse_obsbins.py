import pandas as pd
from scipy import stats

from load_pima import PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
METHODS = ["EM", "MICE", "SoftImpute"]
METRICS = ["auc", "f1", "precision", "recall"]


def load(prefix):
    return pd.concat([pd.read_csv(RES / f"{prefix}{h}.csv").assign(hash=h) for h in range(5)])


new = load("fold_scores_repeated_obsbins_hash")
old = load("fold_scores_repeated_hash")
print("rows per file, new binning (expect 200 each):", new.groupby("hash").size().tolist())

print("\nMean AUC under each hash seed, NEW binning:")
print(new.groupby(["hash", "method"])["auc"].mean().unstack().round(4))


def averaged(d):
    return d.groupby(["method", "seed", "fold"])[METRICS].mean().reset_index()


new_avg, old_avg = averaged(new), averaged(old)
new_avg.to_csv(RES / "fold_scores_hash_averaged_obsbins.csv", index=False)

print("\nHash-averaged mean and std over 50 folds, NEW binning:")
print(new_avg.groupby("method")[METRICS].agg(["mean", "std"]).round(3))


def paired(avg, metric):
    p = avg.pivot_table(index=["seed", "fold"], columns="method", values=metric)
    out = {}
    for m in METHODS:
        d = p[m] - p["Median"]
        nz = d[d.abs() > 1e-12]
        pv = stats.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
        out[m] = (d.mean(), pv, int((d > 1e-12).sum()), int((d < -1e-12).sum()))
    return out


for metric in METRICS:
    o, n = paired(old_avg, metric), paired(new_avg, metric)
    print(f"\n{metric.upper()}: method minus Median on the same folds (hash-averaged)")
    for m in METHODS:
        print(f"  {m:<11} old binning {o[m][0]:+.4f} (p={o[m][1]:.3f})   "
              f"new binning {n[m][0]:+.4f} (p={n[m][1]:.3f}, better/worse {n[m][2]}/{n[m][3]})")

print("\nNEW binning: AUC of each method minus Median under each hash seed")
rows = []
for h in range(5):
    p = new[new.hash == h].pivot_table(index=["seed", "fold"], columns="method", values="auc")
    rows.append({"hash": h, **{m: round((p[m] - p["Median"]).mean(), 4) for m in METHODS}})
print(pd.DataFrame(rows).to_string(index=False))
