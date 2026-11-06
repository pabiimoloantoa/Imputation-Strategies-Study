import pandas as pd

from load_pima import PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
d = pd.concat([pd.read_csv(RES / f"fold_scores_repeated_hash{h}.csv") for h in range(5)])
avg = d.groupby(["method", "seed", "fold"])[["auc", "f1", "precision", "recall"]].mean().reset_index()
avg.to_csv(RES / "fold_scores_hash_averaged.csv", index=False)

print(avg.shape)    # expect (200, 7)
print(avg.groupby("method")[["auc", "f1", "precision", "recall"]].mean().round(3))