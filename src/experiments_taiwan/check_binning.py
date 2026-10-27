from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from inject_mar import bin_states

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "data" / "taiwan" / "taiwan_clean.csv")
y = df["Outcome"]

print("Default rate by PAY_0 value:")
print(df.groupby("PAY_0")["Outcome"].agg(["mean", "size"]).round(3))

print("\nAUC of each variable used on its own (0.5 = useless):")
print(f"{'variable':<10s} {'raw':>6s} {'current bins':>13s} {'proposed bins':>14s}")
for c in ["PAY_0", "PAY_2", "PAY_3", "LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "AGE"]:
    raw = roc_auc_score(y, df[c])
    raw = max(raw, 1 - raw)

    def binned_auc(states):
        rate = pd.Series(y.to_numpy()).groupby(states).transform("mean")
        return roc_auc_score(y, rate)

    cur = binned_auc(bin_states(df[c].to_numpy())[0])
    prop = ""
    if c.startswith("PAY_") and not c.startswith("PAY_AMT"):
        prop = f"{binned_auc(np.searchsorted([0.5, 1.5], df[c].to_numpy(), side='right')):.3f}"
    print(f"{c:<10s} {raw:6.3f} {cur:13.3f} {prop:>14s}")