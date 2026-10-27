import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from soft_impute import SoftImputer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "taiwan"
SEED, LEVEL = 42, 10

clean = pd.read_csv(DATA_DIR / "taiwan_clean.csv")
X_true, y = clean.drop(columns="Outcome"), clean["Outcome"]
spread = X_true.std()
vm = json.load(open(DATA_DIR / "audit_logs" / f"audit_seed{SEED}.json"))["vm_vars"]
miss = pd.read_csv(DATA_DIR / "missing" / f"missing_{LEVEL}pct_seed{SEED}.csv").drop(columns="Outcome")
tr, te = next(iter(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X_true, y)))

imp = SoftImputer()
tr_f = imp.fit_transform(miss.iloc[tr])
te_f = imp.transform(miss.iloc[te])
print("directions kept after shrinkage:", imp.Vt_.shape[0], "of", X_true.shape[1])
print("shrinkage amount:", round(imp.lambda_, 2))


def scaled_error(filled, idx):
    hidden = miss.iloc[idx].isna()
    truth = X_true.iloc[idx]
    return {v: round(float(np.abs(filled[v].to_numpy()[hidden[v].to_numpy()]
                                  - truth[v].to_numpy()[hidden[v].to_numpy()]).mean() / spread[v]), 3)
            for v in vm}


med_f = miss.iloc[te].fillna(miss.iloc[tr].median())
print("\nscaled error on erased cells:")
print("  Soft-Impute, training customers:", scaled_error(tr_f, tr))
print("  Soft-Impute, test customers:    ", scaled_error(te_f, te))
print("  Median, test customers:         ", scaled_error(med_f, te))