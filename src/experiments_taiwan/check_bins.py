import json
import warnings

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_taiwan import IMPUTERS, PAY_EDGES, N_BINS, DATA_DIR, SEEDS, LEVELS
from bn_classifier import Discretizer

warnings.filterwarnings("ignore")

clean = pd.read_csv(DATA_DIR / "taiwan_clean.csv")
X_true, y = clean.drop(columns="Outcome"), clean["Outcome"]

rows = []
for seed in SEEDS:
    vm = json.load(open(DATA_DIR / "audit_logs" / f"audit_seed{seed}.json"))["vm_vars"]
    tr, te = next(iter(StratifiedKFold(10, shuffle=True, random_state=seed).split(X_true, y)))
    ref = Discretizer(N_BINS, fixed_edges=PAY_EDGES).fit(X_true.iloc[tr]).n_states_   # groups on complete data
    for level in LEVELS:
        miss = pd.read_csv(DATA_DIR / "missing" / f"missing_{level}pct_seed{seed}.csv").drop(columns="Outcome")
        row = {"seed": seed, "level": level}
        for name, make in IMPUTERS.items():
            filled = make().fit_transform(miss.iloc[tr])
            n = Discretizer(N_BINS, fixed_edges=PAY_EDGES).fit(filled).n_states_
            lost = [c for c in vm if n[c] < ref[c]]
            row[name] = ", ".join(f"{c}:{ref[c]}->{n[c]}" for c in lost) or "-"
        rows.append(row)

print("Columns with injected missingness that lose groups when binned after imputing (fold 1; '-' = none):")
print(pd.DataFrame(rows).to_string(index=False))