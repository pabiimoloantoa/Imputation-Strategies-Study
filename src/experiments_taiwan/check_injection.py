import json
from pathlib import Path

import numpy as np
import pandas as pd

from inject_mar import bin_states, SEEDS, SPARSITY_LEVELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "taiwan"

clean = pd.read_csv(DATA_DIR / "taiwan_clean.csv")
problems = []

# --- Part 1: file checks for all 20 files ---
for seed in SEEDS:
    audit = json.load(open(DATA_DIR / "audit_logs" / f"audit_seed{seed}.json"))
    vm, vo = audit["vm_vars"], audit["vo_vars"]
    for level in SPARSITY_LEVELS:
        pct = int(round(level * 100))
        m = pd.read_csv(DATA_DIR / "missing" / f"missing_{pct}pct_seed{seed}.csv")
        tag = f"seed {seed}, {pct}%"
        if m.shape != clean.shape:
            problems.append(f"{tag}: wrong shape {m.shape}")
        if m["Outcome"].isna().any() or m[vo].isna().any().any():
            problems.append(f"{tag}: Outcome or a Vo column has missing values")
        for v in vm:
            seen = m[v].notna()
            if not np.allclose(m.loc[seen, v], clean.loc[seen, v]):
                problems.append(f"{tag}: {v} observed cells were changed")
            rate = m[v].isna().mean()
            if abs(rate - level) > 0.01:
                problems.append(f"{tag}: {v} missing rate {rate:.3f} is off target")

print("Part 1 (all 20 files):", "no problems found" if not problems else "")
for p in problems:
    print("   ", p)

# --- Part 2: is the mechanism MAR? (20% files) ---
print("\nPart 2: within each group of the R-parent (20% level)")
for seed in SEEDS:
    audit = json.load(open(DATA_DIR / "audit_logs" / f"audit_seed{seed}.json"))
    m = pd.read_csv(DATA_DIR / "missing" / f"missing_20pct_seed{seed}.csv")
    for v in audit["vm_vars"]:
        p = audit["r_parents"][v]
        state, _ = bin_states(clean[p].to_numpy())
        miss = m[v].isna().to_numpy()
        rates, gaps = [], []
        for s in np.unique(state):
            in_s = state == s
            rates.append(round(float(miss[in_s].mean()), 3))
            gaps.append(round(float(clean.loc[in_s & miss, "Outcome"].mean()
                                    - clean.loc[in_s & ~miss, "Outcome"].mean()), 3))
        print(f"seed {seed} {v:<10s} by {p:<10s} missing rate per group {rates}  default gap {gaps}")