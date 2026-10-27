import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "taiwan"
MISSING_DIR = DATA_DIR / "missing"
AUDIT_DIR = DATA_DIR / "audit_logs"

SEEDS = [42, 43, 44, 45, 46]
SPARSITY_LEVELS = [0.10, 0.20, 0.30, 0.40]
N_VM = 5                    # as in RQ1
N_BINS = 3                  # R-parent is cut into up to 3 ordered states
MAX_PROBABILITY = 0.95      # same safety ceiling as RQ1


def bin_states(x, n_bins=N_BINS):
    """Same rule as the Discretizer: quantile edges, ties merged, ordered low to high."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs))
    return np.searchsorted(edges, x, side="right"), edges


def derive_cpt(marginal, target):
    """P(missing | parent state) = k * multiplier, scaled so the average equals target (as in RQ1)."""
    multipliers = np.array([1 + 0.5 * i for i in range(len(marginal))])
    k = target / float((marginal * multipliers).sum())
    return np.minimum(k * multipliers, MAX_PROBABILITY)


def main():
    df = pd.read_csv(DATA_DIR / "taiwan_clean.csv")
    predictors = [c for c in df.columns if c != "Outcome"]    # Outcome is never Vm or R-parent

    MISSING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        vm = sorted(str(v) for v in rng.choice(predictors, size=N_VM, replace=False))
        vo = [c for c in predictors if c not in vm]

        # a different R-parent for every Vm column
        vo_order = [str(v) for v in rng.permutation(vo)]
        r_parents = {v: vo_order[i] for i, v in enumerate(vm)}

        parent_state, parent_edges, parent_marginal = {}, {}, {}
        for p in set(r_parents.values()):
            states, edges = bin_states(df[p].to_numpy())
            parent_state[p] = states
            parent_edges[p] = [float(e) for e in edges]
            parent_marginal[p] = np.bincount(states, minlength=len(edges) + 1) / len(df)

        audit = {
            "seed": seed, "vm_vars": vm, "vo_vars": vo, "r_parents": r_parents,
            "corr_vm_with_r_parent": {v: round(float(df[v].corr(df[r_parents[v]])), 3) for v in vm},
            "r_parent_bin_edges": parent_edges,
            "cpts": {}, "expected_rates": {}, "achieved_rates": {},
        }

        for level in SPARSITY_LEVELS:
            pct = int(round(level * 100))
            out = df.copy()
            audit["cpts"][f"{pct}pct"], audit["expected_rates"][f"{pct}pct"] = {}, {}
            audit["achieved_rates"][f"{pct}pct"] = {}
            for v in vm:
                p = r_parents[v]
                cpt = derive_cpt(parent_marginal[p], level)             # P(missing | state)
                row_prob = cpt[parent_state[p]]                         # each customer's chance
                is_missing = rng.random(len(df)) < row_prob             # independent draw per level
                out.loc[is_missing, v] = np.nan

                audit["cpts"][f"{pct}pct"][v] = [round(float(c), 4) for c in cpt]
                audit["expected_rates"][f"{pct}pct"][v] = round(float((parent_marginal[p] * cpt).sum()), 4)
                audit["achieved_rates"][f"{pct}pct"][v] = round(float(is_missing.mean()), 4)

            out.to_csv(MISSING_DIR / f"missing_{pct}pct_seed{seed}.csv", index=False)

        with open(AUDIT_DIR / f"audit_seed{seed}.json", "w") as f:
            json.dump(audit, f, indent=2)

        print(f"\nseed {seed}: Vm = {vm}")
        for v in vm:
            rates = [audit["achieved_rates"][f"{int(round(l * 100))}pct"][v] for l in SPARSITY_LEVELS]
            print(f"   {v:<10s} <- {r_parents[v]:<10s} (corr {audit['corr_vm_with_r_parent'][v]:+.2f})  "
                  f"achieved {rates}")


if __name__ == "__main__":
    main()