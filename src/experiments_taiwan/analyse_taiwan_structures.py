import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
AUDIT = ROOT / "data" / "taiwan" / "audit_logs"

edges = pd.read_csv(RES / "edges.csv")
sc = pd.read_csv(RES / "scores.csv")


def blanket(run_edges):
    """Variables directly tied to Outcome: its parents, its children, and the children's other parents."""
    parents = {}
    for p, c in zip(run_edges["parent"], run_edges["child"]):
        parents.setdefault(c, set()).add(p)
    kids = [c for c, ps in parents.items() if "Outcome" in ps]
    b = set(parents.get("Outcome", set())) | set(kids)
    for k in kids:
        b |= parents[k]
    b.discard("Outcome")
    return b


complete = edges[edges.method == "Complete"]

# which variables the network relies on when nothing is damaged
counts = Counter()
n_runs = 0
for _, g in complete.groupby(["seed", "fold"]):
    counts.update(blanket(g))
    n_runs += 1
print(f"Variables directly tied to Outcome in the complete-data networks (out of {n_runs}):")
print(pd.Series(counts).sort_values(ascending=False).to_string())

# average AUC gain over Median at 10-30% missing, per seed
low = sc[sc.level.isin([10, 20, 30])]
p = low.pivot_table(index=["seed", "fold", "level"], columns="method", values="auc")
gain = pd.DataFrame({m: p[m] - p["Median"] for m in ["EM", "MICE", "SoftImpute"]}).groupby("seed").mean().round(4)

print("\nPer seed: damaged columns, and which of them the complete-data network uses")
for seed in sorted(edges.seed.unique()):
    vm = json.load(open(AUDIT / f"audit_seed{seed}.json"))["vm_vars"]
    c = Counter()
    n = 0
    for _, g in complete[complete.seed == seed].groupby("fold"):
        c.update(blanket(g))
        n += 1
    used = {v: f"{c[v]}/{n}" for v in vm if c[v] > 0}
    print(f"\nseed {seed}: damaged = {vm}")
    print(f"   damaged columns the network uses (folds): {used if used else 'none'}")
    print(f"   mean AUC gain over Median at 10-30%: {gain.loc[seed].to_dict()}")