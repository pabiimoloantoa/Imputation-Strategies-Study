import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
AUDIT = ROOT / "data" / "taiwan" / "audit_logs"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(RES / "scores.csv")
edges = pd.read_csv(RES / "edges.csv")

methods = ["EM", "MICE", "SoftImpute"]
labels = ["EM", "MICE", "Soft-Impute"]
colors = {"EM": "#4C72B0", "MICE": "#C44E52", "SoftImpute": "#55A868"}


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


# for each seed: which of its damaged columns does the complete-data network use (in at least half the folds)?
complete = edges[edges.method == "Complete"]
seeds = sorted(sc.seed.unique())
tick = {}
for seed in seeds:
    vm = json.load(open(AUDIT / f"audit_seed{seed}.json"))["vm_vars"]
    c, n = Counter(), 0
    for _, g in complete[complete.seed == seed].groupby("fold"):
        c.update(blanket(g))
        n += 1
    used = [v for v in vm if c[v] >= n / 2]
    tick[seed] = f"Seed {seed}\n" + ("\n".join(used) if used else "none used")


def gains(levels):
    p = sc[sc.level.isin(levels)].pivot_table(index=["seed", "fold", "level"], columns="method", values="auc")
    g = pd.DataFrame({m: p[m] - p["Median"] for m in methods})
    return g.groupby("seed").mean()


panels = [([10, 20, 30], "10 to 30% missing (mean of three levels)"), ([40], "40% missing")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
x = np.arange(len(seeds))
w = 0.26

for ax, (levels, title) in zip(axes, panels):
    g = gains(levels)
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1) * w, g.loc[seeds, m], width=w, color=colors[m], label=labels[i])
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([tick[s] for s in seeds], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Mean AUC difference from Median")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_t4_gain_by_seed.png", dpi=300)
print("saved to", FIG_DIR / "fig_t4_gain_by_seed.png")