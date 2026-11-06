import matplotlib.pyplot as plt
import pandas as pd

from load_pima import load_pima, PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

edges = pd.read_csv(RES / "edges_pooled.csv")
X = load_pima().drop(columns="Outcome")
methods = ["Median", "EM", "MICE", "SoftImpute"]
col_labels = ["Median", "EM", "MICE", "Soft-Impute"]
N_RUNS = 250    # 5 hash seeds x 5 splits x 10 folds


def blanket(g):
    """Variables directly tied to Outcome: its parents, its children, and the children's other parents."""
    parents = {}
    for p, c in zip(g["parent"], g["child"]):
        parents.setdefault(c, set()).add(p)
    kids = [c for c, ps in parents.items() if "Outcome" in ps]
    b = set(parents.get("Outcome", set())) | set(kids)
    for k in kids:
        b |= parents[k]
    b.discard("Outcome")
    return b


counts = pd.DataFrame(0, index=X.columns, columns=methods)
for (m, h, s, f), g in edges.groupby(["method", "hash", "seed", "fold"]):
    for v in blanket(g):
        counts.loc[v, m] += 1
counts.to_csv(RES / "blanket_counts_pooled.csv")
print(counts)

# rows ordered from the most to the least missing in the Pima data
pct = (X.isna().mean() * 100).sort_values(ascending=False)
counts = counts.loc[pct.index]
row_labels = [f"{v} ({p:.1f}% missing)" if p > 0 else f"{v} (complete)" for v, p in pct.items()]

fig, ax = plt.subplots(figsize=(6.8, 4.2))
im = ax.imshow(counts.to_numpy(), cmap="Blues", vmin=0, vmax=N_RUNS, aspect="auto")
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        v = counts.iloc[i, j]
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > N_RUNS / 2 else "black", fontsize=9)
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"Networks in which the variable affects the prediction (out of {N_RUNS})")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_blanket.png", dpi=300)
print("saved to", FIG_DIR / "fig4_blanket.png")