import matplotlib.pyplot as plt
import pandas as pd

from load_pima import load_pima, get_folds, PROJECT_ROOT
from run_rq2_2 import IMPUTERS, N_BINS, SEEDS
from bn_classifier import Discretizer, BNClassifier

FIG_DIR = PROJECT_ROOT / "results" / "rq2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV = PROJECT_ROOT / "results" / "rq2" / "blanket_counts.csv"

df = load_pima()
X, y = df.drop(columns="Outcome"), df["Outcome"]
n_runs = len(SEEDS) * 10

# --- count how often each variable is in Outcome's Markov blanket ---
if CSV.exists():
    counts = pd.read_csv(CSV, index_col=0)          # reuse saved counts
else:
    counts = pd.DataFrame(0, index=X.columns, columns=list(IMPUTERS))
    for seed in SEEDS:
        for tr, te in get_folds(df, seed=seed):
            for name, make in IMPUTERS.items():
                imp = make()
                tr_filled = imp.fit_transform(X.iloc[tr])
                disc = Discretizer(N_BINS).fit(tr_filled)
                D = disc.transform(tr_filled)
                D["Outcome"] = y.iloc[tr].to_numpy()
                names = {c: list(range(n)) for c, n in disc.n_states_.items()}
                names["Outcome"] = [0, 1]
                bn = BNClassifier().fit(D, names)
                for v in bn.model_.get_markov_blanket("Outcome"):
                    counts.loc[v, name] += 1
        print("finished seed", seed)
    counts.to_csv(CSV)

# --- order the rows by how much of each variable is missing ---
pct = (X.isna().mean() * 100).sort_values(ascending=False)
counts = counts.loc[pct.index, list(IMPUTERS)]
row_labels = [f"{v} ({p:.1f}% missing)" if p > 0 else f"{v} (complete)" for v, p in pct.items()]
col_labels = ["Median", "EM", "MICE", "Soft-Impute"]

# --- draw the heatmap ---
fig, ax = plt.subplots(figsize=(6.8, 4.2))
im = ax.imshow(counts.to_numpy(), cmap="Blues", vmin=0, vmax=n_runs, aspect="auto")
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        v = counts.iloc[i, j]
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > n_runs / 2 else "black", fontsize=9)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"Runs in which the variable affects the prediction (out of {n_runs})")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_blanket.png", dpi=300)
print("saved to", FIG_DIR / "fig4_blanket.png")