import matplotlib.pyplot as plt
import pandas as pd

from load_pima import load_pima, get_folds, PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(RES / "fold_scores_repeated.csv")
df = load_pima()
y = df["Outcome"].to_numpy()

# diabetes (P) and non-diabetes (N) patients in every test fold
size = {}
for seed in d["seed"].unique():
    for k, (tr, te) in enumerate(get_folds(df, seed=int(seed)), 1):
        P = int(y[te].sum())
        size[(int(seed), k)] = (P, len(te) - P)

# rebuild the four counts for every run, checking each one
rows = []
for r in d.itertuples():
    P, N = size[(int(r.seed), int(r.fold))]
    tp = round(r.recall * P)
    if tp == 0:
        raise ValueError(f"No true positives in {r.method} seed {r.seed} fold {r.fold}")
    fn = P - tp
    fp = round(tp / r.precision - tp)
    tn = N - fp
    assert min(tp, fn, fp, tn) >= 0
    assert abs(tp / (tp + fp) - r.precision) < 1e-9
    assert abs(tp / P - r.recall) < 1e-9
    rows.append({"method": r.method, "tn": tn, "fp": fp, "fn": fn, "tp": tp})

cm = pd.DataFrame(rows).groupby("method")[["tn", "fp", "fn", "tp"]].sum()
cm.to_csv(RES / "confusion_counts.csv")
print(cm)

order = ["Median", "EM", "MICE", "SoftImpute"]
titles = ["Median", "EM", "MICE", "Soft-Impute"]
classes = ["No diabetes", "Diabetes"]

fig, axes = plt.subplots(2, 2, figsize=(8, 7.2))
for ax, m, title in zip(axes.ravel(), order, titles):
    c = cm.loc[m]
    counts = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    share = [[v / sum(row) for v in row] for row in counts]      # share of each actual class
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{100 * share[i][j]:.1f}%\n(n = {int(counts[i][j])})",
                    ha="center", va="center", fontsize=10,
                    color="white" if share[i][j] > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_confusion.png", dpi=300)
print("saved to", FIG_DIR / "fig6_confusion.png")