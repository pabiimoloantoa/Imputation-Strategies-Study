import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_pima import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "results" / "rq2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(PROJECT_ROOT / "results" / "rq2" / "fold_scores_repeated.csv")

methods = ["EM", "MICE", "SoftImpute"]
labels = ["EM", "MICE", "Soft-Impute"]
panels = [("auc", "AUC-ROC difference from Median"), ("f1", "F1 difference from Median")]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
rng = np.random.default_rng(0)

for ax, (metric, ylabel) in zip(axes, panels):
    p = d.pivot_table(index=["seed", "fold"], columns="method", values=metric)
    ax.axhline(0, color="grey", linewidth=1, linestyle="--", zorder=1)   # no difference
    for i, m in enumerate(methods, start=1):
        diff = (p[m] - p["Median"]).to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(diff)), diff,
                   s=10, alpha=0.4, color="#4C72B0", zorder=3)           # one dot per fold
        ax.scatter(i, diff.mean(), marker="D", s=40, color="#C44E52", zorder=4)  # the mean
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_paired_difference.png", dpi=300)
print("saved to", FIG_DIR / "fig3_paired_difference.png")