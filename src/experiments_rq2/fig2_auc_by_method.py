import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_pima import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "results" / "rq2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(PROJECT_ROOT / "results" / "rq2" / "fold_scores_repeated.csv")

order = ["Median", "EM", "MICE", "SoftImpute"]
labels = ["Median", "EM", "MICE", "Soft-Impute"]
data = [d.loc[d["method"] == m, "auc"].to_numpy() for m in order]

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.boxplot(data, showfliers=False, widths=0.5)

rng = np.random.default_rng(0)
for i, vals in enumerate(data, start=1):
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals,
               s=10, alpha=0.4, color="#4C72B0", zorder=3)         # one dot per run
    ax.scatter(i, vals.mean(), marker="D", s=40, color="#C44E52", zorder=4)  # the mean

ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(labels)
ax.set_ylabel("AUC-ROC")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_auc_by_method.png", dpi=300)
print("saved to", FIG_DIR / "fig2_auc_by_method.png")