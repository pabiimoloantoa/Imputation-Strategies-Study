from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(RES / "scores.csv")

methods = ["EM", "MICE", "SoftImpute"]
labels = ["EM", "MICE", "Soft-Impute"]
colors = {"EM": "#4C72B0", "MICE": "#C44E52", "SoftImpute": "#55A868"}
levels = [10, 20, 30, 40]

fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)
rng = np.random.default_rng(0)

for ax, level in zip(axes, levels):
    p = sc[sc.level == level].pivot_table(index=["seed", "fold"], columns="method", values="auc")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--", zorder=1)       # no difference from Median
    for i, m in enumerate(methods, start=1):
        diff = (p[m] - p["Median"]).to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(diff)), diff,
                   s=10, alpha=0.45, color=colors[m], zorder=3)              # one dot per fold
        ax.scatter(i, diff.mean(), marker="D", s=45, color="black", zorder=4)  # the mean
    ax.set_title(f"{level}% missing", fontsize=10)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("AUC difference from Median")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_t3_paired_auc.png", dpi=300)
print("saved to", FIG_DIR / "fig_t3_paired_auc.png")