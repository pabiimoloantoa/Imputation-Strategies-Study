from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(RES / "scores.csv")

order = ["Median", "EM", "MICE", "SoftImpute"]
labels = {"Median": "Median", "EM": "EM", "MICE": "MICE", "SoftImpute": "Soft-Impute"}
colors = {"Median": "#8C8C8C", "EM": "#4C72B0", "MICE": "#C44E52", "SoftImpute": "#55A868"}
markers = {"Median": "s", "EM": "o", "MICE": "D", "SoftImpute": "^"}
panels = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
for ax, (metric, title) in zip(axes, panels):
    ceiling = sc.loc[sc.level == 0, metric].mean()
    mean = sc[sc.level > 0].groupby(["level", "method"])[metric].mean().unstack()
    ax.axhline(ceiling, color="black", linestyle="--", linewidth=1)      # complete-data score
    for m in order:
        ax.plot(mean.index, mean[m], marker=markers[m], color=colors[m],
                linewidth=1.6, label=labels[m])
    ax.set_title(f"{title} (complete data: {ceiling:.3f})", fontsize=10)
    ax.set_xticks(list(mean.index))
    ax.set_xticklabels([f"{int(l)}%" for l in mean.index])
    ax.set_xlabel("Missing values in each column with injected missingness", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Mean over 50 folds")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_t6_threshold_metrics.png", dpi=300)
print("saved to", FIG_DIR / "fig_t6_threshold_metrics.png")