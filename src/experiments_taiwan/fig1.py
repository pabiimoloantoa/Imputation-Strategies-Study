from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(RES / "scores.csv")
ceiling = sc.loc[sc.level == 0, "auc"].mean()
mean_auc = sc[sc.level > 0].groupby(["level", "method"])["auc"].mean().unstack()

order = ["Median", "EM", "MICE", "SoftImpute"]
labels = {"Median": "Median", "EM": "EM", "MICE": "MICE", "SoftImpute": "Soft-Impute"}
colors = {"Median": "#8C8C8C", "EM": "#4C72B0", "MICE": "#C44E52", "SoftImpute": "#55A868"}
markers = {"Median": "s", "EM": "o", "MICE": "D", "SoftImpute": "^"}

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.axhline(ceiling, color="black", linestyle="--", linewidth=1,
           label=f"Complete data ({ceiling:.3f})")
for m in order:
    ax.plot(mean_auc.index, mean_auc[m], marker=markers[m], color=colors[m],
            linewidth=1.6, label=labels[m])

ax.set_xticks(list(mean_auc.index))
ax.set_xticklabels([f"{int(l)}%" for l in mean_auc.index])
ax.set_xlabel("Missing values in each damaged column")
ax.set_ylabel("Mean AUC-ROC (50 folds)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_t1_auc_by_level.png", dpi=300)
print("saved to", FIG_DIR / "fig_t1_auc_by_level.png")