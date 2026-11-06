from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ac = pd.read_csv(RES / "accuracy.csv")

# one number per run (the average over that seed's damaged columns), then the mean over the 50 folds
per_run = ac.groupby(["seed", "level", "fold", "method"])["scaled_error"].mean().reset_index()
mean_err = per_run.groupby(["level", "method"])["scaled_error"].mean().unstack()

order = ["Median", "EM", "MICE", "SoftImpute"]
labels = {"Median": "Median", "EM": "EM", "MICE": "MICE", "SoftImpute": "Soft-Impute"}
colors = {"Median": "#8C8C8C", "EM": "#4C72B0", "MICE": "#C44E52", "SoftImpute": "#55A868"}
markers = {"Median": "s", "EM": "o", "MICE": "D", "SoftImpute": "^"}

fig, ax = plt.subplots(figsize=(6.4, 4))
for m in order:
    ax.plot(mean_err.index, mean_err[m], marker=markers[m], color=colors[m],
            linewidth=1.6, label=labels[m])

ax.set_xticks(list(mean_err.index))
ax.set_xticklabels([f"{int(l)}%" for l in mean_err.index])
ax.set_ylim(0, 0.75)
ax.set_xlabel("Missing values in each damaged column")
ax.set_ylabel("Scaled error of filled values (lower is better)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_t2_error_by_level.png", dpi=300)
print("saved to", FIG_DIR / "fig_t2_error_by_level.png")