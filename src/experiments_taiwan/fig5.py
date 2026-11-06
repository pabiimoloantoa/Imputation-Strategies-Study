from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ac = pd.read_csv(RES / "accuracy.csv")

order = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "PAY_0", "PAY_2", "PAY_3",
         "PAY_AMT1", "PAY_AMT2", "LIMIT_BAL", "AGE"]
order = [c for c in order if c in ac["column"].unique()]
methods = ["Median", "EM", "MICE", "SoftImpute"]
labels = {"Median": "Median", "EM": "EM", "MICE": "MICE", "SoftImpute": "Soft-Impute"}
colors = {"Median": "#8C8C8C", "EM": "#4C72B0", "MICE": "#C44E52", "SoftImpute": "#55A868"}

err = ac.groupby(["column", "method"])["scaled_error"].mean().unstack().loc[order, methods]
n_seeds = ac.groupby("column")["seed"].nunique()

fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(order))
w = 0.2
for i, m in enumerate(methods):
    ax.bar(x + (i - 1.5) * w, err[m], width=w, color=colors[m], label=labels[m])

ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n({n_seeds[c]} seed{'s' if n_seeds[c] > 1 else ''})" for c in order], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Scaled error of filled values (lower is better)")
ax.legend(frameon=False, ncol=4, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_t5_error_by_column.png", dpi=300)
print("saved to", FIG_DIR / "fig_t5_error_by_column.png")