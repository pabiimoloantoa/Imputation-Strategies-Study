from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "taiwan"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(RES / "scores.csv")
clean = pd.read_csv(ROOT / "data" / "taiwan" / "taiwan_clean.csv")
X, y = clean.drop(columns="Outcome"), clean["Outcome"]

# number of defaulters (P) and non-defaulters (N) in every test fold, rebuilt from the same splits
size = {}
for seed in sc.seed.unique():
    splits = StratifiedKFold(10, shuffle=True, random_state=int(seed)).split(X, y)
    for k, (tr, te) in enumerate(splits, 1):
        P = int(y.iloc[te].sum())
        size[(int(seed), k)] = (P, len(te) - P)

# rebuild the four counts for every run, checking each one against the saved scores
rows = []
for r in sc.itertuples():
    P, N = size[(int(r.seed), int(r.fold))]
    tp = round(r.recall * P)
    if tp == 0:
        raise ValueError(f"No true positives in {r.method}, seed {r.seed}, fold {r.fold}")
    fn = P - tp
    fp = round(tp / r.precision - tp)
    tn = N - fp
    assert min(tp, fn, fp, tn) >= 0
    assert abs(tp / (tp + fp) - r.precision) < 1e-9
    assert abs(tp / P - r.recall) < 1e-9
    assert abs(2 * tp / (2 * tp + fp + fn) - r.f1) < 1e-9
    rows.append({"level": r.level, "method": r.method, "tn": tn, "fp": fp, "fn": fn, "tp": tp})

cm = pd.DataFrame(rows).groupby(["level", "method"])[["tn", "fp", "fn", "tp"]].sum()
cm["recall"] = (cm.tp / (cm.tp + cm.fn)).round(3)
cm["specificity"] = (cm.tn / (cm.tn + cm.fp)).round(3)
cm.to_csv(RES / "confusion_counts.csv")
print(cm.to_string())

LABEL = {"Complete": "Complete data", "Median": "Median", "EM": "EM", "MICE": "MICE", "SoftImpute": "Soft-Impute"}
METHODS = ["Median", "EM", "MICE", "SoftImpute"]


def draw(ax, level, method):
    c = cm.loc[(level, method)]
    counts = [[c.tn, c.fp], [c.fn, c.tp]]
    share = [[v / sum(row) for v in row] for row in counts]       # share of each actual class
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{100 * share[i][j]:.1f}%\n(n = {int(counts[i][j]):,})",
                    ha="center", va="center", fontsize=8,
                    color="white" if share[i][j] > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No default", "Default"], fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No default", "Default"], fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    title = LABEL[method] if level == 0 else f"{LABEL[method]}, {level}%"
    ax.set_title(title, fontsize=9, fontweight="bold")


def make_figure(levels, name, height):
    fig, axes = plt.subplots(len(levels) + 1, 4, figsize=(9, height), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].axis("on")
    draw(axes[0, 0], 0, "Complete")                                # the ceiling, on its own row
    for r, level in enumerate(levels, start=1):
        for c, m in enumerate(METHODS):
            axes[r, c].axis("on")
            draw(axes[r, c], level, m)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=300)
    print("saved to", FIG_DIR / name)


make_figure([10, 40], "fig_t7_confusion_main.png", 8.0)
make_figure([10, 20, 30, 40], "fig_t7_confusion_all.png", 12.5)