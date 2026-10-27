import matplotlib.pyplot as plt

from load_pima import load_pima, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "results" / "rq2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = load_pima().drop(columns="Outcome")
pct = (df.isna().mean() * 100).sort_values()
pct = pct[pct > 0]            # only columns that actually have gaps

fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.barh(pct.index, pct.values, color="#4C72B0")
for b, v in zip(bars, pct.values):
    ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)

ax.set_xlabel("Missing values (%)")
ax.set_xlim(0, 60)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_missingness.png", dpi=300)
print("saved to", FIG_DIR / "fig1_missingness.png")