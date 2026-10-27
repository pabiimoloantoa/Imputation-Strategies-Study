"""
plot_marginals.py

Bar charts of the empirical marginal distribution of each R-parent variable,
computed from the ground-truth dataset (see compute_marginals.py). These are
the real proportions used to derive the R-node CPTs.
"""

import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# Real proportions from compute_marginals.py output, on groundtruth.csv (n=5000)
MARGINALS = {
    "DebtRatio": {"low": 0.3786, "medium": 0.2056, "high": 0.4158},
    "NumberOfCreditInquiries": {"low": 0.3728, "medium": 0.3164, "high": 0.3108},
    "CreditHistoryLength": {"short": 0.3706, "medium": 0.3310, "long": 0.2984},
    "PaymentHistory": {"excellent": 0.2370, "good": 0.3164, "fair": 0.1616, "poor": 0.2850},
}

# Preferred state ordering per variable, for readability
STATE_ORDER = {
    "DebtRatio": ["low", "medium", "high"],
    "NumberOfCreditInquiries": ["low", "medium", "high"],
    "CreditHistoryLength": ["short", "medium", "long"],
    "PaymentHistory": ["excellent", "good", "fair", "poor"],
}

BAR_COLOR = "#B5D4F4"
BAR_EDGE = "#185FA5"


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    for ax, (var, dist) in zip(axes, MARGINALS.items()):
        states = STATE_ORDER[var]
        proportions = [dist[s] * 100 for s in states]
        ax.bar(states, proportions, color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.8)
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 50)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        for i, p in enumerate(proportions):
            ax.text(i, p + 1, f"{p:.1f}%", ha="center", fontsize=8)

    axes[0].set_ylabel("Proportion of dataset (%)")
    fig.suptitle("Marginal distributions of R-parent variables (n = 5000)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "r_parent_marginals.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()