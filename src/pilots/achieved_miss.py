"""
plot_achieved_vs_target.py

Bar chart of achieved missingness rate per Vm variable, at each target
sparsity level (10%/20%/30%), using real output from a seed-0 injection run.
A dashed reference line marks the intended target for each panel.
"""

import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# Real achieved rates from inject_mar_missingness.py, seed=0
ACHIEVED = {
    0.10: {"Income": 0.1054, "Default": 0.1004, "EmploymentStatus": 0.1018,
           "CreditUtilization": 0.0954, "Age": 0.1030},
    0.20: {"Income": 0.1982, "Default": 0.2094, "EmploymentStatus": 0.1938,
           "CreditUtilization": 0.2024, "Age": 0.2040},
    0.30: {"Income": 0.2932, "Default": 0.2932, "EmploymentStatus": 0.3136,
           "CreditUtilization": 0.2924, "Age": 0.2912},
}

VARS = ["Income", "Default", "EmploymentStatus", "CreditUtilization", "Age"]
BAR_COLOR = "#F5C4B3"
BAR_EDGE = "#D85A30"
TARGET_COLOR = "#2C2C2A"


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, target in zip(axes, [0.10, 0.20, 0.30]):
        rates = [ACHIEVED[target][v] * 100 for v in VARS]
        ax.bar(VARS, rates, color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.8)
        ax.axhline(target * 100, color=TARGET_COLOR, linestyle="--", linewidth=1)
        ax.set_title(f"Target: {target:.0%}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 40)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    axes[0].set_ylabel("Achieved missingness rate (%)")
    fig.suptitle("Achieved vs. target missingness rates (seed = 0)", fontsize=13, fontweight="bold")

    # Single legend entry for the target line
    axes[-1].axhline(target * 100, color=TARGET_COLOR, linestyle="--", linewidth=1, label="Target sparsity")
    axes[-1].legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "achieved_vs_target_missingness.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()