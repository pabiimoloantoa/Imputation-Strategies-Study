"""
plot_cpt_curves.py

For each R-node, plots the derived missingness probability across its
parent's states, at each of the three target sparsity levels (10/20/30%).
Visualizes the CPTs derived analytically in the methodology (see
inject_mar_missingness.py for the source values).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# vm_var -> (parent_name, {sparsity: {state: prob}}), states listed in a
# meaningful order (e.g. low -> high) for each parent
R_NODE_CPTS = {
    "Income": {
        "parent": "DebtRatio",
        "states": ["low", "medium", "high"],
        0.10: {"low": 0.066, "medium": 0.099, "high": 0.132},
        0.20: {"low": 0.132, "medium": 0.198, "high": 0.263},
        0.30: {"low": 0.198, "medium": 0.296, "high": 0.395},
    },
    "Default": {
        "parent": "NumberOfCreditInquiries",
        "states": ["low", "medium", "high"],
        0.10: {"low": 0.068, "medium": 0.102, "high": 0.136},
        0.20: {"low": 0.136, "medium": 0.204, "high": 0.272},
        0.30: {"low": 0.204, "medium": 0.306, "high": 0.408},
    },
    "EmploymentStatus": {
        "parent": "CreditHistoryLength",
        "states": ["short", "medium", "long"],
        0.10: {"short": 0.130, "medium": 0.098, "long": 0.065},
        0.20: {"short": 0.260, "medium": 0.195, "long": 0.130},
        0.30: {"short": 0.391, "medium": 0.293, "long": 0.195},
    },
    "CreditUtilization": {
        "parent": "CreditHistoryLength",
        "states": ["short", "medium", "long"],
        0.10: {"short": 0.068, "medium": 0.103, "long": 0.137},
        0.20: {"short": 0.137, "medium": 0.205, "long": 0.273},
        0.30: {"short": 0.205, "medium": 0.307, "long": 0.410},
    },
    "Age": {
        "parent": "PaymentHistory",
        "states": ["excellent", "good", "fair", "poor"],
        0.10: {"excellent": 0.068, "good": 0.088, "fair": 0.108, "poor": 0.135},
        0.20: {"excellent": 0.135, "good": 0.176, "fair": 0.217, "poor": 0.271},
        0.30: {"excellent": 0.203, "good": 0.264, "fair": 0.325, "poor": 0.406},
    },
}

SPARSITY_COLORS = {0.10: "#B5D4F4", 0.20: "#F5C4B3", 0.30: "#CECBF6"}
SPARSITY_EDGES = {0.10: "#185FA5", 0.20: "#D85A30", 0.30: "#534AB7"}


def main():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

    for ax, (vm_var, spec) in zip(axes, R_NODE_CPTS.items()):
        states = spec["states"]
        n_states = len(states)
        n_targets = 3
        bar_width = 0.25
        x = np.arange(n_states)

        for i, target in enumerate([0.10, 0.20, 0.30]):
            probs = [spec[target][s] * 100 for s in states]
            offset = (i - 1) * bar_width
            ax.bar(x + offset, probs, width=bar_width,
                   color=SPARSITY_COLORS[target], edgecolor=SPARSITY_EDGES[target],
                   linewidth=0.7, label=f"{target:.0%} target")

        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"R_{vm_var}\n(parent: {spec['parent']})", fontsize=9.5)
        ax.set_ylim(0, 45)

    axes[0].set_ylabel("P(missing) (%)")
    axes[-1].legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1))

    fig.suptitle("Derived R-node CPTs: missingness probability by parent state",
                 fontsize=13, fontweight="bold", y=1.05)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "r_node_cpt_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()