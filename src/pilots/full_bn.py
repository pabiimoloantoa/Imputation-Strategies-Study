"""
plot_full_bn_highlighted.py

Recreates the full 11-variable ground-truth BN structure, with the five Vm
variables (selected for missingness injection) highlighted in coral against
the six Vo variables in blue. Intended as a companion "before" figure to
plot_mar_mechanism.py's "after/mechanism" figure.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

VM_VARS = {"Income", "Default", "EmploymentStatus", "CreditUtilization", "Age"}

# Layout mirrors the original ground-truth diagram's tiered structure
NODES = {
    "DebtRatio": (1.5, 5),
    "LoanAmount": (5, 5),
    "PaymentHistory": (8.5, 5),
    "CreditHistoryLength": (2.5, 4),
    "Income": (7, 4),
    "CreditUtilization": (2.5, 3),
    "NumberOfDefaults": (7, 3),
    "EmploymentStatus": (2.5, 2),
    "NumberOfCreditInquiries": (7, 2),
    "Age": (2.5, 1),
    "Default": (7, 1),
}

EDGES = [
    ("DebtRatio", "Income"),
    ("DebtRatio", "Default"),
    ("LoanAmount", "Income"),
    ("LoanAmount", "CreditHistoryLength"),
    ("LoanAmount", "NumberOfCreditInquiries"),
    ("LoanAmount", "NumberOfDefaults"),
    ("CreditHistoryLength", "Age"),
    ("CreditHistoryLength", "EmploymentStatus"),
    ("CreditHistoryLength", "NumberOfDefaults"),
    ("CreditHistoryLength", "NumberOfCreditInquiries"),
    ("CreditHistoryLength", "CreditUtilization"),
    ("CreditHistoryLength", "Default"),
    ("CreditUtilization", "EmploymentStatus"),
    ("PaymentHistory", "Age"),
    ("Income", "EmploymentStatus"),
    ("NumberOfDefaults", "NumberOfCreditInquiries"),
    ("NumberOfCreditInquiries", "Age"),
    ("NumberOfCreditInquiries", "Default"),
]

VO_COLOR, VO_EDGE = "#B5D4F4", "#185FA5"
VM_COLOR, VM_EDGE = "#F5C4B3", "#D85A30"
TEXT_COLOR = "#2C2C2A"

BOX_W, BOX_H = 1.9, 0.55


def draw_box(ax, x, y, label, is_vm):
    fill, edge = (VM_COLOR, VM_EDGE) if is_vm else (VO_COLOR, VO_EDGE)
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.8,
             color=TEXT_COLOR, zorder=4)


def draw_edge(ax, src, dst):
    x1, y1 = NODES[src]
    x2, y2 = NODES[dst]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color="#8A8A86",
        shrinkA=16, shrinkB=16, zorder=2,
        connectionstyle="arc3,rad=0.05",
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(9, 7))

    for src, dst in EDGES:
        draw_edge(ax, src, dst)

    for name, (x, y) in NODES.items():
        draw_box(ax, x, y, name, is_vm=name in VM_VARS)

    # Legend
    legend_vo = FancyBboxPatch((0.3, -0.4), 0.4, 0.3, boxstyle="round,pad=0.02",
                                 linewidth=0.9, edgecolor=VO_EDGE, facecolor=VO_COLOR)
    legend_vm = FancyBboxPatch((3.3, -0.4), 0.4, 0.3, boxstyle="round,pad=0.02",
                                 linewidth=0.9, edgecolor=VM_EDGE, facecolor=VM_COLOR)
    ax.add_patch(legend_vo)
    ax.add_patch(legend_vm)
    ax.text(0.85, -0.25, "Vo — fully observed throughout", fontsize=9, va="center")
    ax.text(3.85, -0.25, "Vm — selected for missingness injection", fontsize=9, va="center")

    ax.set_title("Ground-truth BN structure (11 variables), with Vm variables highlighted",
                 fontsize=12, fontweight="bold", pad=15)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.7, 5.7)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "ground_truth_bn_highlighted.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()