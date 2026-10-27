"""
plot_mar_mechanism.py

Generates a figure showing the MAR missingness injection mechanism:
- Left column: Vo variables acting as R-parents (fully observed)
- Middle column: R indicator nodes (one per Vm variable)
- Right column: Vm variables (subject to injected missingness)

Dashed arrows: Vo-parent -> R (the MAR dependency)
Dotted arrows: R -> Vm (R governs whether that value is observed or hidden)

Saves the figure to the project's figures folder.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# --- Paths (adjust if your folder structure differs) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# --- Structure definition: Vm variable -> (Vo R-parent) ---
# Rows are drawn top to bottom in this order.
ROWS = [
    ("DebtRatio", "Income"),
    ("NumberOfCreditInquiries", "Default"),
    ("CreditHistoryLength", "EmploymentStatus"),   # shares parent with next row
    ("CreditHistoryLength", "CreditUtilization"),  # shares parent with previous row
    ("PaymentHistory", "Age"),
]

# --- Style constants ---
VO_COLOR = "#B5D4F4"      # light blue fill
VO_EDGE = "#185FA5"
VM_COLOR = "#F5C4B3"      # light coral fill
VM_EDGE = "#D85A30"
R_COLOR = "#CECBF6"       # light purple fill
R_EDGE = "#534AB7"
TEXT_COLOR = "#2C2C2A"

BOX_W_VO = 2.6
BOX_W_VM = 2.2
BOX_W_R = 1.5
BOX_H = 0.55

VO_X = 0.5
R_X = 4.1
VM_X = 6.3

ROW_Y = [5.0, 4.0, 3.0, 2.0, 1.0]  # top to bottom, evenly spaced


def draw_box(ax, x, y, w, h, label, fill, edge, fontsize=9):
    box = FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.8, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y, label, ha="center", va="center",
             fontsize=fontsize, color=TEXT_COLOR, wrap=True)


def draw_r_node(ax, x, y, vm_var):
    draw_box(ax, x, y, BOX_W_R, BOX_H, f"R_{vm_var}", R_COLOR, R_EDGE, fontsize=8)


def draw_arrow(ax, x1, y1, x2, y2, style):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color="#5F5E5A",
        linestyle=style, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 7))

    # Figure title
    ax.text(4.5, 6.15, "MAR missingness injection mechanism",
             ha="center", fontsize=13, fontweight="bold")

    # Column headers + subtitles
    ax.text(VO_X + BOX_W_VO / 2, 5.75, "Vo (fully observed)",
             ha="center", fontsize=10, fontweight="bold")
    ax.text(VO_X + BOX_W_VO / 2, 5.5, "never missing; used as R-parent",
             ha="center", fontsize=7.5, style="italic", color="#5F5E5A")

    ax.text(R_X + BOX_W_R / 2, 5.75, "R indicator node",
             ha="center", fontsize=10, fontweight="bold")
    ax.text(R_X + BOX_W_R / 2, 5.5, "decides: observed or missing?",
             ha="center", fontsize=7.5, style="italic", color="#5F5E5A")

    ax.text(VM_X + BOX_W_VM / 2, 5.75, "Vm (has missingness)",
             ha="center", fontsize=10, fontweight="bold")
    ax.text(VM_X + BOX_W_VM / 2, 5.5, "value shown to imputation methods",
             ha="center", fontsize=7.5, style="italic", color="#5F5E5A")

    drawn_vo_boxes = {}  # parent_name -> (x, y_top, y_bottom) to avoid redrawing shared parents

    for i, (vo_parent, vm_var) in enumerate(ROWS):
        y = ROW_Y[i]

        # Draw Vo box only once per unique parent; track its span if reused
        if vo_parent not in drawn_vo_boxes:
            drawn_vo_boxes[vo_parent] = {"y_values": [y]}
        else:
            drawn_vo_boxes[vo_parent]["y_values"].append(y)

        # R node + Vm box are always drawn per row
        r_y = y
        draw_r_node(ax, R_X, r_y, vm_var)
        draw_box(ax, VM_X, y, BOX_W_VM, BOX_H, vm_var, VM_COLOR, VM_EDGE)

        # Arrow R -> Vm (dotted)
        draw_arrow(ax, R_X + BOX_W_R, r_y, VM_X, y, style=(0, (1, 1.5)))

    # Now draw Vo boxes (merging shared parents into one spanning box) and Vo -> R arrows
    for vo_parent, info in drawn_vo_boxes.items():
        y_values = info["y_values"]
        y_center = sum(y_values) / len(y_values)
        if len(y_values) > 1:
            height = (max(y_values) - min(y_values)) + BOX_H
        else:
            height = BOX_H
        draw_box(ax, VO_X, y_center, BOX_W_VO, height, vo_parent, VO_COLOR, VO_EDGE)

        for y in y_values:
            draw_arrow(ax, VO_X + BOX_W_VO, y, R_X, y, style=(0, (4, 3)))

    # Legend
    ax.plot([0.5, 1.1], [0.15, 0.15], color="#5F5E5A", linestyle=(0, (4, 3)), linewidth=1)
    ax.text(1.2, 0.15, "Vo-parent influences R (MAR dependency)", fontsize=8, va="center")
    ax.plot([0.5, 1.1], [-0.2, -0.2], color="#5F5E5A", linestyle=(0, (1, 1.5)), linewidth=1)
    ax.text(1.2, -0.2, "R governs whether the Vm value is observed or hidden", fontsize=8, va="center")

    ax.set_xlim(0, 9)
    ax.set_ylim(-0.6, 6.5)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "mar_missingness_mechanism.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()