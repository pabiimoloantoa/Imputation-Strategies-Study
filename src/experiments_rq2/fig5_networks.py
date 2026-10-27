"""
Usage:
    python fig5_networks.py
    python fig5_networks.py --threshold 15
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EDGES_CSV = PROJECT_ROOT / "results" / "rq2" / "edge_counts.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "rq2" / "figures"

VO_COLOR, VO_EDGE = "#B5D4F4", "#185FA5"
VM_COLOR, VM_EDGE = "#F5C4B3", "#D85A30"
NORMAL_EDGE_COLOR = "#B8B8B4"
OUTCOME_EDGE_COLOR = "#534AB7"
TEXT_COLOR = "#2C2C2A"

BOX_W, BOX_H = 2.0, 0.75

# Vm = variables that have missing values in Pima; Vo = fully observed
VM_VARS = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}
LABELS = {"DiabetesPedigreeFunction": "Diabetes\nPedigreeFunction"}

# Same layered layout in every panel, so the four networks are easy to compare
POS = {
    "Insulin": (-3.0, 0.0), "Glucose": (0.0, 0.0), "DiabetesPedigreeFunction": (3.0, 0.0),
    "SkinThickness": (-3.0, -1.7), "Outcome": (0.0, -1.7), "Age": (3.0, -1.7),
    "BMI": (-1.5, -3.4), "BloodPressure": (1.5, -3.4), "Pregnancies": (4.5, -3.4),
}
METHODS = [("Median", "Median"), ("EM", "EM"), ("MICE", "MICE"), ("SoftImpute", "Soft-Impute")]


def draw_box(ax, x, y, label, is_vm):
    fill, edge = (VM_COLOR, VM_EDGE) if is_vm else (VO_COLOR, VO_EDGE)
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, LABELS.get(label, label), ha="center", va="center", fontsize=7.5,
            color=TEXT_COLOR, zorder=4)


def clip_to_box_edge(from_pt, to_pt, half_w, half_h):
    """Returns the point where the segment from_pt->to_pt crosses the
    boundary of an axis-aligned box centered at to_pt."""
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    if dx == 0 and dy == 0:
        return to_pt
    tx = half_w / abs(dx) if dx != 0 else float("inf")
    ty = half_h / abs(dy) if dy != 0 else float("inf")
    t = min(tx, ty)
    return (to_pt[0] - dx * t, to_pt[1] - dy * t)


def draw_edge(ax, a, b, share, touches_outcome):
    x1, y1 = POS[a]
    x2, y2 = POS[b]
    start = clip_to_box_edge((x2, y2), (x1, y1), BOX_W / 2, BOX_H / 2)
    end = clip_to_box_edge((x1, y1), (x2, y2), BOX_W / 2, BOX_H / 2)
    if touches_outcome:
        color, width, z = OUTCOME_EDGE_COLOR, 1.0 + 1.8 * share, 2
    else:
        color, width, z = NORMAL_EDGE_COLOR, 0.5 + 1.5 * share, 1
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-", linewidth=width, color=color,
        shrinkA=2, shrinkB=2, zorder=z,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=25,
                        help="draw a connection only if it appears in at least this many runs")
    args = parser.parse_args()

    edges = pd.read_csv(EDGES_CSV)
    pairs = edges.apply(lambda r: tuple(sorted([r["parent"], r["child"]])), axis=1)
    edges["a"], edges["b"] = pairs.str[0], pairs.str[1]      # direction ignored
    n_runs = edges[["seed", "fold"]].drop_duplicates().shape[0]
    counts = edges.groupby(["method", "a", "b"]).size()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, (key, title) in zip(axes.ravel(), METHODS):
        for (a, b), n in counts.loc[key].items():
            if n >= args.threshold:
                draw_edge(ax, a, b, n / n_runs, touches_outcome=("Outcome" in (a, b)))
        for node, (x, y) in POS.items():
            draw_box(ax, x, y, node, is_vm=node in VM_VARS)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xlim(-4.4, 5.9)
        ax.set_ylim(-4.2, 0.8)
        ax.set_aspect("equal")
        ax.axis("off")

    handles = [
        Patch(facecolor=VO_COLOR, edgecolor=VO_EDGE, label="Vo \u2014 fully observed"),
        Patch(facecolor=VM_COLOR, edgecolor=VM_EDGE, label="Vm \u2014 has missingness"),
        Line2D([0], [0], color=OUTCOME_EDGE_COLOR, linewidth=1.8, label="Connection to Outcome"),
        Line2D([0], [0], color=NORMAL_EDGE_COLOR, linewidth=1.4, label="Other connection"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.text(0.5, 0.045,
             f"Line width is proportional to the share of the {n_runs} runs in which the connection "
             f"was learned (drawn if found in at least {args.threshold}). Direction is not shown.",
             ha="center", fontsize=8.5)
    fig.suptitle("Learned Bayesian Network structures on Pima, by imputation method",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig5_networks.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()