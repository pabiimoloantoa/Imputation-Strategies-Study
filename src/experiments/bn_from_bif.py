"""
plot_bn_from_bif.py

Loads a saved .bif network file for a given seed and draws its plain
structure (all nodes the same style, no Vm/Vo/R-parent highlighting) --
useful as a simple "what does this replicate's network look like"
reference figure, independent of the missingness mechanism.

Usage:
    python plot_bn_from_bif.py --seed 0
"""

import argparse
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pgmpy.readwrite import BIFReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

NODE_COLOR, NODE_EDGE = "#B5D4F4", "#185FA5"
TARGET_COLOR, TARGET_EDGE = "#F5C4B3", "#D85A30"
TEXT_COLOR = "#2C2C2A"
BOX_W, BOX_H = 2.0, 0.75


def draw_box(ax, x, y, label, n_states, is_target=False):
    fill, edge = (TARGET_COLOR, TARGET_EDGE) if is_target else (NODE_COLOR, NODE_EDGE)
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, f"{label}\n({n_states} states)", ha="center", va="center",
             fontsize=7.5, color=TEXT_COLOR, zorder=4)


def clip_to_box_edge(from_pt, to_pt, half_w, half_h):
    """Returns the point where the segment from_pt->to_pt crosses the
    boundary of an axis-aligned box of half-width/half-height centered at
    to_pt. Used to stop an arrow exactly at a box's edge, regardless of
    figure size or data range (unlike matplotlib's point-based shrink)."""
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    if dx == 0 and dy == 0:
        return to_pt
    tx = half_w / abs(dx) if dx != 0 else float("inf")
    ty = half_h / abs(dy) if dy != 0 else float("inf")
    t = min(tx, ty)
    return (to_pt[0] - dx * t, to_pt[1] - dy * t)


def draw_edge(ax, pos, src, dst):
    x1, y1 = pos[src]
    x2, y2 = pos[dst]
    start = clip_to_box_edge((x2, y2), (x1, y1), BOX_W / 2, BOX_H / 2)
    end = clip_to_box_edge((x1, y1), (x2, y2), BOX_W / 2, BOX_H / 2)
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=0.9, color="#5F5E5A",
        shrinkA=2, shrinkB=2, zorder=1,
        connectionstyle="arc3,rad=0.08",
    )
    ax.add_patch(arrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    bif_path = MODELS_DIR / f"model_seed{args.seed}.bif"
    model = BIFReader(str(bif_path)).get_model()

    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    generations = list(nx.topological_generations(G))
    pos = {}
    for depth, layer in enumerate(generations):
        n = len(layer)
        for j, node in enumerate(sorted(layer)):
            x = (j - (n - 1) / 2) * 3.0
            y = -depth * 1.7
            pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(12, 8.5))

    for src, dst in G.edges():
        draw_edge(ax, pos, src, dst)

    for node in G.nodes():
        x, y = pos[node]
        n_states = model.get_cardinality(node)
        draw_box(ax, x, y, node, n_states, is_target=(node == "Default"))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    # Legend
    legend_y = min(ys) - 1.0
    legend_normal = FancyBboxPatch((min(xs), legend_y), 0.5, 0.35, boxstyle="round,pad=0.02",
                                     linewidth=1.0, edgecolor=NODE_EDGE, facecolor=NODE_COLOR)
    legend_target = FancyBboxPatch((min(xs) + 3.0, legend_y), 0.5, 0.35, boxstyle="round,pad=0.02",
                                     linewidth=1.0, edgecolor=TARGET_EDGE, facecolor=TARGET_COLOR)
    ax.add_patch(legend_normal)
    ax.add_patch(legend_target)
    ax.text(min(xs) + 0.65, legend_y + 0.17, "Predictor variable", fontsize=9, va="center")
    ax.text(min(xs) + 3.65, legend_y + 0.17, "Default \u2014 target variable", fontsize=9, va="center")

    ax.set_title(f"Replicate seed={args.seed}: randomly generated network structure",
                 fontsize=12, fontweight="bold", pad=15)

    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(legend_y - 0.6, max(ys) + 1)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"bn_structure_seed{args.seed}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()