"""
plot_replicate_structure_rnodes.py

Same style as plot_replicate_structure.py, extended with the missingness
R-nodes. For every Vm variable it shows:

  1. Vo parent -> Vm        (causal edge from the network; highlighted purple, thick)
  2. Vo parent -> R-node    (real edge in the augmented pgmpy network; purple, solid)
  3. R-node  ---> Vm        (CONCEPTUAL, dashed: the R-node decides whether Vm is
                             observed; it is NOT an edge in the pgmpy model and
                             does not change Vm's values)

The R-node structure is identical at 10/20/30% sparsity (only the CPT numbers
differ), so one R-node per Vm variable is drawn.

Reads only the audit log (edges, vm_vars, vo_vars, r_parents).

Usage:
    python plot_replicate_structure_rnodes.py --seed 0
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_DIR = PROJECT_ROOT / "data" / "synthetic" / "audit_logs"
FIGURES_DIR = PROJECT_ROOT / "figures"

VO_COLOR, VO_EDGE = "#B5D4F4", "#185FA5"
VM_COLOR, VM_EDGE = "#F5C4B3", "#D85A30"
R_COLOR, R_EDGE = "#CECBF6", "#534AB7"
NORMAL_EDGE_COLOR = "#B8B8B4"
R_EDGE_COLOR = "#534AB7"
TEXT_COLOR = "#2C2C2A"

BOX_W, BOX_H = 2.0, 0.75
R_W, R_H = 1.7, 0.6
X_SPACING = 5.4      # wider than the original (3.0) to leave room for R-nodes
Y_SPACING = 1.9
R_OFFSET_X = 2.8     # R-node sits to the right of its Vm variable


def draw_box(ax, x, y, w, h, label, fill, edge, fontsize=7.5):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=TEXT_COLOR, zorder=4)


def clip_to_box_edge(from_pt, to_pt, half_w, half_h):
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    if dx == 0 and dy == 0:
        return to_pt
    tx = half_w / abs(dx) if dx != 0 else float("inf")
    ty = half_h / abs(dy) if dy != 0 else float("inf")
    t = min(tx, ty)
    return (to_pt[0] - dx * t, to_pt[1] - dy * t)


def draw_edge(ax, pos, sizes, src, dst, color, width, linestyle="-", rad=0.08, z=1):
    x1, y1 = pos[src]
    x2, y2 = pos[dst]
    sw, sh = sizes[src]
    dw, dh = sizes[dst]
    start = clip_to_box_edge((x2, y2), (x1, y1), sw / 2, sh / 2)
    end = clip_to_box_edge((x1, y1), (x2, y2), dw / 2, dh / 2)
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color, linestyle=linestyle,
        shrinkA=2, shrinkB=2, zorder=z,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    audit = json.load(open(AUDIT_DIR / f"audit_seed{args.seed}.json"))
    edges = [tuple(e) for e in audit["edges"]]
    vm_vars = set(audit["vm_vars"])
    r_parents = audit["r_parents"]  # {vm_var: vo_parent}
    r_parent_edges = {(p, vm) for vm, p in r_parents.items()}

    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.add_nodes_from(set(audit["vm_vars"]) | set(audit["vo_vars"]))

    # Layered layout by topological generation (same approach as the original figure)
    pos, sizes = {}, {}
    for depth, layer in enumerate(nx.topological_generations(G)):
        n = len(layer)
        for j, node in enumerate(sorted(layer)):
            pos[node] = ((j - (n - 1) / 2) * X_SPACING, -depth * Y_SPACING)
            sizes[node] = (BOX_W, BOX_H)

    # R-nodes: one per Vm variable, placed to the right of that variable
    r_name = {vm: f"R_{vm}" for vm in vm_vars}
    for vm in vm_vars:
        x, y = pos[vm]
        pos[r_name[vm]] = (x + R_OFFSET_X, y)
        sizes[r_name[vm]] = (R_W, R_H)

    fig, ax = plt.subplots(figsize=(15, 9))

    # Network edges (grey, or purple thick if it is an R-parent -> Vm edge)
    for src, dst in G.edges():
        if (src, dst) in r_parent_edges:
            draw_edge(ax, pos, sizes, src, dst, R_EDGE_COLOR, 1.8, z=2)
        else:
            draw_edge(ax, pos, sizes, src, dst, NORMAL_EDGE_COLOR, 0.9, z=1)

    # Arrow 2 (Vo parent -> R-node, solid) and arrow 3 (R-node ---> Vm, dashed)
    for vm in vm_vars:
        draw_edge(ax, pos, sizes, r_parents[vm], r_name[vm], R_EDGE_COLOR, 1.4, rad=-0.12, z=2)
        draw_edge(ax, pos, sizes, r_name[vm], vm, R_EDGE_COLOR, 1.4, linestyle="--", rad=0.0, z=2)

    for node in G.nodes():
        x, y = pos[node]
        fill, edge = (VM_COLOR, VM_EDGE) if node in vm_vars else (VO_COLOR, VO_EDGE)
        draw_box(ax, x, y, BOX_W, BOX_H, node, fill, edge)
    for vm in vm_vars:
        x, y = pos[r_name[vm]]
        draw_box(ax, x, y, R_W, R_H, r_name[vm], R_COLOR, R_EDGE, fontsize=7)

    # Legend (two rows)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x0 = min(xs) - 1.0
    row1 = min(ys) - 1.4
    row2 = row1 - 0.7

    for dx, (fill, edge, text) in zip(
        [0.0, 4.2, 8.6],
        [(VO_COLOR, VO_EDGE, "Vo \u2014 fully observed"),
         (VM_COLOR, VM_EDGE, "Vm \u2014 has missingness"),
         (R_COLOR, R_EDGE, "R-node \u2014 missingness indicator")],
    ):
        ax.add_patch(FancyBboxPatch((x0 + dx, row1), 0.5, 0.35, boxstyle="round,pad=0.02",
                                    linewidth=1.0, edgecolor=edge, facecolor=fill))
        ax.text(x0 + dx + 0.65, row1 + 0.17, text, fontsize=9, va="center")

    line_items = [
        (0.0, NORMAL_EDGE_COLOR, 0.9, "-", "causal edge"),
        (3.6, R_EDGE_COLOR, 1.8, "-", "R-parent \u2192 Vm (causal)"),
        (8.0, R_EDGE_COLOR, 1.4, "-", "R-parent \u2192 R-node (model edge)"),
        (13.0, R_EDGE_COLOR, 1.4, "--", "R-node decides if Vm is observed (not a model edge)"),
    ]
    for dx, color, lw, ls, text in line_items:
        ax.plot([x0 + dx, x0 + dx + 0.7], [row2 + 0.17] * 2, color=color, linewidth=lw, linestyle=ls)
        ax.text(x0 + dx + 0.85, row2 + 0.17, text, fontsize=9, va="center")

    ax.set_title(f"Replicate seed={args.seed}: missingness mechanism with R-nodes "
                 f"(structure identical at 10/20/30% sparsity)",
                 fontsize=11.5, fontweight="bold", pad=15)

    ax.set_xlim(min(xs) - 2, max(xs) + 2 + R_W)
    ax.set_ylim(row2 - 0.8, max(ys) + 1)
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"replicate_structure_rnodes_seed{args.seed}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()