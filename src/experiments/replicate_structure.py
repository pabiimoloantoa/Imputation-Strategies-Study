"""
Usage:
    python plot_replicate_structure.py --seed 0
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
NORMAL_EDGE_COLOR = "#B8B8B4"
R_EDGE_COLOR = "#534AB7"
TEXT_COLOR = "#2C2C2A"

BOX_W, BOX_H = 2.0, 0.75


def draw_box(ax, x, y, label, is_vm):
    fill, edge = (VM_COLOR, VM_EDGE) if is_vm else (VO_COLOR, VO_EDGE)
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
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


def draw_edge(ax, pos, src, dst, is_r_parent_edge):
    x1, y1 = pos[src]
    x2, y2 = pos[dst]
    start = clip_to_box_edge((x2, y2), (x1, y1), BOX_W / 2, BOX_H / 2)
    end = clip_to_box_edge((x1, y1), (x2, y2), BOX_W / 2, BOX_H / 2)
    if is_r_parent_edge:
        color, width, z = R_EDGE_COLOR, 1.8, 2
    else:
        color, width, z = NORMAL_EDGE_COLOR, 0.9, 1
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color,
        shrinkA=2, shrinkB=2, zorder=z,
        connectionstyle="arc3,rad=0.08",
    )
    ax.add_patch(arrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    audit_path = AUDIT_DIR / f"audit_seed{args.seed}.json"
    audit = json.load(open(audit_path))

    edges = [tuple(e) for e in audit["edges"]]
    vm_vars = set(audit["vm_vars"])
    r_parents = audit["r_parents"]  # {vm_var: parent_var}

    # The specific (parent -> vm_var) edges that also serve as R-parent edges
    r_parent_edges = {(parent, vm) for vm, parent in r_parents.items()}

    G = nx.DiGraph()
    G.add_edges_from(edges)
    # include any isolated/root/leaf nodes not captured by edges alone
    all_nodes = set(vm_vars) | set(audit["vo_vars"])
    G.add_nodes_from(all_nodes)

    # Layered layout by topological generation (same approach as the
    # original ground-truth diagram)
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
        is_r = (src, dst) in r_parent_edges
        draw_edge(ax, pos, src, dst, is_r)

    for node in G.nodes():
        x, y = pos[node]
        draw_box(ax, x, y, node, is_vm=node in vm_vars)

    # Legend
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    legend_y = min(ys) - 1.3

    legend_vo = FancyBboxPatch((min(xs), legend_y), 0.5, 0.35, boxstyle="round,pad=0.02",
                                 linewidth=1.0, edgecolor=VO_EDGE, facecolor=VO_COLOR)
    legend_vm = FancyBboxPatch((min(xs) + 3.2, legend_y), 0.5, 0.35, boxstyle="round,pad=0.02",
                                 linewidth=1.0, edgecolor=VM_EDGE, facecolor=VM_COLOR)
    ax.add_patch(legend_vo)
    ax.add_patch(legend_vm)
    ax.text(min(xs) + 0.65, legend_y + 0.17, "Vo \u2014 fully observed", fontsize=9, va="center")
    ax.text(min(xs) + 3.85, legend_y + 0.17, "Vm \u2014 has missingness", fontsize=9, va="center")

    ax.plot([min(xs) + 6.5, min(xs) + 7.2], [legend_y + 0.17]*2,
             color=R_EDGE_COLOR, linewidth=1.8)
    ax.text(min(xs) + 7.35, legend_y + 0.17, "R-parent \u2192 Vm dependency", fontsize=9, va="center")

    ax.set_title(f"Replicate seed={args.seed}: full network structure, Vm variables and their R-parent edges highlighted",
                 fontsize=11.5, fontweight="bold", pad=15)

    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(legend_y - 0.8, max(ys) + 1)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"replicate_structure_seed{args.seed}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()