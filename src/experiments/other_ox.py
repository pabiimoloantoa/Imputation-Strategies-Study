"""
plot_r_node_network.py

Draws the Bayesian Network with the embedded missingness R-nodes for ONE
sparsity level (the structure is identical across levels; only CPTs differ).

Reads:
    models/model_with_r_nodes_seed<N>.bif
    audit_logs/audit_seed<N>.json   (for the Vm / Vo split)

Usage (from src/experiment/ or anywhere):
    python plot_r_node_network.py --seed 0 --pct 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.readwrite import BIFReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "models"
AUDIT_DIR = PROJECT_ROOT / "data" / "synthetic" / "audit_logs"
FIG_DIR = PROJECT_ROOT / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pct", type=int, default=10, choices=[10, 20, 30])
    args = parser.parse_args()

    model = BIFReader(str(MODELS_DIR / f"model_with_r_nodes_seed{args.seed}.bif")).get_model()
    with open(AUDIT_DIR / f"audit_seed{args.seed}.json") as f:
        audit = json.load(f)
    vm_vars = set(audit["vm_vars"])

    suffix = f"_{args.pct}pct"
    r_nodes = [n for n in model.nodes() if n.startswith("R_") and n.endswith(suffix)]
    base_nodes = [n for n in model.nodes() if not n.startswith("R_")]
    keep = set(base_nodes) | set(r_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(keep)
    G.add_edges_from((u, v) for u, v in model.edges() if u in keep and v in keep)

    # Layered layout by topological generation; R-nodes get their own last layer.
    base_G = G.subgraph(base_nodes)
    for layer, nodes in enumerate(nx.topological_generations(base_G)):
        for n in nodes:
            G.nodes[n]["layer"] = layer
    last = max(G.nodes[n]["layer"] for n in base_nodes) + 1
    for n in r_nodes:
        G.nodes[n]["layer"] = last
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

    colors = []
    for n in G.nodes():
        if n in r_nodes:
            colors.append("#f4a261")        # R-nodes
        elif n in vm_vars:
            colors.append("#e76f51")        # Vm (gets missingness)
        else:
            colors.append("#8ecae6")        # Vo (fully observed)

    r_edges = [(u, v) for u, v in G.edges() if v in r_nodes]
    base_edges = [(u, v) for u, v in G.edges() if v not in r_nodes]

    fig, ax = plt.subplots(figsize=(14, 8))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=base_edges, arrows=True, arrowsize=14,
                           node_size=2600, edge_color="#444444", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=r_edges, arrows=True, arrowsize=14,
                           node_size=2600, edge_color="#f4a261", style="dashed", ax=ax)

    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color="#8ecae6", label="Vo (fully observed)"),
        plt.Line2D([], [], marker="o", ls="", color="#e76f51", label="Vm (gets missingness)"),
        plt.Line2D([], [], marker="o", ls="", color="#f4a261", label=f"R-node ({args.pct}% target)"),
    ], loc="upper left")
    ax.set_title(f"Bayesian Network with embedded missingness R-nodes "
                 f"(seed {args.seed}, {args.pct}% sparsity)")
    ax.axis("off")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"bn_with_r_nodes_seed{args.seed}_{args.pct}pct.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()