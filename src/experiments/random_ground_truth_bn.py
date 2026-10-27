"""
Randomly generated ground-truth Bayesian Network for RQ1 -- v2, 11 variables.

Structure and CPD parameters are generated entirely at random (see
generate_random_dag() and pgmpy's get_random_cpds()). Nothing about which
edges exist or what any probability equals is chosen by hand anywhere in
this file.

The only human input is:
  1. The LIST of variable names (fixed before any generation happens, so
     names can't be reverse-fitted to whatever structure comes out).
  2. The NUMBER OF STATES per variable, which reflects each variable's
     real-world type (continuous -> more bins, true categories -> their
     natural count) -- not a lever used to shape the network's behaviour.
  3. High-level constraints (n_nodes, max_indegree, edge_prob) derived from
     the literature review of real credit-risk datasets with missing data
     (HMEQ, Give Me Some Credit, Australian Credit Approval).

Running this file directly will:
  - build the random ground-truth BN
  - run and print the pre-registered validity checks
  - sample the 5000-row complete synthetic dataset and SAVE it to
    ../data/synthetic/ground_truth_complete.csv
  - draw the DAG and SAVE it to ../figures/random_bn_structure.png
"""

import os
import random

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Absolute path to the project root (one level up from this file's own
# location), computed automatically -- so saving works correctly no matter
# which folder you were in when you ran the script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "ground_truth_complete.csv")
DEFAULT_FIGURE_PATH = os.path.join(PROJECT_ROOT, "figures", "random_bn_structure.png")

# Fixed variable list + state counts, decided BEFORE any random generation.
# Continuous-like variables: 3-4 states. True categories / counts: 2-3 states.
NODE_STATES = {
    "Age":                     3,
    "EmploymentStatus":        3,   # employed / unemployed / self_employed
    "Income":                  3,
    "DebtRatio":               3,
    "CreditUtilization":       3,
    "PaymentHistory":          4,   # poor / fair / good / excellent
    "NumberOfDefaults":        3,   # 0 / 1 / 2plus
    "LoanAmount":              3,
    "CreditHistoryLength":     3,
    "NumberOfCreditInquiries": 3,
    "Default":                 2,   # target: 0 / 1
}

# Human-readable labels for each variable's states, applied AFTER the random
# structure/CPDs are already finalized (see relabel_states()) -- this is
# purely cosmetic: it changes how a state displays, never which edges exist
# or what any probability equals. Order matters (index 0 keeps whatever
# probability index 0 already had -- only the label changes).
VARIABLE_STATE_LABELS = {
    "Age":                     ["young", "middle", "senior"],
    "EmploymentStatus":        ["employed", "unemployed", "self_employed"],
    "Income":                  ["low", "medium", "high"],
    "DebtRatio":               ["low", "medium", "high"],
    "CreditUtilization":       ["low", "medium", "high"],
    "PaymentHistory":          ["poor", "fair", "good", "excellent"],
    "NumberOfDefaults":        ["0", "1", "2plus"],
    "LoanAmount":              ["low", "medium", "high"],
    "CreditHistoryLength":     ["short", "medium", "long"],
    "NumberOfCreditInquiries": ["low", "medium", "high"],
    "Default":                 ["0", "1"],
}


def relabel_states(model, labels_dict: dict = VARIABLE_STATE_LABELS):
    """
    Replace pgmpy's default integer state labels (0, 1, 2, ...) with
    readable names, for every CPD that mentions each variable (both as the
    CPD's own variable, and anywhere it appears as evidence for another
    variable). This must go through `store_state_names()` rather than
    editing `cpd.state_names` directly -- pgmpy keeps a separate internal
    name<->number lookup table that direct dict edits don't update, which
    would otherwise leave sampling still using the old integer labels.
    """
    for cpd in model.get_cpds():
        new_state_names = dict(cpd.state_names)
        changed = False
        for var in cpd.variables:
            if var in labels_dict:
                new_state_names[var] = labels_dict[var]
                changed = True
        if changed:
            cpd.store_state_names(cpd.variables, cpd.cardinality, new_state_names)
    return model


def generate_random_dag(nodes: list, max_indegree: int, edge_prob: float, seed: int):
    """
    Generate a random DAG over the given node list, respecting a maximum
    in-degree constraint. The random topological order guarantees
    acyclicity: edges only ever point from an earlier node to a later one.
    """
    rng = random.Random(seed)

    topo_order = nodes[:]
    rng.shuffle(topo_order)

    edges = []
    for i, node in enumerate(topo_order):
        possible_parents = topo_order[:i]
        if not possible_parents:
            continue
        max_p = min(max_indegree, len(possible_parents))
        candidates = [p for p in possible_parents if rng.random() < edge_prob]
        if len(candidates) > max_p:
            candidates = rng.sample(candidates, max_p)
        for parent in candidates:
            edges.append((parent, node))

    return edges


def build_random_ground_truth_bn(
    node_states: dict = NODE_STATES,
    max_indegree: int = 3,
    edge_prob: float = 0.4,
    seed: int = 42,
):
    """One-call generator: random structure + random, fully valid CPDs."""
    nodes = list(node_states.keys())
    edges = generate_random_dag(nodes, max_indegree, edge_prob, seed)

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)   # keeps any node with zero edges in the model
    model.add_edges_from(edges)

    model.get_random_cpds(n_states=node_states, inplace=True, seed=seed)

    assert model.check_model(), "Randomly generated CPDs are inconsistent with the DAG structure."
    return model


def check_validity(model) -> dict:
    """Pre-registered structural validity checks (decided before generation)."""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    undirected = G.to_undirected()

    in_degrees = {n: model.in_degree(n) for n in model.nodes()}
    return {
        "is_valid_bn": model.check_model(),
        "is_single_connected_component": nx.is_connected(undirected),
        "n_connected_components": nx.number_connected_components(undirected),
        "isolated_nodes": [n for n in G.nodes() if G.degree(n) == 0],
        "max_in_degree_observed": max(in_degrees.values()),
        "in_degrees": in_degrees,
    }


def sample_synthetic_dataset(model, n_samples: int = 5000, seed: int = 42):
    sampler = BayesianModelSampling(model)
    return sampler.forward_sample(size=n_samples, seed=seed, show_progress=False)


def save_synthetic_dataset(df, save_path: str = DEFAULT_DATA_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved synthetic dataset to: {save_path}")


def plot_bn_structure(model, node_states: dict = NODE_STATES,
                       save_path: str = DEFAULT_FIGURE_PATH):
    """Draw the random ground-truth BN (square nodes, layered by topological
    generation) and save it as a PNG."""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    generations = list(nx.topological_generations(G))
    pos = {}
    for depth, layer in enumerate(generations):
        n = len(layer)
        for j, node in enumerate(sorted(layer)):
            x = (j - (n - 1) / 2) * 3.0
            y = -depth * 1.9
            pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(13, 9))
    patches = {}
    for node in G.nodes():
        x, y = pos[node]
        w, h = 2.3, 0.85
        color = "#ef9a9a" if node == "Default" else "#a7c7e7"
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.4, edgecolor="#333333", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        patches[node] = box
        ax.text(x, y, f"{node}\n({node_states[node]} states)",
                 ha="center", va="center", fontsize=8, zorder=3)

    for src, dst in G.edges():
        arrow = FancyArrowPatch(
            pos[src], pos[dst], patchA=patches[src], patchB=patches[dst],
            arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color="#444444",
            connectionstyle="arc3,rad=0.12", zorder=1,
        )
        ax.add_patch(arrow)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_title("Randomly generated ground-truth BN (11 variables)", fontsize=12)
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved BN structure diagram to: {save_path}")


if __name__ == "__main__":
    model = build_random_ground_truth_bn(
        node_states=NODE_STATES, max_indegree=3, edge_prob=0.4, seed=42
    )
    model = relabel_states(model)

    print("Nodes:", list(model.nodes()))
    print("Edges:", list(model.edges()))
    print()

    checks = check_validity(model)
    for k, v in checks.items():
        print(f"{k}: {v}")

    df = sample_synthetic_dataset(model, n_samples=5000, seed=42)
    print("\nSample of generated data:")
    print(df.head())

    save_synthetic_dataset(df)
    plot_bn_structure(model)

    