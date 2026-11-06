import os
import random

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "ground_truth_complete.csv")
DEFAULT_FIGURE_PATH = os.path.join(PROJECT_ROOT, "figures", "random_bn_structure.png")

NODE_STATES = {
    "Age":                     3,
    "EmploymentStatus":        3,
    "Income":                  3,
    "DebtRatio":               3,
    "CreditUtilization":       3,
    "PaymentHistory":          4,
    "NumberOfDefaults":        3,
    "LoanAmount":              3,
    "CreditHistoryLength":     3,
    "NumberOfCreditInquiries": 3,
    "Default":                 2,
}

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

SINK_NODES = ["Default"]


def relabel_states(model, labels_dict: dict = VARIABLE_STATE_LABELS):
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


def generate_random_dag(nodes: list, max_indegree: int, edge_prob: float, seed: int,
                         sink_nodes: list = None):
    
    rng = random.Random(seed)
    sink_nodes = sink_nodes or []

    non_sink = [n for n in nodes if n not in sink_nodes]
    rng.shuffle(non_sink)
    sink_shuffled = sink_nodes[:]
    rng.shuffle(sink_shuffled)
    topo_order = non_sink + sink_shuffled  # sinks always placed last

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
    sink_nodes: list = SINK_NODES,
):
    nodes = list(node_states.keys())
    edges = generate_random_dag(nodes, max_indegree, edge_prob, seed, sink_nodes=sink_nodes)
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    model.get_random_cpds(n_states=node_states, inplace=True, seed=seed)
    assert model.check_model(), "Randomly generated CPDs are inconsistent with the DAG structure."
    return model


def check_validity(model) -> dict:
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


