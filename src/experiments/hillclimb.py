#hillclimb.py

import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#HILLCLCIMBING (LEARNING THE NETWORK FROM COMPLETED DATA)
def learn_structure(source, max_indegree=None):
    """source can be a CSV path or a DataFrame."""
    data = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    assert not data.isna().any().any(), "Missing values in the data given to learn_structure"
    data = data.astype("category")
    hc = HillClimbSearch(data)
    dag = hc.estimate(
        scoring_method="bic-d",
        max_indegree=max_indegree,
        show_progress=False,
    )
    return dag

#SHD (MEASURES HOW CLOSE THE LEARNED GRAPH IS TO THE TRUE GRAPH)
def edge_state_map(graph, use_cpdag=True):
    """
    Describe every connected pair of nodes in a graph.
    Returns a dict:  pair -> 'undirected'  or  (parent, child)
    """
    state = {}
    if use_cpdag:
        pdag = graph.to_pdag()          # converts the DAG to its CPDAG
        for a, b in pdag.directed_edges:
            state[frozenset((a, b))] = (a, b)
        for a, b in pdag.undirected_edges:
            state[frozenset((a, b))] = "undirected"
    else:
        for a, b in graph.edges():
            state[frozenset((a, b))] = (a, b)
    return state


def shd(true_graph, learned_graph, use_cpdag=True):
    """
    Structural Hamming Distance: number of node pairs whose
    connection differs between the two graphs.
    A pair counts as 1 error if it is missing in one graph, extra in
    the other, or present in both but with a different direction.
    """
    t = edge_state_map(true_graph, use_cpdag)
    l = edge_state_map(learned_graph, use_cpdag)
    all_pairs = set(t) | set(l)
    return sum(1 for p in all_pairs if t.get(p) != l.get(p))

#PLAIN VERSION KL-DIVERGENCE FUNTION

try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:                       # older pgmpy versions
    from pgmpy.models import BayesianNetwork as BN


def log_joint(model, order, states):
    """
    Log-probability the network gives to EVERY combination of states.
    Returns an array with one axis per variable (in `order`).
    """
    shape = [len(states[n]) for n in order]
    total = np.zeros(shape)
    with np.errstate(divide="ignore"):
        for cpd in model.get_cpds():
            vars_ = cpd.variables                  # [child, parent1, ...]
            vals = cpd.values
            # put each axis into the agreed state order
            for ax, v in enumerate(vars_):
                idx = [cpd.state_names[v].index(s) for s in states[v]]
                vals = np.take(vals, idx, axis=ax)
            # line the axes up with the global variable order
            pos = [order.index(v) for v in vars_]
            vals = np.transpose(vals, np.argsort(pos))
            bshape = [1] * len(order)
            for p in pos:
                bshape[p] = shape[p]
            total = total + np.log(vals.reshape(bshape))
    return total


def kl_divergence(true_model, learned_dag, data):
    """
    KL(true || learned), in nats. 0 = identical behaviour.
    Fits CPTs onto learned_dag's structure using `data`
    (K2 prior = a pseudo-count of 1 in every cell).
    """
    order = list(true_model.nodes())
    states = {n: list(true_model.get_cpds(n).state_names[n]) for n in order}

    # all labels as text, so they match the .bif state names
    data = data[order].astype(str)

    # 1. fit the learned network
    learned = BN()
    learned.add_nodes_from(order)
    learned.add_edges_from(learned_dag.edges())
    estimator = BayesianEstimator(learned, data, state_names=states)
    cpds = [estimator.estimate_cpd(n, prior_type="K2") for n in order]
    learned.add_cpds(*cpds)

    # 2. joint distribution of each network
    logp = log_joint(true_model, order, states)   # true
    logq = log_joint(learned, order, states)      # learned
    p = np.exp(logp)
    assert abs(p.sum() - 1) < 1e-6, "True joint does not sum to 1"

    # 3. KL = sum P * (log P - log Q), over combinations where P > 0
    mask = p > 0
    return float(np.sum(p[mask] * (logp[mask] - logq[mask])))



if __name__ == "__main__":
    # test code goes here (everything from TRUE_BIF = ... down to the prints)
    TRUE_BIF = "../../data/synthetic/models/model_seed0.bif"
    CSV = "../../data/synthetic/ground_truth_per_seed/ground_truth_seed0.csv"

    true_model = BIFReader(TRUE_BIF).get_model()
    true_edges = set(true_model.edges())

    dag = learn_structure(CSV)
    learned_edges = set(dag.edges())

    same_dir = true_edges & learned_edges

    true_skel = {frozenset(e) for e in true_edges}
    learned_skel = {frozenset(e) for e in learned_edges}
    same_skel = true_skel & learned_skel

    print("True edges:", len(true_edges))
    print("Learned edges:", len(learned_edges))
    print("Correct direction:", len(same_dir))
    print("Right connection (any direction):", len(same_skel))
    print("Missed:", sorted(tuple(e) for e in true_skel - learned_skel))
    print("Extra:", sorted(tuple(e) for e in learned_skel - true_skel))


    print("Raw SHD (DAGs):", shd(true_model, dag, use_cpdag=False))
    print("CPDAG SHD:", shd(true_model, dag, use_cpdag=True))

    t = edge_state_map(true_model)
    l = edge_state_map(dag)
    print("True CPDAG: undirected edges =", sum(v == "undirected" for v in t.values()), "of", len(t))
    print("Learned CPDAG: undirected edges =", sum(v == "undirected" for v in l.values()), "of", len(l))


    # FOR KL
    raw = pd.read_csv(CSV)
    print("KL, learned graph:", kl_divergence(true_model, dag, raw))
    print("KL, true graph (noise floor):", kl_divergence(true_model, true_model, raw))


    