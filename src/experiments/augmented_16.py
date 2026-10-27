"""
augmented.py -- helpers for the 16-vs-16 evaluation.

$env:PYTHONHASHSEED = "0"
python augmented_16.py

"""

from pathlib import Path

import os
import time

import pandas as pd
from pgmpy.base import DAG
from pgmpy.readwrite import BIFReader

from hillclimb import learn_structure, shd, kl_divergence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYN = PROJECT_ROOT / "data" / "synthetic"


# ------------------------------------------------ step 1: the true network
def load_trimmed_truth(seed, pct):
    """
    Load the true network with R-nodes (all sparsity levels) and keep only
    the R-nodes of one level (11 original + 5 R-nodes = 16 nodes).
    """
    path = SYN / "models" / f"model_with_r_nodes_seed{seed}.bif"
    model = BIFReader(str(path)).get_model()

    suffix = f"_{pct}pct"
    for node in list(model.nodes()):
        if node.startswith("R_") and not node.endswith(suffix):
            model.remove_node(node)     # R-nodes are leaves, so nothing else changes
    return model


# --------------------------------------------- step 2: the learning table
def dataset_path(seed, pct, method):
    """'perfect' = the true complete data (no imputation errors)."""
    if method == "perfect":
        return SYN / "ground_truth_per_seed" / f"ground_truth_seed{seed}.csv"
    return SYN / f"{method}_completed" / f"{method}_completed_{pct}pct_seed{seed}.csv"


def build_learning_table(seed, pct, method, truth):
    """
    The 11 completed columns plus one R column per Vm variable.
    R columns are rebuilt from the blanks in the missing file:
    'missing' where the cell was blank, 'observed' otherwise.
    Column names come from the true network's R-nodes, so they match exactly.
    """
    completed = pd.read_csv(dataset_path(seed, pct, method), dtype=str)
    missing = pd.read_csv(SYN / "missing" / f"missing_{pct}pct_seed{seed}.csv", dtype=str)

    assert len(completed) == len(missing), "Row counts differ between the two files"
    assert not completed.isna().any().any(), "Completed data still has blanks"

    r_nodes = [n for n in truth.nodes() if n.startswith("R_")]
    suffix = f"_{pct}pct"
    vm_vars = [r[2:-len(suffix)] for r in r_nodes]

    # only the Vm variables should have blanks in the missing file
    blank_cols = set(missing.columns[missing.isna().any()])
    assert blank_cols == set(vm_vars), f"Blank columns {blank_cols} != Vm variables {set(vm_vars)}"

    table = completed.copy()
    for r, var in zip(r_nodes, vm_vars):
        table[r] = missing[var].isna().map({True: "missing", False: "observed"})
    return table


def observed_cells_preserved(seed, pct, method, truth):
    """Diagnostic: share of originally observed Vm cells the imputation left unchanged."""
    completed = pd.read_csv(dataset_path(seed, pct, method), dtype=str)
    missing = pd.read_csv(SYN / "missing" / f"missing_{pct}pct_seed{seed}.csv", dtype=str)
    suffix = f"_{pct}pct"
    vm_vars = [n[2:-len(suffix)] for n in truth.nodes() if n.startswith("R_")]
    same = total = 0
    for v in vm_vars:
        seen = missing[v].notna()
        same += (completed.loc[seen, v] == missing.loc[seen, v]).sum()
        total += seen.sum()
    return same / total


# ------------------------------------------------ step 3: score one dataset
def keep_original_only(graph):
    """Copy of a graph with the R-nodes (and their edges) removed."""
    keep = [n for n in graph.nodes() if not n.startswith("R_")]
    sub = DAG()
    sub.add_nodes_from(keep)
    sub.add_edges_from(e for e in graph.edges() if e[0] in keep and e[1] in keep)
    return sub


def score_dataset(seed, pct, method):
    """Learn a 16-node network from one dataset and score it against the truth."""
    truth = load_trimmed_truth(seed, pct)
    table = build_learning_table(seed, pct, method, truth)

    dag = learn_structure(table)                      # Hill Climbing + BIC on 16 columns

    # how well was the missingness mechanism (Vo -> R edges) found?
    true_r = {frozenset(e) for e in truth.edges() if e[1].startswith("R_")}
    learned_r = {frozenset(e) for e in dag.edges() if any(n.startswith("R_") for n in e)}

    # the footprint signal: R-nodes linked to their OWN variable (should be 0)
    suffix = f"_{pct}pct"
    r_nodes = [n for n in truth.nodes() if n.startswith("R_")]
    own_links = sum(frozenset((r, r[2:-len(suffix)])) in learned_r for r in r_nodes)

    truth11, dag11 = keep_original_only(truth), keep_original_only(dag)
    return {
        "n_rows": len(table),
        "n_edges": len(dag.edges()),
        "cpdag_shd_16": shd(truth, dag, use_cpdag=True),
        "raw_shd_16": shd(truth, dag, use_cpdag=False),
        "cpdag_shd_11": shd(truth11, dag11, use_cpdag=True),
        "raw_shd_11": shd(truth11, dag11, use_cpdag=False),
        "r_edges_found": len(true_r & learned_r),
        "r_edges_extra": len(learned_r - true_r),
        "r_own_links": own_links,
        "r_extra_pairs": "; ".join(sorted("-".join(sorted(p)) for p in learned_r - true_r)),
        "kl_16": kl_divergence(truth, dag, table),
        "learned_edges": ";".join(f"{a}>{b}" for a, b in sorted(dag.edges())),
    }


if __name__ == "__main__":
    if "PYTHONHASHSEED" not in os.environ:
        raise SystemExit("Set PYTHONHASHSEED first, e.g.  $env:PYTHONHASHSEED = '0'")

    TESTS = [(11, 10), (7, 40)]            # (seed, sparsity %)
    METHODS = ["perfect", "em", "mice", "soft"]

    for seed, pct in TESTS:
        print(f"\n=== seed {seed}, {pct}% ===")
        print(f"{'method':<9}{'edges':>6}{'shd16':>7}{'shd11':>7}{'R found':>9}{'R extra':>9}{'kl16':>9}")
        extras = {}
        for method in METHODS:
            s = score_dataset(seed, pct, method)
            print(f"{method:<9}{s['n_edges']:>6}{s['cpdag_shd_16']:>7}{s['cpdag_shd_11']:>7}"
                  f"{s['r_edges_found']:>9}{s['r_edges_extra']:>9}{s['kl_16']:>9.3f}")
            extras[method] = s["r_extra_pairs"]
        for method in METHODS:
            if extras[method]:
                print(f"  extra R-links ({method}): {extras[method]}")
