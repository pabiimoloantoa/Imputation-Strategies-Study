"""
evaluate.py -- runs Hill Climbing + SHD + KL on every completed dataset
(plain 11-variable version).

Run from src/experiments with the hash seed set first:
$env:PYTHONHASHSEED = "0"
python evaluate.py
"""

import os
import json
import warnings
from pathlib import Path

import pandas as pd
from pgmpy.readwrite import BIFReader

from hillclimb import learn_structure, shd, kl_divergence

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Settings -------------------------------------------------------------
SEEDS = list(range(20))                       # test with one seed first, then: list(range(20))
SPARSITIES = [10, 20, 30, 40]
METHODS = ["em", "mice", "soft"]  # folders: em_completed, mice_completed, soft_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYN = PROJECT_ROOT / "data" / "synthetic"
RESULTS = PROJECT_ROOT / "results" / "rq1" / "plain_results.csv"

# One seed's worth of work: the complete-data ceiling, then for each sparsity
# level the listwise-deletion floor and the three imputation methods.
JOBS = [("none", "complete")] + [
    (pct, m) for pct in SPARSITIES for m in ["listwise"] + METHODS
]


def load_dataset(seed, sparsity, method):
    # dtype=str keeps every label as text, so Default stays "0"/"1".
    # (Otherwise pandas turns it into 0.0/1.0 in files that have blanks.)
    if method == "complete":
        path = SYN / "ground_truth_per_seed" / f"ground_truth_seed{seed}.csv"
        return pd.read_csv(path, dtype=str)
    if method == "listwise":
        path = SYN / "missing" / f"missing_{sparsity}pct_seed{seed}.csv"
        return pd.read_csv(path, dtype=str).dropna()
    path = SYN / f"{method}_completed" / f"{method}_completed_{sparsity}pct_seed{seed}.csv"
    return pd.read_csv(path, dtype=str)


def check_states(data, true_model):
    """Stop early if a column holds labels the true network doesn't know."""
    for col in true_model.nodes():
        allowed = set(true_model.get_cpds(col).state_names[col])
        found = set(data[col].astype(str).unique())
        if not found <= allowed:
            raise ValueError(f"{col}: unexpected states {found - allowed}")


def evaluate(data, true_model):
    check_states(data, true_model)
    dag = learn_structure(data)
    return {
        "n_rows": len(data),
        "n_edges": len(dag.edges()),
        "cpdag_shd": shd(true_model, dag, use_cpdag=True),
        "raw_shd": shd(true_model, dag, use_cpdag=False),
        "kl": kl_divergence(true_model, dag, data),
    }


def fallback_used(seed):
    try:
        with open(SYN / "audit_logs" / f"audit_seed{seed}.json") as f:
            return json.load(f)["fallback_used"]
    except Exception:
        return None


def done_keys():
    if not RESULTS.exists():
        return set()
    old = pd.read_csv(RESULTS)
    return set(zip(old.seed.astype(int), old.sparsity.astype(str), old.method))


def save_row(row):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        RESULTS, mode="a", header=not RESULTS.exists(), index=False
    )


def main():
    if "PYTHONHASHSEED" not in os.environ:
        raise SystemExit("Set PYTHONHASHSEED first, e.g.  $env:PYTHONHASHSEED = '0'")
    print("PYTHONHASHSEED =", os.environ["PYTHONHASHSEED"])

    done = done_keys()
    for seed in SEEDS:
        true_model = BIFReader(str(SYN / "models" / f"model_seed{seed}.bif")).get_model()
        fb = fallback_used(seed)
        for sparsity, method in JOBS:
            if (seed, str(sparsity), method) in done:
                continue
            try:
                data = load_dataset(seed, sparsity, method)
                scores = evaluate(data, true_model)
            except Exception as e:
                print(f"FAILED seed={seed} sparsity={sparsity} method={method}: {e}")
                continue
            save_row({"seed": seed, "sparsity": sparsity, "method": method,
                      "fallback_used": fb, **scores})
            print(f"seed {seed} | {str(sparsity):>4} | {method:<9s} "
                  f"CPDAG SHD={scores['cpdag_shd']:>3}  KL={scores['kl']:.4f}")


if __name__ == "__main__":
    main()



