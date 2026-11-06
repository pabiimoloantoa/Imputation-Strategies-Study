"""
evaluate_augmented.py -- the 16-vs-16 evaluation over all seeds, sparsity levels
and methods ('perfect' = true values plus R columns, the baseline).

Run from src/experiments with the hash seed set first:
    $env:PYTHONHASHSEED = "0"
    python evaluate_augmented.py
"""

import os
import json
import warnings

import pandas as pd

from augmented_16 import score_dataset, SYN, PROJECT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Settings -------------------------------------------------------------
SEEDS = list(range(20))                     # test with one seed first, then: list(range(20))
SPARSITIES = [10, 20, 30, 40]
METHODS = ["perfect", "em", "mice", "soft"]

RESULTS = PROJECT_ROOT / "results" / "rq1" / "augmented_results.csv"


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
    return set(zip(old.seed.astype(int), old.sparsity.astype(int), old.method))


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
        fb = fallback_used(seed)
        for pct in SPARSITIES:
            for method in METHODS:
                if (seed, pct, method) in done:
                    continue
                try:
                    scores = score_dataset(seed, pct, method)
                except Exception as e:
                    print(f"FAILED seed={seed} sparsity={pct} method={method}: {e}")
                    continue
                save_row({"seed": seed, "sparsity": pct, "method": method,
                          "fallback_used": fb, **scores})
                print(f"seed {seed} | {pct:>3} | {method:<8s} "
                      f"SHD16={scores['cpdag_shd_16']:>3}  SHD11={scores['cpdag_shd_11']:>3}  "
                      f"R found={scores['r_edges_found']}  extra={scores['r_edges_extra']:>2}  "
                      f"own={scores['r_own_links']}  KL16={scores['kl_16']:.4f}")


if __name__ == "__main__":
    main()






