"""
noise_floor.py -- how much do Hill Climbing's scores wobble with the hash seed?

Runs the 16-node scoring (score_dataset) on the same datasets under several
PYTHONHASHSEED values. Python only reads the hash seed at start-up, so this
script launches a fresh Python process for each value.

Run from src/experiments (you do NOT need to set the hash seed yourself):
    python noise_floor.py
Results: results/rq1/noise_floor.csv   (can be stopped and restarted)
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

from augmented_16 import score_dataset, PROJECT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Settings -------------------------------------------------------------
SEEDS = list(range(20))
HASH_SEEDS = list(range(10))          # 0 is the value used for all earlier results
SPARSITIES = [20]
METHODS = ["perfect"]                 # later you can add "em", "mice", "soft"

RESULTS = PROJECT_ROOT / "results" / "rq1" / "noise_floor.csv"
KEEP = ["n_edges", "cpdag_shd_16", "cpdag_shd_11",
        "r_edges_found", "r_edges_extra", "r_own_links", "kl_16"]


def done_keys():
    if not RESULTS.exists():
        return set()
    old = pd.read_csv(RESULTS)
    return set(zip(old.hash_seed.astype(int), old.seed.astype(int),
                   old.sparsity.astype(int), old.method))


def save_row(row):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        RESULTS, mode="a", header=not RESULTS.exists(), index=False
    )


def worker(hash_seed):
    """Runs inside a Python process started with PYTHONHASHSEED=hash_seed."""
    assert os.environ.get("PYTHONHASHSEED") == str(hash_seed), "hash seed not set correctly"
    done = done_keys()
    for seed in SEEDS:
        for pct in SPARSITIES:
            for method in METHODS:
                if (hash_seed, seed, pct, method) in done:
                    continue
                try:
                    scores = score_dataset(seed, pct, method)
                except Exception as e:
                    print(f"  FAILED hash={hash_seed} seed={seed} {pct}% {method}: {e}")
                    continue
                save_row({"hash_seed": hash_seed, "seed": seed, "sparsity": pct,
                          "method": method, **{k: scores[k] for k in KEEP}})


def main():
    for h in HASH_SEEDS:
        print(f"Hash seed {h} ...", flush=True)
        env = dict(os.environ, PYTHONHASHSEED=str(h))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(h)],
                       env=env, check=False)
    print("Done. Results in", RESULTS)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()









