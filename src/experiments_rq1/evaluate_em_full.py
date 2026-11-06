"""
evaluate_em_full.py  -  Hill Climbing + scoring for the rerun EM files only.

Runs BOTH evaluations (16-node and plain 11-variable) on the files written by run_em_full.py,
with exactly the same scoring code as before (augmented_16.score_dataset and evaluate.evaluate).

Place in <project>/src/experiments/ and run with the hash seed set FIRST:

    $env:PYTHONHASHSEED = "0"
    python evaluate_em_full.py                  # all 20 seeds x 4 levels
    python evaluate_em_full.py --seeds 0,1      # a few seeds only
    python evaluate_em_full.py --method em_old  # score the em_old files instead

Writes  results/rq1/augmented_results_em_full.csv   (same columns as augmented_results.csv)
        results/rq1/plain_results_em_full.csv       (same columns as plain_results.csv)
Your original result files are NOT touched. Safe to stop and rerun: finished rows are skipped.

Before scoring anything it re-scores one dataset that already exists in augmented_results.csv
(seed 0, 10%, perfect baseline) and stops if the number differs: that would mean the hash seed or
the library versions are not the ones used for the main results.
"""
import argparse
import json
import os
import warnings

import pandas as pd
from pgmpy.readwrite import BIFReader

from augmented_16 import score_dataset, SYN, PROJECT_ROOT
from evaluate import load_dataset, evaluate

warnings.filterwarnings("ignore", category=FutureWarning)

LEVELS = [10, 20, 30, 40]
RES = PROJECT_ROOT / "results" / "rq1"


def fallback_used(seed):
    try:
        with open(SYN / "audit_logs" / f"audit_seed{seed}.json") as f:
            return json.load(f)["fallback_used"]
    except Exception:
        return None


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def environment_check():
    main = RES / "augmented_results.csv"
    if not main.exists():
        print("(augmented_results.csv not found, skipping the environment check)")
        return
    ref = pd.read_csv(main)
    ref = ref[(ref.seed == 0) & (ref.sparsity == 10) & (ref.method == "perfect")]
    if ref.empty:
        print("(seed 0, 10%, perfect not in augmented_results.csv, skipping the environment check)")
        return
    now = score_dataset(0, 10, "perfect")
    same = abs(now["kl_16"] - float(ref.kl_16.iloc[0])) < 1e-9 and \
        now["cpdag_shd_16"] == int(ref.cpdag_shd_16.iloc[0])
    print("Environment check (seed 0, 10%, perfect): ", "OK, reproduces the main results" if same else "MISMATCH")
    if not same:
        raise SystemExit("The scores do not reproduce your main results. Check PYTHONHASHSEED=0 and the pgmpy version.")


def main():
    if "PYTHONHASHSEED" not in os.environ:
        raise SystemExit('Set PYTHONHASHSEED first, e.g.  $env:PYTHONHASHSEED = "0"')
    print("PYTHONHASHSEED =", os.environ["PYTHONHASHSEED"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(20)))
    ap.add_argument("--method", default="em_full")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    method = args.method

    environment_check()

    aug_path = RES / f"augmented_results_{method}.csv"
    plain_path = RES / f"plain_results_{method}.csv"
    aug_done = set()
    if aug_path.exists():
        old = pd.read_csv(aug_path)
        aug_done = set(zip(old.seed.astype(int), old.sparsity.astype(int)))
    plain_done = set()
    if plain_path.exists():
        old = pd.read_csv(plain_path)
        plain_done = set(zip(old.seed.astype(int), old.sparsity.astype(int)))

    for seed in seeds:
        fb = fallback_used(seed)
        true_model = BIFReader(str(SYN / "models" / f"model_seed{seed}.bif")).get_model()
        for pct in LEVELS:
            if (seed, pct) not in aug_done:
                try:
                    s = score_dataset(seed, pct, method)
                    append_row(aug_path, {"seed": seed, "sparsity": pct, "method": method,
                                          "fallback_used": fb, **s})
                    print(f"16-node  seed {seed:>2} | {pct}% | SHD16={s['cpdag_shd_16']:>3} "
                          f"R found={s['r_edges_found']} extra={s['r_edges_extra']:>2} "
                          f"own={s['r_own_links']} KL16={s['kl_16']:.4f}")
                except Exception as e:
                    print(f"FAILED 16-node seed={seed} {pct}%: {e}")
            if (seed, pct) not in plain_done:
                try:
                    data = load_dataset(seed, pct, method)
                    s = evaluate(data, true_model)
                    append_row(plain_path, {"seed": seed, "sparsity": pct, "method": method,
                                            "fallback_used": fb, **s})
                    print(f"plain    seed {seed:>2} | {pct}% | SHD={s['cpdag_shd']:>3} KL={s['kl']:.4f}")
                except Exception as e:
                    print(f"FAILED plain seed={seed} {pct}%: {e}")

    print("\nDone.", aug_path.name, "and", plain_path.name, "are in", RES)
    print("For the fill diagnostics (accuracy and mix distance) run:  python check_mice_fills.py", method)


if __name__ == "__main__":
    main()