"""
run_mice_shallow.py  -  rerun MICE with a depth-limited tree on a few seeds.

Purpose: test whether MICE's clean result depends on its fully grown trees.
Same code, same settings as run_mice_pipeline.py (10 iterations, random_state 0);
the ONLY change is max_depth (default 3 here instead of None).

Output goes to a NEW folder, so your existing mice_completed files are not touched:
    data/synthetic/mice_depth3_completed/mice_depth3_completed_<pct>pct_seed<N>.csv

Put this file in the same folder as mice_impute.py, encode_decode.py and random_bn.py
(the folder you run run_mice_pipeline.py from) and run:

    python run_mice_shallow.py

Safe to stop and rerun: files that already exist are skipped.
"""
import time
from pathlib import Path

import pandas as pd

from mice_impute import mice_impute
from random_bn import VARIABLE_STATE_LABELS

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
LEVELS = [10, 20, 30, 40]
MAX_DEPTH = 3
N_ITERATIONS = 10
RANDOM_STATE = 0
ROOT = Path(__file__).resolve().parents[2]
MISSING_DIR = ROOT / "data" / "synthetic" / "missing"
TAG = f"mice_depth{MAX_DEPTH}"
OUT_DIR = ROOT / "data" / "synthetic" / f"{TAG}_completed"
# ----------------------------------------------------------------------------


def observed_cells_unchanged(original: pd.DataFrame, completed: pd.DataFrame) -> bool:
    """Every cell that was observed must come back unchanged ('1.0' and '1' count as equal)."""
    for col in original.columns:
        m = original[col].notna()
        a = original.loc[m, col].astype(str).str.replace(r"\.0+$", "", regex=True)
        b = completed.loc[m, col].astype(str).str.replace(r"\.0+$", "", regex=True)
        if not (a == b).all():
            return False
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"MICE with max_depth={MAX_DEPTH}; output folder: {OUT_DIR}")
    done = skipped = 0
    for seed in SEEDS:
        for pct in LEVELS:
            src = MISSING_DIR / f"missing_{pct}pct_seed{seed}.csv"
            dst = OUT_DIR / f"{TAG}_completed_{pct}pct_seed{seed}.csv"
            if dst.exists():
                skipped += 1
                continue
            if not src.exists():
                print(f"  missing input, skipped: {src.name}")
                continue
            df = pd.read_csv(src)
            t0 = time.time()
            completed = mice_impute(df, VARIABLE_STATE_LABELS, n_iterations=N_ITERATIONS,
                                    random_state=RANDOM_STATE, max_depth=MAX_DEPTH)
            ok = observed_cells_unchanged(df, completed)
            completed.to_csv(dst, index=False)
            done += 1
            print(f"  seed {seed:>2}, {pct}%: saved in {time.time() - t0:.1f}s, observed cells unchanged: {ok}")
    print(f"Finished: {done} new files, {skipped} already existed.")


if __name__ == "__main__":
    main()

