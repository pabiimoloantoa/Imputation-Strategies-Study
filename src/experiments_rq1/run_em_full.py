"""
run_em_full.py  -  EM imputation with the FULL M-step, for every seed and sparsity level.

Place in <project>/src/experiments/ next to em_impute.py (the fixed one), encode_decode.py
and random_bn.py, then run:

    python run_em_full.py                 # all 20 seeds x 4 levels = 80 files
    python run_em_full.py --seeds 0,1     # only some seeds (for a quick test)
    python run_em_full.py --old           # re-run the ORIGINAL EM (no conditional covariance)
                                          # into a separate folder, to check the pipeline reproduces
                                          # your existing em_completed files

Reads   data/synthetic/missing/missing_<pct>pct_seed<N>.csv
Writes  data/synthetic/em_full_completed/em_full_completed_<pct>pct_seed<N>.csv
        (or em_old_completed/em_old_completed_... with --old)
        results/rq1/em_full_log.csv   (iterations, convergence, log-likelihood check)

Your existing em_completed files are NOT touched.
Safe to stop and rerun: files that already exist are skipped.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from encode_decode import encode, decode
from em_impute import reduce_columns, restore_columns, em_impute
from random_bn import VARIABLE_STATE_LABELS

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
LEVELS = [10, 20, 30, 40]
MAX_ITER = 100
TOL = 1e-4


def observed_cells_unchanged(df_original: pd.DataFrame, decoded: pd.DataFrame) -> bool:
    """Every cell that was observed must come back exactly as it was."""
    for col in df_original.columns:
        mask = df_original[col].notna()
        original = df_original.loc[mask, col]
        if pd.api.types.is_numeric_dtype(original):
            original = original.astype("Int64").astype("string")
        if not (decoded.loc[mask, col] == original).all():
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(20)),
                    help="comma-separated seeds, default 0..19")
    ap.add_argument("--old", action="store_true",
                    help="use the original M-step (no conditional covariance) instead of the full one")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    full = not args.old
    tag = "em_full" if full else "em_old"
    out_dir = SYN / f"{tag}_completed"
    log_path = ROOT / "results" / "rq1" / f"{tag}_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"EM variant: {'FULL M-step' if full else 'ORIGINAL (no conditional covariance)'}  ->  {out_dir}")

    log_rows = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    done_log = {(r["seed"], r["sparsity"]) for r in log_rows}

    bad = []
    for seed in seeds:
        for pct in LEVELS:
            src = SYN / "missing" / f"missing_{pct}pct_seed{seed}.csv"
            dst = out_dir / f"{tag}_completed_{pct}pct_seed{seed}.csv"
            if dst.exists() and (seed, pct) in done_log:
                continue
            if not src.exists():
                print(f"  MISSING INPUT, skipped: {src.name}")
                continue

            df = pd.read_csv(src)
            encoded = encode(df)
            reduced, dropped = reduce_columns(encoded, VARIABLE_STATE_LABELS)
            completed_reduced, info = em_impute(reduced, max_iter=MAX_ITER, tol=TOL,
                                                full_mstep=full, return_info=True)
            completed_full = restore_columns(completed_reduced, VARIABLE_STATE_LABELS, dropped)
            decoded = decode(completed_full)

            ok = observed_cells_unchanged(df, decoded)
            ll = np.array(info["logliks"])
            monotone = bool(np.all(np.diff(ll) >= -1e-6)) if len(ll) > 1 else True
            decoded.to_csv(dst, index=False)

            log_rows = [r for r in log_rows if not (r["seed"] == seed and r["sparsity"] == pct)]
            log_rows.append({"seed": seed, "sparsity": pct, "variant": tag,
                             "n_iter": info["n_iter"], "converged": info["converged"],
                             "loglik_never_decreased": monotone, "observed_cells_ok": ok,
                             "final_loglik": float(ll[-1])})
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

            flag = "" if ok else "   <-- OBSERVED CELLS CHANGED"
            if not ok:
                bad.append((seed, pct))
            print(f"seed {seed:>2}  {pct}%  iterations={info['n_iter']:>3}  "
                  f"converged={info['converged']}  loglik never decreased={monotone}{flag}")

    if bad:
        raise SystemExit(f"Observed cells were changed in {bad}. Do not use these files.")
    print("\nDone. Log:", log_path)
    print("Next: run evaluate_em_full.py with PYTHONHASHSEED=0.")


if __name__ == "__main__":
    main()