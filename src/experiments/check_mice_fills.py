"""
check_mice_fills.py  -  how good are the fills, cell by cell?

For every seed (0-19) and sparsity (10-40%) it looks only at the cells that were blanked and compares
what each method put there with the TRUE value from the complete data:

  accuracy : share of blanked cells where the filled state equals the true state
  tvd      : total variation distance between the mix of filled states and the mix of true states
             in those cells (0 = identical mix, 1 = completely different).
             This is the number behind your Income example, where EM under-filled "high".

Reference row "mode_fill": every blank filled with that variable's most common observed state.

Reads  data/synthetic/ground_truth_per_seed, missing, em_completed, mice_completed, soft_completed
Writes results/rq1/fill_diagnostics.csv  and prints summary tables.
Only reads existing files; nothing is changed.

Place in <project>/src/experiments/ (or src/experiment/) and run:   python check_mice_fills.py

Optional: add one more method to the comparison, e.g. the depth-3 MICE rerun:
    python check_mice_fills.py mice_depth3
That restricts the tables to the seeds that have those files, and saves to fill_diagnostics_mice_depth3.csv
instead of overwriting the main results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic"
EXTRA = sys.argv[1] if len(sys.argv) > 1 else None      # e.g. "mice_depth3"
OUT_CSV = ROOT / "results" / "rq1" / (f"fill_diagnostics_{EXTRA}.csv" if EXTRA else "fill_diagnostics.csv")
SEEDS = range(20)
LEVELS = [10, 20, 30, 40]
METHODS = ["em", "mice", "soft"]
TRUTH_FILE = "ground_truth_per_seed/ground_truth_seed{seed}.csv"
MISSING_FILE = "missing/missing_{pct}pct_seed{seed}.csv"
COMPLETED_FILE = "{m}_completed/{m}_completed_{pct}pct_seed{seed}.csv"
# ----------------------------------------------------------------------------

LABEL = {"mode_fill": "Mode fill (reference)", "em": "EM", "mice": "MICE", "soft": "Soft-Impute"}
ORDER = ["mode_fill", "em", "mice", "soft"]
if EXTRA:
    METHODS = METHODS + [EXTRA]
    LABEL[EXTRA] = EXTRA
    ORDER = ORDER + [EXTRA]


def read_clean(path):
    """Read as text and turn '1.0' into '1' so Default compares equal across files."""
    df = pd.read_csv(path, dtype=str)
    return df.apply(lambda s: s.str.replace(r"\.0+$", "", regex=True))


def tvd(a, b):
    states = a.index.union(b.index)
    return 0.5 * float(np.abs(a.reindex(states, fill_value=0) - b.reindex(states, fill_value=0)).sum())


def main():
    rows, skipped = [], []
    for seed in SEEDS:
        for pct in LEVELS:
            paths = {"truth": DATA / TRUTH_FILE.format(seed=seed),
                     "miss": DATA / MISSING_FILE.format(pct=pct, seed=seed)}
            for m in METHODS:
                paths[m] = DATA / COMPLETED_FILE.format(m=m, pct=pct, seed=seed)
            if EXTRA and not paths[EXTRA].exists():
                continue                      # this seed/level was not rerun; leave it out quietly
            absent = [k for k, p in paths.items() if not p.exists()]
            if absent:
                skipped.append((seed, pct, absent))
                continue
            truth, miss = read_clean(paths["truth"]), read_clean(paths["miss"])
            comp = {m: read_clean(paths[m]) for m in METHODS}
            if any(len(d) != len(truth) for d in [miss, *comp.values()]):
                raise SystemExit(f"Row counts differ for seed {seed}, {pct}%. Files are not aligned.")
            blank = miss.isna()
            for var in [c for c in miss.columns if blank[c].any()]:
                b = blank[var]
                t = truth.loc[b, var]
                true_mix = t.value_counts(normalize=True)
                mode_state = miss.loc[~b, var].mode().iloc[0]
                fills = {"mode_fill": pd.Series(mode_state, index=t.index)}
                fills.update({m: comp[m].loc[b, var] for m in METHODS})
                for name, f in fills.items():
                    rows.append(dict(seed=seed, sparsity=pct, variable=var, method=name,
                                     n_blank=int(b.sum()),
                                     accuracy=float((f == t).mean()),
                                     tvd=tvd(true_mix, f.value_counts(normalize=True))))
    if not rows:
        raise SystemExit(f"No complete sets of files found under {DATA}. First problem: {skipped[:3]}")
    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    n_sets = res[["seed", "sparsity"]].drop_duplicates().shape[0]
    if EXTRA:
        print(f"Restricted to the {n_sets} seed/level combinations that have {EXTRA} files; "
              f"{len(skipped)} skipped for other missing files.")
    else:
        print(f"Used {n_sets} of {len(SEEDS) * len(LEVELS)} seed/level combinations; "
              f"{len(skipped)} skipped for missing files.")
    if skipped:
        print("  first skipped:", skipped[:3])
    print("Each number is a mean over seeds and over the variables that had blanks.\n")

    def table(col, by, title, fmt):
        t = res.pivot_table(index="method", columns=by, values=col, aggfunc="mean").reindex(ORDER)
        t.index = [LABEL[i] for i in t.index]
        print(title)
        print(t.to_string(float_format=lambda x: format(x, fmt)))
        print()

    table("accuracy", "sparsity", "Accuracy in blanked cells, by sparsity (higher = better):", ".3f")
    table("tvd", "sparsity", "Mix distance (TVD) in blanked cells, by sparsity (lower = better):", ".3f")
    table("accuracy", "variable", "Accuracy in blanked cells, by variable (all levels pooled):", ".3f")
    table("tvd", "variable", "Mix distance (TVD) in blanked cells, by variable (all levels pooled):", ".3f")
    print(f"Per-variable, per-seed rows saved to {OUT_CSV}")


if __name__ == "__main__":
    main()