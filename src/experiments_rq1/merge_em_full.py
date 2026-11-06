"""
merge_em_full.py  -  swap the rerun EM (full M-step) into your results files, so that every figure and
table script keeps working unchanged (they all look for the method name "em").

Run from src/experiments:   python merge_em_full.py

What it does
  1. Copies your current files to results/rq1/original_em_backup/ (only the first time it is run).
  2. In augmented_results.csv, plain_results.csv and fill_diagnostics.csv it replaces every row whose
     method is "em" by the matching row from the *_em_full.csv files, renamed to "em".
  3. Saves the ORIGINAL EM rows to  results/rq1/original_em_backup/*_em_original.csv  for the old-versus-new comparison.

It stops without changing anything if the rerun files are incomplete.
To undo: copy the files in original_em_backup/ back over the ones in results/rq1/.
"""
import shutil
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results" / "rq1"
BACKUP = RES / "original_em_backup"

FILES = {
    # file to update      : (rerun file,                         key columns, expected rows for "em")
    "augmented_results.csv": ("augmented_results_em_full.csv", ["seed", "sparsity"], 80),
    "plain_results.csv":     ("plain_results_em_full.csv",     ["seed", "sparsity"], 80),
}


def main():
    BACKUP.mkdir(exist_ok=True)

    # ---- read everything first, so nothing is written if something is wrong -------------------
    new = {}
    for name, (rerun, keys, n_expected) in FILES.items():
        main_df = pd.read_csv(RES / name, dtype={"sparsity": str})
        full = pd.read_csv(RES / rerun, dtype={"sparsity": str})
        if len(full) != n_expected or full.duplicated(keys).any():
            raise SystemExit(f"{rerun}: expected {n_expected} unique rows, found {len(full)}. Not merged.")
        old_em = main_df[main_df.method == "em"]
        if len(old_em) != n_expected:
            raise SystemExit(f"{name}: expected {n_expected} 'em' rows, found {len(old_em)}. Not merged.")
        full = full.assign(method="em")[main_df.columns]
        new[name] = (pd.concat([main_df[main_df.method != "em"], full], ignore_index=True)
                     .sort_values(keys + ["method"], key=lambda s: s.astype(str) if s.name == "sparsity" else s)
                     .reset_index(drop=True), old_em, main_df)

    fd = pd.read_csv(RES / "fill_diagnostics.csv")
    fd_full = pd.read_csv(RES / "fill_diagnostics_em_full.csv")
    fd_new_em = fd_full[fd_full.method == "em_full"].assign(method="em")
    fd_old_em = fd[fd.method == "em"]
    if len(fd_new_em) != len(fd_old_em):
        raise SystemExit(f"fill diagnostics: {len(fd_new_em)} new EM rows but {len(fd_old_em)} old ones. Not merged.")
    fd_merged = pd.concat([fd[fd.method != "em"], fd_new_em], ignore_index=True)

    # ---- write -------------------------------------------------------------------------------
    for name, (merged, old_em, original) in new.items():
        if not (BACKUP / name).exists():
            shutil.copy(RES / name, BACKUP / name)
        old_em.to_csv(BACKUP / name.replace(".csv", "_em_original.csv"), index=False)
        merged.to_csv(RES / name, index=False)
        print(f"{name}: replaced {len(old_em)} EM rows with the rerun")
    if not (BACKUP / "fill_diagnostics.csv").exists():
        shutil.copy(RES / "fill_diagnostics.csv", BACKUP / "fill_diagnostics.csv")
    fd_old_em.to_csv(BACKUP / "fill_diagnostics_em_original.csv", index=False)
    fd_merged.to_csv(RES / "fill_diagnostics.csv", index=False)
    print(f"fill_diagnostics.csv: replaced {len(fd_old_em)} EM rows with the rerun")
    print("\nOriginals are in", BACKUP)
    print("Now rerun: make_figures.py, make_figures_1.py, make_figures_2.py (and your table scripts).")


if __name__ == "__main__":
    main()