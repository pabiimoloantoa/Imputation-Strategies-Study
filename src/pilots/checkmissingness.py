"""
check_missingness.py

Reads the missing-data CSVs produced by inject_mar_missingness.py and
reports:
  1. Per-column missing rate (should be close to the target sparsity level)
  2. Dataset-wide missing rate, computed two ways:
     a) relative to ALL cells in the dataset (11 columns) -- this will be
        much lower than the target, since only the 5 Vm columns can ever
        contain missing values.
     b) relative to only the 5 Vm columns' cells -- this is the number
        that should land close to the target sparsity level overall.

Usage:
    python check_missingness.py --seed 0
"""

import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MISSING_DIR = PROJECT_ROOT / "data" / "synthetic" / "missing"

VM_VARS = ["Income", "Default", "EmploymentStatus", "CreditUtilization", "Age"]
SPARSITY_LEVELS = [10, 20, 30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    for pct in SPARSITY_LEVELS:
        path = MISSING_DIR / f"missing_{pct}pct_seed{args.seed}.csv"
        if not path.exists():
            print(f"[!] File not found: {path}")
            continue

        df = pd.read_csv(path)

        n_rows, n_cols = df.shape
        total_cells_all = n_rows * n_cols
        total_cells_vm = n_rows * len(VM_VARS)

        total_missing = df[VM_VARS].isna().sum().sum()

        print(f"\n=== {path.name} (target: {pct}%) ===")
        print("Per-column missing rate:")
        for var in VM_VARS:
            rate = df[var].isna().mean()
            print(f"  {var:<20s} {rate:.4f} ({rate:.1%})")

        rate_all_cells = total_missing / total_cells_all
        rate_vm_cells = total_missing / total_cells_vm

        print("Dataset-wide missing rate:")
        print(f"  Relative to ALL {n_cols} columns ({total_cells_all} cells): "
              f"{rate_all_cells:.4f} ({rate_all_cells:.1%})")
        print(f"  Relative to the {len(VM_VARS)} Vm columns only ({total_cells_vm} cells): "
              f"{rate_vm_cells:.4f} ({rate_vm_cells:.1%})  <-- compare this to the {pct}% target")


if __name__ == "__main__":
    main()