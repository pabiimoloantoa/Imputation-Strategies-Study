"""
compute_marginals.py

Computes the marginal (state) distribution of each variable that will act as
an R-parent (i.e. a variable driving MAR missingness for another variable),
using the named-states ground-truth dataset.

These marginals are needed to solve for the per-state missingness rates in
each R node's CPT, so that the overall (weighted-average) missingness rate
lands close to the target sparsity level (10% / 20% / 30%).
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "ground_truth_complete.csv"

# The variables that will act as R-parents in this study.
# (CreditHistoryLength appears once here even though it's used as the
#  R-parent for two Vm variables -- EmploymentStatus and CreditUtilization --
#  its marginal distribution is the same regardless of which child uses it.)
R_PARENT_VARS = [
    "DebtRatio",
    "NumberOfCreditInquiries",
    "CreditHistoryLength",
    "PaymentHistory",
]


def main():
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

    for var in R_PARENT_VARS:
        if var not in df.columns:
            print(f"[!] Column '{var}' not found in dataset -- check spelling/casing.\n")
            continue

        print(f"--- {var} ---")
        counts = df[var].value_counts(dropna=False)
        proportions = df[var].value_counts(normalize=True, dropna=False)

        summary = pd.DataFrame({
            "count": counts,
            "proportion": proportions.round(4)
        })
        print(summary)
        print()


if __name__ == "__main__":
    main()