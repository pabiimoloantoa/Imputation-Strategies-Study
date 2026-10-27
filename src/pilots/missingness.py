"""
inject_mar_missingness.py

Injects MAR missingness into the ground-truth synthetic dataset, following
the m-graph approach (Mohan, Pearl & Tian, 2013): each Vm variable gets an
auxiliary R indicator whose probability of "missing" is conditioned on a
single Vo parent variable (never on the Vm variable itself, and never on
another Vm variable -- this is what keeps the mechanism MAR rather than
MNAR).

For each of the three sparsity levels (10%, 20%, 30%), this script produces
a separate copy of the dataset with values blanked out according to the
per-R-node CPTs derived analytically (see conversation / methodology notes
for the derivation of each CPT).

Usage:
    python inject_mar_missingness.py --seed 0
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths (adjust if your folder structure differs) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "synthetic" / "ground_truth_complete.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "missing"

# --- R-node definitions: Vm variable -> (R-parent, CPT per sparsity level) ---
# Each CPT maps {parent_state: P(missing)} for a given target sparsity level.
# These were derived by:
#   1. Choosing a plausible ratio of missingness rates across parent states
#      (the "story" for why that state should have more/less missingness).
#   2. Solving for the rate that makes the parent-marginal-weighted average
#      equal the target sparsity level.

R_NODE_CPTS = {
    "Income": {
        "parent": "DebtRatio",
        0.10: {"low": 0.066, "medium": 0.099, "high": 0.132},
        0.20: {"low": 0.132, "medium": 0.198, "high": 0.263},
        0.30: {"low": 0.198, "medium": 0.296, "high": 0.395},
    },
    "Default": {
        "parent": "NumberOfCreditInquiries",
        0.10: {"low": 0.068, "medium": 0.102, "high": 0.136},
        0.20: {"low": 0.136, "medium": 0.204, "high": 0.272},
        0.30: {"low": 0.204, "medium": 0.306, "high": 0.408},
    },
    "EmploymentStatus": {
        "parent": "CreditHistoryLength",
        0.10: {"short": 0.130, "medium": 0.098, "long": 0.065},
        0.20: {"short": 0.260, "medium": 0.195, "long": 0.130},
        0.30: {"short": 0.391, "medium": 0.293, "long": 0.195},
    },
    "CreditUtilization": {
        "parent": "CreditHistoryLength",
        0.10: {"short": 0.068, "medium": 0.103, "long": 0.137},
        0.20: {"short": 0.137, "medium": 0.205, "long": 0.273},
        0.30: {"short": 0.205, "medium": 0.307, "long": 0.410},
    },
    "Age": {
        "parent": "PaymentHistory",
        0.10: {"excellent": 0.068, "good": 0.088, "fair": 0.108, "poor": 0.135},
        0.20: {"excellent": 0.135, "good": 0.176, "fair": 0.217, "poor": 0.271},
        0.30: {"excellent": 0.203, "good": 0.264, "fair": 0.325, "poor": 0.406},
    },
}

SPARSITY_LEVELS = [0.10, 0.20, 0.30]


def inject_missingness(df: pd.DataFrame, sparsity: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Returns a copy of df with MAR missingness injected at the given sparsity
    level, following R_NODE_CPTS. Original df (ground truth) is left untouched.
    """
    df_missing = df.copy()

    for vm_var, spec in R_NODE_CPTS.items():
        parent_col = spec["parent"]
        cpt = spec[sparsity]

        # Map each row's parent state to its P(missing), then roll the coin.
        p_missing = df[parent_col].map(cpt)

        if p_missing.isna().any():
            unmapped = df.loc[p_missing.isna(), parent_col].unique()
            raise ValueError(
                f"CPT for '{vm_var}' (parent '{parent_col}') has no entry for "
                f"state(s) {unmapped}. Check state names match the dataset exactly."
            )

        coin_flips = rng.random(len(df))
        is_missing = coin_flips < p_missing.values

        df_missing.loc[is_missing, vm_var] = np.nan

    return df_missing


def report_achieved_rates(df_missing: pd.DataFrame, sparsity: float):
    print(f"\n--- Target sparsity: {sparsity:.0%} ---")
    for vm_var in R_NODE_CPTS:
        achieved = df_missing[vm_var].isna().mean()
        print(f"  {vm_var:<20s} achieved: {achieved:.4f} ({achieved:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                         help="Random seed for this run (controls the coin flips).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(GROUND_TRUTH_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sparsity in SPARSITY_LEVELS:
        df_missing = inject_missingness(df, sparsity, rng)
        report_achieved_rates(df_missing, sparsity)

        out_path = OUTPUT_DIR / f"missing_{int(sparsity*100)}pct_seed{args.seed}.csv"
        df_missing.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()