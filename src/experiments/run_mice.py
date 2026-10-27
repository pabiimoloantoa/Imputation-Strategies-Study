"""
ran for manually
"""

import re
from pathlib import Path

import pandas as pd

from mice_impute import mice_impute
from random_bn import VARIABLE_STATE_LABELS

MISSING_FILE = "../../data/synthetic/missing/missing_40pct_seed19.csv"

# Output goes to data/synthetic/mice_completed/mice_completed_Xpct_seedY.csv
COMPLETED_DIR = Path("../../data/synthetic/mice_completed")

N_ITERATIONS = 10
RANDOM_STATE = 0


def check_observed_preserved(df_original: pd.DataFrame, completed: pd.DataFrame) -> None:
    print("\n--- Observed-cell preservation check ---")
    for col in df_original.columns:
        mask = df_original[col].notna()
        original = df_original.loc[mask, col]
        if pd.api.types.is_numeric_dtype(original):
            original = original.astype("Int64").astype("string")
        matches = (completed.loc[mask, col] == original).all()
        print(f"  {col:<25s} {matches}")


def main():
    df = pd.read_csv(MISSING_FILE)
    print(f"Loaded {MISSING_FILE} -> shape {df.shape}")
    print("\nMissing counts per column:")
    print(df.isna().sum())

    completed = mice_impute(
        df,
        VARIABLE_STATE_LABELS,
        n_iterations=N_ITERATIONS,
        random_state=RANDOM_STATE,
    )

    print(f"\nCompleted (imputed, named-states) shape: {completed.shape}")
    print(completed.head(10))

    # --- Sanity check: did MICE corrupt any originally-observed cell? ---
    check_observed_preserved(df, completed)

    # --- Save the completed dataset ---
    input_name = Path(MISSING_FILE).name
    match = re.search(r"(\d+pct_seed\d+)\.csv$", input_name)
    if not match:
        raise ValueError(
            f"Couldn't parse sparsity/seed from filename '{input_name}' -- "
            f"expected a name like 'missing_10pct_seed0.csv'."
        )
    suffix = match.group(1)  # e.g. "10pct_seed0"

    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPLETED_DIR / f"mice_completed_{suffix}.csv"
    completed.to_csv(out_path, index=False)
    print(f"\nSaved MICE-completed dataset -> {out_path}")


if __name__ == "__main__":
    main()