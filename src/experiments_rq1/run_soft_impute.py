import re
from pathlib import Path

import pandas as pd

from encode_decode import encode, decode
from soft_impute import soft_impute
from random_bn import VARIABLE_STATE_LABELS

MISSING_FILE = "../../data/synthetic/missing/missing_40pct_seed19.csv"

# Output goes to data/synthetic/soft_completed/soft_completed_Xpct_seedY.csv
COMPLETED_DIR = Path("../../data/synthetic/soft_completed")

LAMBDA = 1.0
MAX_ITER = 100


def check_observed_preserved(df_original: pd.DataFrame, decoded: pd.DataFrame) -> None:
    
    print("\n--- Observed-cell preservation check ---")
    for col in df_original.columns:
        mask = df_original[col].notna()
        original = df_original.loc[mask, col]
        if pd.api.types.is_numeric_dtype(original):
            original = original.astype("Int64").astype("string")
        matches = (decoded.loc[mask, col] == original).all()
        print(f"  {col:<25s} {matches}")


def main():
    df = pd.read_csv(MISSING_FILE)
    print(f"Loaded {MISSING_FILE} -> shape {df.shape}")
    print("\nMissing counts per column:")
    print(df.isna().sum())

    # --- Run the pipeline ---
    encoded = encode(df)
    print(f"\nEncoded shape: {encoded.shape}")

    completed_encoded = soft_impute(encoded, lambda_=LAMBDA, max_iter=MAX_ITER)
    decoded = decode(completed_encoded)

    print(f"\nDecoded (imputed, named-states) shape: {decoded.shape}")
    print(decoded.head(10))

    # --- Sanity check: did Soft-Impute corrupt any originally-observed cell? ---
    check_observed_preserved(df, decoded)

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
    out_path = COMPLETED_DIR / f"soft_completed_{suffix}.csv"
    decoded.to_csv(out_path, index=False)
    print(f"\nSaved Soft-Impute-completed dataset -> {out_path}")


if __name__ == "__main__":
    main()


