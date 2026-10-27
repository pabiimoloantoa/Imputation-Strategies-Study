import numpy as np
import pandas as pd


def soft_impute(
    df_encoded: pd.DataFrame,
    lambda_: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> pd.DataFrame:
    
    X = df_encoded.to_numpy(dtype=float)
    missing_mask = np.isnan(X)

    # Step 1: initialize missing entries with 0
    Z = np.where(missing_mask, 0.0, X)

    prev_Z = None
    for iteration in range(max_iter):
        # Step 2a: SVD of the current completed matrix
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)

        # Step 2b: soft-threshold the singular values
        S_shrunk = np.maximum(S - lambda_, 0.0)

        # Step 2c: reconstruct
        Z_new = U @ np.diag(S_shrunk) @ Vt

        # Step 2d: re-fill ONLY missing entries; observed entries reset
        # to their true values every iteration, so they can never drift
        Z_new = np.where(missing_mask, Z_new, X)

        # Convergence: relative change in the full matrix
        if prev_Z is not None:
            change = np.linalg.norm(Z_new - prev_Z) / (np.linalg.norm(prev_Z) + 1e-12)
            if change < tol:
                Z = Z_new
                break

        prev_Z = Z_new
        Z = Z_new

    return pd.DataFrame(Z, columns=df_encoded.columns, index=df_encoded.index)