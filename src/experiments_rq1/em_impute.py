from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def reduce_columns(
    df_encoded: pd.DataFrame, labels_dict: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, Dict[str, str]]:

    dropped_columns = {}
    keep_cols = []
    for var, states in labels_dict.items():
        var_cols = [f"{var}__{s}" for s in states]
        if not all(c in df_encoded.columns for c in var_cols):
            continue
        dropped_columns[var] = var_cols[-1]  # drop the last state's column
        keep_cols.extend(var_cols[:-1])

    return df_encoded[keep_cols].copy(), dropped_columns


def restore_columns(
    df_reduced: pd.DataFrame, labels_dict: Dict[str, List[str]], dropped_columns: Dict[str, str]
) -> pd.DataFrame:

    df_full = df_reduced.copy()
    for var, states in labels_dict.items():
        if var not in dropped_columns:
            continue
        kept_cols = [f"{var}__{s}" for s in states[:-1]]
        df_full[dropped_columns[var]] = 1.0 - df_full[kept_cols].sum(axis=1)

    return df_full


def em_impute(
    df_reduced: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-4,
    reg: float = 1e-6,
    full_mstep: bool = True,
    return_info: bool = False,
):
    """
    Gaussian EM imputation on the (reference-column-dropped) one-hot matrix.

    full_mstep=True  (the correct EM, used for the rerun)
        E-step: each blank is replaced by its conditional mean given the observed
                values of its row, AND the conditional covariance of the blanks
                (the uncertainty of those guesses) is accumulated.
        M-step: mu = mean of the filled matrix;
                cov = (sum of outer products of the filled rows + the accumulated
                       conditional covariances) / n
                The conditional-covariance term stops the covariance from being
                understated. With it, the observed-data log-likelihood can never go
                down from one iteration to the next.

    full_mstep=False (the ORIGINAL version, kept only to reproduce the old results)
        cov = np.cov(filled matrix), i.e. the filled values are treated as if they
        were observed. This is iterated conditional-mean imputation, not full EM.

    return_info=True also returns a dict with the number of iterations, whether the
    tolerance was reached, and the log-likelihood after every iteration.
    """
    X = df_reduced.to_numpy(dtype=float)
    n_rows, n_cols = X.shape
    missing_mask = np.isnan(X)

    # --- Initialization: fill missing with column means, get starting mu/cov ---
    col_means = np.nanmean(X, axis=0)
    X_filled = np.where(missing_mask, col_means, X)
    mu = X_filled.mean(axis=0)
    cov = np.cov(X_filled, rowvar=False)

    patterns = [tuple(row) for row in missing_mask]
    unique_patterns = set(patterns)
    pattern_row_idx = {p: np.array([i for i, pp in enumerate(patterns) if pp == p]) for p in unique_patterns}

    prev_loglik = None
    logliks = []
    converged = False
    for iteration in range(max_iter):
        # Sum over all rows of the conditional covariance of the blanks (needed by the full M-step)
        extra = np.zeros((n_cols, n_cols))

        # --- E-step: fill each missing-pattern group via conditional expectation ---
        for pattern, row_idx in pattern_row_idx.items():
            miss_cols = np.array(pattern)
            if not miss_cols.any():
                continue  # fully observed rows, nothing to fill

            obs_cols = ~miss_cols
            if not obs_cols.any():
                # nothing observed in these rows: fill with the mean; uncertainty = full covariance
                X_filled[np.ix_(row_idx, miss_cols)] = mu[miss_cols]
                extra[np.ix_(miss_cols, miss_cols)] += len(row_idx) * cov[np.ix_(miss_cols, miss_cols)]
                continue

            mu_obs, mu_mis = mu[obs_cols], mu[miss_cols]
            cov_oo = cov[np.ix_(obs_cols, obs_cols)] + reg * np.eye(obs_cols.sum())
            cov_mo = cov[np.ix_(miss_cols, obs_cols)]

            x_obs = X_filled[np.ix_(row_idx, obs_cols)]
            gain = np.linalg.solve(cov_oo, cov_mo.T)
            cond_mean = mu_mis + (x_obs - mu_obs) @ gain

            X_filled[np.ix_(row_idx, miss_cols)] = cond_mean

            # uncertainty of the guesses for this pattern: cov_mm - cov_mo cov_oo^-1 cov_om
            cond_cov = cov[np.ix_(miss_cols, miss_cols)] - cov_mo @ gain
            extra[np.ix_(miss_cols, miss_cols)] += len(row_idx) * cond_cov

        # --- M-step: re-estimate mu, cov ---
        mu = X_filled.mean(axis=0)
        if full_mstep:
            centred = X_filled - mu
            cov = (centred.T @ centred + extra) / n_rows
        else:
            cov = np.cov(X_filled, rowvar=False)

        # --- Convergence check: observed-data log-likelihood ---
        loglik = _observed_loglik(X, missing_mask, mu, cov, reg)
        logliks.append(loglik)
        if prev_loglik is not None and abs(loglik - prev_loglik) < tol:
            converged = True
            break
        prev_loglik = loglik

    out = pd.DataFrame(X_filled, columns=df_reduced.columns, index=df_reduced.index)
    if return_info:
        return out, {"n_iter": iteration + 1, "converged": converged, "logliks": logliks}
    return out


def _observed_loglik(X, missing_mask, mu, cov, reg) -> float:

    total = 0.0
    n_rows, n_cols = X.shape
    patterns = [tuple(row) for row in missing_mask]
    for pattern in set(patterns):
        miss_cols = np.array(pattern)
        obs_cols = ~miss_cols
        if not obs_cols.any():
            continue
        row_idx = np.array([i for i, p in enumerate(patterns) if p == pattern])
        mu_obs = mu[obs_cols]
        cov_oo = cov[np.ix_(obs_cols, obs_cols)] + reg * np.eye(obs_cols.sum())
        x_obs = X[np.ix_(row_idx, obs_cols)]

        diff = x_obs - mu_obs
        sign, logdet = np.linalg.slogdet(cov_oo)
        inv_term = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov_oo), diff)
        k = obs_cols.sum()
        loglik_rows = -0.5 * (k * np.log(2 * np.pi) + logdet + inv_term)
        total += loglik_rows.sum()

    return total