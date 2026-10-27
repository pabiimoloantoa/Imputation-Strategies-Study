from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from encode_decode import encode


def _initialize_missing(
    df: pd.DataFrame, missing_mask: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    
    df_filled = df.copy()
    for var in df.columns:
        n_missing = missing_mask[var].sum()
        if n_missing == 0:
            continue
        observed_values = df.loc[~missing_mask[var], var]
        proportions = observed_values.value_counts(normalize=True)
        draws = rng.choice(proportions.index, size=n_missing, p=proportions.values)
        df_filled.loc[missing_mask[var], var] = draws

    return df_filled


def _normalize_state_dtypes(df: pd.DataFrame, labels_dict: Dict[str, List[str]]) -> pd.DataFrame:
   
    df_norm = df.copy()
    for var in labels_dict:
        if var not in df_norm.columns:
            continue
        col = df_norm[var]
        if pd.api.types.is_float_dtype(col):
            df_norm[var] = col.astype("Int64").astype("string")
    return df_norm


def mice_impute(
    df: pd.DataFrame,
    labels_dict: Dict[str, List[str]],
    n_iterations: int = 10,
    random_state: int = None,
    max_depth: int = None,
) -> pd.DataFrame:
    
    df = _normalize_state_dtypes(df, labels_dict)
    rng = np.random.default_rng(random_state)
    missing_mask = df.isna()
    vars_with_missing = [c for c in df.columns if missing_mask[c].any()]

    if not vars_with_missing:
        return df.copy()

    df_current = _initialize_missing(df, missing_mask, rng)

    for iteration in range(n_iterations):
        for target_var in vars_with_missing:
            train_mask = ~missing_mask[target_var]
            predict_mask = missing_mask[target_var]

            predictor_labels = {v: s for v, s in labels_dict.items() if v != target_var}
            X_all = encode(df_current, predictor_labels)

            X_train = X_all.loc[train_mask]
            y_train = df.loc[train_mask, target_var]  # true observed labels, never imputed guesses
            X_predict = X_all.loc[predict_mask]

            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_predict)

            df_current.loc[predict_mask, target_var] = predictions

    return df_current


