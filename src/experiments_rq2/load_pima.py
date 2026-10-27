from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIMA = PROJECT_ROOT / "data" / "pima" / "diabetes.csv"

ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_pima():
    df = pd.read_csv(PIMA)
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    return df


def get_folds(df, n_splits=10, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    return [(tr, te) for tr, te in skf.split(X, y)]


if __name__ == "__main__":
    df = load_pima()
    print(df.isna().sum())
    for i, (tr, te) in enumerate(get_folds(df), 1):
        print(f"fold {i}: train={len(tr)}, test={len(te)}, "
              f"test positives={df.loc[te, 'Outcome'].sum()}")