import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class MICEImputer:
    """Chained-equations imputation with decision-tree models (continuous data).

    fit(train)     -> fills the training data and keeps one tree per column with blanks
    transform(new) -> fills blanks in new rows (e.g. the test fold) using those trees
    """

    def __init__(self, n_iterations=10, min_samples_leaf=5, random_state=0):
        self.n_iterations = n_iterations
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _initial_fill(self, df, miss, rng):
        """Start: fill each blank with a random draw from that column's training values."""
        cur = df.copy()
        for col in self.targets_:
            n = miss[col].sum()
            if n:
                cur.loc[miss[col], col] = rng.choice(self.observed_[col], size=n)
        return cur

    def fit(self, df):
        rng = np.random.default_rng(self.random_state)
        miss = df.isna()
        self.columns_ = list(df.columns)
        self.targets_ = [c for c in df.columns if miss[c].any()]
        self.observed_ = {c: df.loc[~miss[c], c].to_numpy() for c in self.targets_}
        self.models_ = {}

        cur = self._initial_fill(df, miss, rng)
        for _ in range(self.n_iterations):
            for t in self.targets_:
                predictors = [c for c in self.columns_ if c != t]
                seen = ~miss[t]
                model = DecisionTreeRegressor(
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                # learn from the truly observed values only, never from guesses
                model.fit(cur.loc[seen, predictors], df.loc[seen, t])
                cur.loc[miss[t], t] = model.predict(cur.loc[miss[t], predictors])
                self.models_[t] = model

        self.train_filled_ = cur
        return self

    def transform(self, df):
        rng = np.random.default_rng(self.random_state)
        miss = df.isna()
        unknown = [c for c in df.columns if miss[c].any() and c not in self.models_]
        if unknown:
            raise ValueError(f"No trained model for columns with blanks: {unknown}")

        cur = self._initial_fill(df, miss, rng)
        for _ in range(self.n_iterations):
            for t in self.targets_:
                if not miss[t].any():
                    continue
                predictors = [c for c in self.columns_ if c != t]
                cur.loc[miss[t], t] = self.models_[t].predict(cur.loc[miss[t], predictors])
        return cur

    def fit_transform(self, df):
        return self.fit(df).train_filled_