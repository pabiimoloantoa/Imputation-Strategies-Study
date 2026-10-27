import numpy as np
import pandas as pd


class SoftImputer:
    """Soft-Impute (SVD with shrinkage) for continuous data.

    fit(train)     -> chooses the shrinkage by hold-out validation, fills the training data
    transform(new) -> fills blanks in new rows (e.g. the test fold) using the same shrinkage
    """

    def __init__(self, fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4), holdout=0.10,
                 max_rows_for_choice=5000, max_iter=100, tol=1e-4, random_state=0):
        self.fractions = fractions                    # candidate shrinkages, as fractions of the top singular value
        self.holdout = holdout                        # share of observed cells hidden to compare candidates
        self.max_rows_for_choice = max_rows_for_choice
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _soft_impute(Xs, lam, max_iter, tol):
        miss = np.isnan(Xs)
        Z = np.where(miss, 0.0, Xs)                   # blanks start at the column average (0)
        for _ in range(max_iter):
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            S_shr = np.maximum(S - lam, 0.0)          # soft-threshold the singular values
            Z_new = np.where(miss, (U * S_shr) @ Vt, Xs)    # refill the blanks only
            change = np.linalg.norm(Z_new - Z) / (np.linalg.norm(Z) + 1e-12)
            Z = Z_new
            if change < tol:
                break
        return Z, S_shr, Vt

    def fit(self, df):
        X = df.to_numpy(float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.lo_ = np.nanmin(X, axis=0)
        self.hi_ = np.nanmax(X, axis=0)
        Xs = (X - self.mean_) / self.std_             # standardise
        rng = np.random.default_rng(self.random_state)

        # 1) choose the shrinkage: hide some observed cells, see which candidate recovers them best
        rows = np.arange(len(Xs))
        if len(rows) > self.max_rows_for_choice:      # a subsample is enough, and keeps this fast
            rows = rng.choice(rows, self.max_rows_for_choice, replace=False)
        Xc = Xs[rows].copy()
        seen = np.argwhere(~np.isnan(Xc))
        pick = seen[rng.random(len(seen)) < self.holdout]
        truth = Xc[pick[:, 0], pick[:, 1]].copy()
        Xc[pick[:, 0], pick[:, 1]] = np.nan
        sigma1 = np.linalg.svd(np.where(np.isnan(Xc), 0.0, Xc), compute_uv=False)[0]

        errors = {}
        for f in self.fractions:
            Z, _, _ = self._soft_impute(Xc, f * sigma1, self.max_iter, self.tol)
            errors[f] = float(np.sqrt(np.mean((Z[pick[:, 0], pick[:, 1]] - truth) ** 2)))
        self.fraction_ = min(errors, key=errors.get)
        self.choice_errors_ = errors

        # 2) final fit on all the training rows, with the chosen shrinkage
        sigma1_full = np.linalg.svd(np.where(np.isnan(Xs), 0.0, Xs), compute_uv=False)[0]
        self.lambda_ = self.fraction_ * sigma1_full
        Z, S_shr, Vt = self._soft_impute(Xs, self.lambda_, self.max_iter, self.tol)
        keep = S_shr > 0
        self.Vt_ = Vt[keep]
        self.B_ = Vt[keep].T * np.sqrt(S_shr[keep])   # patterns, scaled by their shrunk strength
        self.train_filled_ = self._unscale(Z, df)
        return self

    def _unscale(self, Zs, df):
        X = np.clip(Zs * self.std_ + self.mean_, self.lo_, self.hi_)   # no impossible values
        return pd.DataFrame(X, columns=df.columns, index=df.index)

    def transform(self, df):
        Xs = (df.to_numpy(float) - self.mean_) / self.std_
        out = Xs.copy()
        miss = np.isnan(Xs)
        k = self.B_.shape[1]
        for pattern in {tuple(r) for r in miss}:      # rows with the same blanks are solved together
            m = np.array(pattern)
            if not m.any():
                continue
            rows = np.where((miss == m).all(axis=1))[0]
            o = ~m
            if k == 0 or not o.any():
                out[np.ix_(rows, m)] = 0.0
                continue
            Bo, Bm = self.B_[o], self.B_[m]
            # ridge regression of the observed values onto the patterns, penalty = the shrinkage
            a = np.linalg.solve(Bo.T @ Bo + self.lambda_ * np.eye(k), Bo.T @ Xs[np.ix_(rows, o)].T)
            out[np.ix_(rows, m)] = (Bm @ a).T
        return self._unscale(out, df)

    def fit_transform(self, df):
        return self.fit(df).train_filled_