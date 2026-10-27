import numpy as np
import pandas as pd


class SoftImputer:
    """Soft-Impute (SVD with shrinkage) for continuous data.

    fit(train)     -> fills the training data and keeps the low-rank directions
    transform(new) -> fills blanks in new rows using those directions
    """

    def __init__(self, shrinkage=None, max_iter=100, tol=1e-4, ridge=1e-3):
        self.shrinkage = shrinkage   # None = largest singular value / 50
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    def fit(self, df):
        X = df.to_numpy(float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.lo_ = np.nanmin(X, axis=0)
        self.hi_ = np.nanmax(X, axis=0)

        Xs = (X - self.mean_) / self.std_          # standardise
        miss = np.isnan(Xs)
        Z = np.where(miss, 0.0, Xs)                # 0 = column average

        lam = self.shrinkage
        if lam is None:
            lam = np.linalg.svd(Z, compute_uv=False)[0] / 50
        self.lambda_ = lam

        for _ in range(self.max_iter):
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            S_shr = np.maximum(S - lam, 0.0)                   # soft-threshold
            Z_new = np.where(miss, (U * S_shr) @ Vt, Xs)       # refill blanks only
            change = np.linalg.norm(Z_new - Z) / (np.linalg.norm(Z) + 1e-12)
            Z = Z_new
            if change < self.tol:
                break

        self.Vt_ = Vt[S_shr > 0]                   # directions that survived shrinkage
        self.train_filled_ = self._unscale(Z, df)
        return self

    def _unscale(self, Zs, df):
        X = Zs * self.std_ + self.mean_
        X = np.clip(X, self.lo_, self.hi_)         # no impossible values
        return pd.DataFrame(X, columns=df.columns, index=df.index)

    def transform(self, df):
        Xs = (df.to_numpy(float) - self.mean_) / self.std_
        out = Xs.copy()
        k = self.Vt_.shape[0]
        for i in range(Xs.shape[0]):
            m = np.isnan(Xs[i])
            if not m.any():
                continue
            o = ~m
            if k == 0 or not o.any():
                out[i, m] = 0.0                    # fall back to the column average
                continue
            A = self.Vt_[:, o].T                   # loadings of the observed columns
            c = np.linalg.solve(A.T @ A + self.ridge * np.eye(k), A.T @ Xs[i, o])
            out[i, m] = c @ self.Vt_[:, m]
        return self._unscale(out, df)

    def fit_transform(self, df):
        return self.fit(df).train_filled_