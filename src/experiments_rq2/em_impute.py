import numpy as np
import pandas as pd


class EMImputer:
    """EM for continuous data with the full M-step.

    fit(train)       -> learns the column averages (mu) and relationships (cov)
    transform(any)   -> fills blanks in any table using what fit() learned
    """

    def __init__(self, max_iter=100, tol=1e-4, reg=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    def _fill(self, X):
        """Fill blanks with their expected values, and also return how uncertain
        those guesses are (needed so the M-step does not understate the spread)."""
        X = X.copy()
        miss = np.isnan(X)
        extra = np.zeros((X.shape[1], X.shape[1]))
        for pattern in {tuple(r) for r in miss}:
            m = np.array(pattern)
            if not m.any():
                continue
            rows = np.where((miss == m).all(axis=1))[0]
            o = ~m
            if not o.any():                            # nothing observed: use the average
                X[np.ix_(rows, m)] = self.mu_[m]
                extra[np.ix_(m, m)] += len(rows) * self.cov_[np.ix_(m, m)]
                continue
            cov_oo = self.cov_[np.ix_(o, o)] + self.reg * np.eye(o.sum())
            cov_mo = self.cov_[np.ix_(m, o)]
            gain = np.linalg.solve(cov_oo, cov_mo.T)
            X[np.ix_(rows, m)] = self.mu_[m] + (X[np.ix_(rows, o)] - self.mu_[o]) @ gain
            cond_cov = self.cov_[np.ix_(m, m)] - cov_mo @ gain   # uncertainty of the guesses
            extra[np.ix_(m, m)] += len(rows) * cond_cov
        return X, extra

    def fit(self, df):
        X = df.to_numpy(float)
        n = X.shape[0]
        miss = np.isnan(X)

        # start: fill blanks with column averages
        X_start = np.where(miss, np.nanmean(X, axis=0), X)
        self.mu_ = X_start.mean(axis=0)
        self.cov_ = np.cov(X_start, rowvar=False)

        for _ in range(self.max_iter):
            X_filled, extra = self._fill(X)            # E-step
            mu_new = X_filled.mean(axis=0)             # M-step
            centred = X_filled - mu_new
            cov_new = (centred.T @ centred + extra) / n    # adds back the uncertainty
            change = max(np.abs(mu_new - self.mu_).max(), np.abs(cov_new - self.cov_).max())
            self.mu_, self.cov_ = mu_new, cov_new
            if change < self.tol:
                break

        self.lo_ = np.nanmin(X, axis=0)
        self.hi_ = np.nanmax(X, axis=0)
        return self

    def transform(self, df):
        X, _ = self._fill(df.to_numpy(float))
        X = np.clip(X, self.lo_, self.hi_)
        return pd.DataFrame(X, columns=df.columns, index=df.index)

    def fit_transform(self, df):
        return self.fit(df).transform(df)