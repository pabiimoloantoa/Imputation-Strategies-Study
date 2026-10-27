import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from load_pima import load_pima, get_folds, PROJECT_ROOT
from em_impute import EMImputer
from mice_impute import MICEImputer
from soft_impute import SoftImputer
from bn_classifier import Discretizer, BNClassifier

N_BINS = 3
OUT_DIR = PROJECT_ROOT / "results" / "rq2"


class MedianImputer:
    """Baseline: fill blanks with the training median of each column."""
    def fit_transform(self, df):
        self.med_ = df.median()
        return df.fillna(self.med_)
    def transform(self, df):
        return df.fillna(self.med_)


IMPUTERS = {
    "Median": MedianImputer,
    "EM": EMImputer,
    "MICE": MICEImputer,
    "SoftImpute": SoftImputer,
}


def run_fold(make_imputer, X_tr, X_te, y_tr, y_te):
    imp = make_imputer()
    tr_filled = imp.fit_transform(X_tr)      # learn from training patients only
    te_filled = imp.transform(X_te)          # fill test patients with what was learned

    disc = Discretizer(N_BINS).fit(tr_filled)
    D_tr, D_te = disc.transform(tr_filled), disc.transform(te_filled)
    D_tr["Outcome"] = y_tr.to_numpy()        # Outcome goes back in only for the network

    state_names = {c: list(range(n)) for c, n in disc.n_states_.items()}
    state_names["Outcome"] = [0, 1]

    p = BNClassifier().fit(D_tr, state_names).predict_proba(D_te)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_te, p),
        "f1": f1_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
    }


def main():
    df = load_pima()
    X, y = df.drop(columns="Outcome"), df["Outcome"]
    folds = get_folds(df)                    # the same folds for every method

    rows = []
    for name, make_imputer in IMPUTERS.items():
        for k, (tr, te) in enumerate(folds, 1):
            scores = run_fold(make_imputer, X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te])
            rows.append({"method": name, "fold": k, **scores})
            print(name, k, {m: round(v, 3) for m, v in scores.items()})

    results = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "fold_scores.csv", index=False)
    print(results.groupby("method")[["auc", "f1", "precision", "recall"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()