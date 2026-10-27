import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from em_impute import EMImputer
from mice_impute import MICEImputer
from soft_impute import SoftImputer
from bn_classifier import Discretizer, BNClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "taiwan"

QUICK_TEST = False            # True: one seed, one level, 2 folds, saved to a separate test folder.
                             # False: the full run, saved to results/taiwan.
OUT_DIR = PROJECT_ROOT / "results" / ("taiwan_quicktest" if QUICK_TEST else "taiwan")

SEEDS = [42, 43, 44, 45, 46]
LEVELS = [10, 20, 30, 40]
N_FOLDS = 10
N_BINS = 3
PAY_EDGES = {c: [0.5, 1.5] for c in ["PAY_0", "PAY_2", "PAY_3"]}   # no delay | 1 month | 2+ months


class MedianImputer:
    """Baseline: fill blanks with the training median of each column."""
    def fit_transform(self, df):
        self.med_ = df.median()
        return df.fillna(self.med_)

    def transform(self, df):
        return df.fillna(self.med_)


class NoImputer:
    """For the complete-data ceiling: nothing to fill."""
    def fit_transform(self, df):
        return df

    def transform(self, df):
        return df


IMPUTERS = {"Median": MedianImputer, "EM": EMImputer, "MICE": MICEImputer, "SoftImpute": SoftImputer}


def run_pipeline(imp, X_tr, X_te, y_tr, y_te):
    """Fill, bin, learn the network, predict, score.
    Returns the scores, the filled test part, and the learned network's connections."""
    tr_f = imp.fit_transform(X_tr)           # learn from the training customers only
    te_f = imp.transform(X_te)               # fill the test customers with what was learned

    disc = Discretizer(N_BINS, fixed_edges=PAY_EDGES).fit(tr_f)
    D_tr, D_te = disc.transform(tr_f), disc.transform(te_f)
    D_tr["Outcome"] = y_tr.to_numpy()        # Outcome goes back in only for the network
    names = {c: list(range(k)) for c, k in disc.n_states_.items()}
    names["Outcome"] = [0, 1]

    bn = BNClassifier().fit(D_tr, names)
    p = bn.predict_proba(D_te)
    pred = (p >= 0.5).astype(int)
    scores = {
        "auc": roc_auc_score(y_te, p),
        "f1": f1_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
    }
    edges = sorted((str(a), str(b)) for a, b in bn.model_.edges())
    return scores, te_f, edges


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(DATA_DIR / "taiwan_clean.csv")      # the answer key
    X_true = clean.drop(columns="Outcome")
    y = clean["Outcome"]
    spread = X_true.std()                                   # used to scale the accuracy score

    seeds = [42] if QUICK_TEST else SEEDS
    levels = [10] if QUICK_TEST else LEVELS
    names = ("scores", "accuracy", "edges")

    for seed in seeds:
        paths = {n: OUT_DIR / f"{n}_seed{seed}.csv" for n in names}
        if not QUICK_TEST and all(p.exists() for p in paths.values()):
            print(f"seed {seed}: already done, skipping")
            continue

        t0 = time.time()
        vm = json.load(open(DATA_DIR / "audit_logs" / f"audit_seed{seed}.json"))["vm_vars"]
        folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_true, y))
        if QUICK_TEST:
            folds = folds[:2]

        scores, acc, edge_rows = [], [], []

        def add_edges(level, fold, method, edges):
            for parent, child in edges:
                edge_rows.append({"seed": seed, "level": level, "fold": fold, "method": method,
                                  "parent": parent, "child": child})

        # ceiling: the complete data, nothing hidden, nothing to fill
        for k, (tr, te) in enumerate(folds, 1):
            s, _, e = run_pipeline(NoImputer(), X_true.iloc[tr], X_true.iloc[te], y.iloc[tr], y.iloc[te])
            scores.append({"seed": seed, "level": 0, "fold": k, "method": "Complete", **s})
            add_edges(0, k, "Complete", e)

        for level in levels:
            miss = pd.read_csv(DATA_DIR / "missing" / f"missing_{level}pct_seed{seed}.csv")
            X_miss = miss.drop(columns="Outcome")
            for k, (tr, te) in enumerate(folds, 1):
                X_tr, X_te = X_miss.iloc[tr], X_miss.iloc[te]
                hidden = X_te.isna()
                truth = X_true.iloc[te]
                for name, make in IMPUTERS.items():
                    s, te_f, e = run_pipeline(make(), X_tr, X_te, y.iloc[tr], y.iloc[te])
                    scores.append({"seed": seed, "level": level, "fold": k, "method": name, **s})
                    add_edges(level, k, name, e)

                    # accuracy: only the TEST customers' erased cells, error scaled by column spread
                    for v in vm:
                        cells = hidden[v].to_numpy()
                        if cells.sum() == 0:
                            continue
                        err = np.abs(te_f[v].to_numpy()[cells] - truth[v].to_numpy()[cells]).mean() / spread[v]
                        acc.append({"seed": seed, "level": level, "fold": k, "method": name,
                                    "column": v, "n_cells": int(cells.sum()), "scaled_error": float(err)})
                print(f"seed {seed}  level {level}%  fold {k} done  ({time.time() - t0:.0f}s)")

        pd.DataFrame(scores).to_csv(paths["scores"], index=False)
        pd.DataFrame(acc).to_csv(paths["accuracy"], index=False)
        pd.DataFrame(edge_rows).to_csv(paths["edges"], index=False)
        print(f"seed {seed}: saved")

    # combine the seeds into one file each, and print the summary
    for n in names:
        pd.concat([pd.read_csv(OUT_DIR / f"{n}_seed{s}.csv") for s in seeds]).to_csv(OUT_DIR / f"{n}.csv", index=False)
    sc = pd.read_csv(OUT_DIR / "scores.csv")
    ac = pd.read_csv(OUT_DIR / "accuracy.csv")
    print("\nMean AUC by level and method:")
    print(sc.groupby(["level", "method"])["auc"].mean().unstack().round(3))
    print("\nMean scaled error by level and method:")
    print(ac.groupby(["level", "method"])["scaled_error"].mean().unstack().round(3))


if __name__ == "__main__":
    main()