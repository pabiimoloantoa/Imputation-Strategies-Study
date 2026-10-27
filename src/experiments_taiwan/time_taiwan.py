import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from em_impute import EMImputer
from mice_impute import MICEImputer
from soft_impute import SoftImputer
from bn_classifier import Discretizer, BNClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "data" / "taiwan" / "taiwan_clean.csv")

SIZES = [30000]
SEED = 42


class MedianImputer:
    def fit_transform(self, d):
        self.med_ = d.median()
        return d.fillna(self.med_)

    def transform(self, d):
        return d.fillna(self.med_)


IMPUTERS = {"Median": MedianImputer, "EM": EMImputer, "MICE": MICEImputer, "SoftImpute": SoftImputer}

for n in SIZES:
    if n >= len(df):
        sample = df.copy()          # use every customer
    else:
        sample, _ = train_test_split(df, train_size=n, stratify=df["Outcome"], random_state=SEED)
    X = sample.drop(columns="Outcome").reset_index(drop=True)
    y = sample["Outcome"].reset_index(drop=True)

    # STAND-IN missingness, for timing only. The real MAR injection comes later.
    rng = np.random.default_rng(0)
    for col in ["LIMIT_BAL", "PAY_0", "BILL_AMT2", "PAY_AMT1"]:
        X.loc[rng.random(len(X)) < 0.2, col] = np.nan

    tr, te = next(iter(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y)))
    print(f"\n{n} customers (train {len(tr)}, test {len(te)})")

    for name, make in IMPUTERS.items():
        t0 = time.time()
        imp = make()
        tr_f = imp.fit_transform(X.iloc[tr])
        te_f = imp.transform(X.iloc[te])
        t1 = time.time()

        disc = Discretizer(3).fit(tr_f)
        D_tr, D_te = disc.transform(tr_f), disc.transform(te_f)
        D_tr["Outcome"] = y.iloc[tr].to_numpy()
        names = {c: list(range(k)) for c, k in disc.n_states_.items()}
        names["Outcome"] = [0, 1]
        BNClassifier().fit(D_tr, names).predict_proba(D_te)
        t2 = time.time()

        print(f"  {name:<11} imputing {t1 - t0:6.1f}s   network {t2 - t1:6.1f}s")