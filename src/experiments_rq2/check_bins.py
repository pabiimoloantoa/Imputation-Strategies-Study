import warnings

import pandas as pd

from load_pima import load_pima, get_folds
from em_impute import EMImputer
from mice_impute import MICEImputer
from soft_impute import SoftImputer
from bn_classifier import Discretizer

warnings.filterwarnings("ignore")


class MedianImputer:
    def fit_transform(self, d):
        self.med_ = d.median()
        return d.fillna(self.med_)

    def transform(self, d):
        return d.fillna(self.med_)


IMPUTERS = {"Median": MedianImputer, "EM": EMImputer, "MICE": MICEImputer, "SoftImpute": SoftImputer}

df = load_pima()
X = df.drop(columns="Outcome")
rows = []
for k, (tr, te) in enumerate(get_folds(df, seed=42), 1):
    for name, make in IMPUTERS.items():
        tr_f = make().fit_transform(X.iloc[tr])
        disc = Discretizer(3).fit(tr_f)
        for col, n in disc.n_states_.items():
            rows.append({"method": name, "column": col, "groups": n})

t = pd.DataFrame(rows)
print("Average number of groups per column over 10 folds (3 = a full split, 2 = two cut points merged):")
print(t.groupby(["column", "method"])["groups"].mean().unstack().round(1))