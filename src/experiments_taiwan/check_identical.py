import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from soft_impute import SoftImputer
from bn_classifier import Discretizer, BNClassifier
from run_taiwan import MedianImputer, PAY_EDGES, N_BINS, DATA_DIR

SEED, LEVEL = 42, 10

clean = pd.read_csv(DATA_DIR / "taiwan_clean.csv")
X_true, y = clean.drop(columns="Outcome"), clean["Outcome"]
miss = pd.read_csv(DATA_DIR / "missing" / f"missing_{LEVEL}pct_seed{SEED}.csv").drop(columns="Outcome")
tr, te = next(iter(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X_true, y)))


def run(make):
    imp = make()
    tr_f = imp.fit_transform(miss.iloc[tr])
    te_f = imp.transform(miss.iloc[te])
    disc = Discretizer(N_BINS, fixed_edges=PAY_EDGES).fit(tr_f)
    D_tr, D_te = disc.transform(tr_f), disc.transform(te_f)
    D_tr["Outcome"] = y.iloc[tr].to_numpy()
    names = {c: list(range(k)) for c, k in disc.n_states_.items()}
    names["Outcome"] = [0, 1]
    bn = BNClassifier().fit(D_tr, names)
    return D_te, bn.predict_proba(D_te), sorted(bn.model_.get_markov_blanket("Outcome"))


D_med, p_med, mb_med = run(MedianImputer)
D_soft, p_soft, mb_soft = run(SoftImputer)

print("customers whose group differs (Median vs Soft-Impute), by column:")
print((D_med != D_soft).sum())
print("\ncustomers whose predicted probability differs:", int((np.abs(p_med - p_soft) > 1e-9).sum()), "of", len(p_med))
print("largest difference in probability:", round(float(np.abs(p_med - p_soft).max()), 4))
print("\nvariables directly tied to Outcome")
print("  Median:     ", mb_med)
print("  Soft-Impute:", mb_soft)