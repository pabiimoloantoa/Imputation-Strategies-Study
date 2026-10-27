from collections import Counter

from load_pima import load_pima, get_folds
from run_rq2 import IMPUTERS, N_BINS
from bn_classifier import Discretizer, BNClassifier

df = load_pima()
X, y = df.drop(columns="Outcome"), df["Outcome"]
folds = get_folds(df)

for name, make in IMPUTERS.items():
    counts = Counter()
    for tr, te in folds:
        imp = make()
        tr_filled = imp.fit_transform(X.iloc[tr])
        disc = Discretizer(N_BINS).fit(tr_filled)
        D = disc.transform(tr_filled)
        D["Outcome"] = y.iloc[tr].to_numpy()
        names = {c: list(range(n)) for c, n in disc.n_states_.items()}
        names["Outcome"] = [0, 1]
        bn = BNClassifier().fit(D, names)
        counts.update(bn.model_.get_markov_blanket("Outcome"))
    print(name, dict(counts))