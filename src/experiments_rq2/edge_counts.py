import pandas as pd

from load_pima import load_pima, get_folds, PROJECT_ROOT
from run_rq2_2 import IMPUTERS, N_BINS, SEEDS
from bn_classifier import Discretizer, BNClassifier

OUT = PROJECT_ROOT / "results" / "rq2" / "edge_counts.csv"

df = load_pima()
X, y = df.drop(columns="Outcome"), df["Outcome"]

rows = []
for seed in SEEDS:
    for k, (tr, te) in enumerate(get_folds(df, seed=seed), 1):
        for name, make in IMPUTERS.items():
            imp = make()
            tr_filled = imp.fit_transform(X.iloc[tr])
            disc = Discretizer(N_BINS).fit(tr_filled)
            D = disc.transform(tr_filled)
            D["Outcome"] = y.iloc[tr].to_numpy()
            names = {c: list(range(n)) for c, n in disc.n_states_.items()}
            names["Outcome"] = [0, 1]
            bn = BNClassifier().fit(D, names)
            for a, b in bn.model_.edges():
                rows.append({"method": name, "seed": seed, "fold": k, "parent": a, "child": b})
    print("finished seed", seed)

edges = pd.DataFrame(rows)
edges.to_csv(OUT, index=False)

n_runs = len(SEEDS) * 10
edges["pair"] = edges.apply(lambda r: " - ".join(sorted([r["parent"], r["child"]])), axis=1)

print("\nAverage number of connections per network:")
print((edges.groupby("method").size() / n_runs).round(1))

freq = edges.groupby(["pair", "method"]).size().unstack(fill_value=0)
freq = freq.loc[freq.max(axis=1).sort_values(ascending=False).index]
print(f"\nRuns (out of {n_runs}) in which each connection appears:")
print(freq.head(15))