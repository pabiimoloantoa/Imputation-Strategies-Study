import pandas as pd

from load_pima import PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"
OLD_POOLED = RES / "edges_pooled_oldbins.csv"

if (RES / "edges_pooled.csv").exists() and not OLD_POOLED.exists():
    raise SystemExit("Rename edges_pooled.csv to edges_pooled_oldbins.csv first.")

parts = []
for h in range(5):
    e = pd.read_csv(RES / f"edge_counts_obsbins_hash{h}.csv")
    e["hash"] = h
    parts.append(e)
edges = pd.concat(parts, ignore_index=True)
edges["pair"] = edges.apply(lambda r: " - ".join(sorted([r["parent"], r["child"]])), axis=1)  # direction ignored
edges.to_csv(RES / "edges_pooled.csv", index=False)

nets = edges.drop_duplicates(["method", "hash", "seed", "fold"])
print("Networks per method (expect 250):")
print(nets.groupby("method").size())

per_net = edges.groupby(["method", "hash", "seed", "fold"]).size().rename("n").reset_index()
print("\nAverage connections per network, by hash seed:")
print(per_net.groupby(["hash", "method"])["n"].mean().unstack().round(1))

new = edges.groupby(["pair", "method"]).size().unstack(fill_value=0)
top = new.loc[new.max(axis=1).sort_values(ascending=False).index].head(15)
print("\nNetworks (out of 250) in which each connection appears, pooled over 5 hash seeds:")
print(top)

if OLD_POOLED.exists():
    old = pd.read_csv(OLD_POOLED).groupby(["pair", "method"]).size().unstack(fill_value=0)
    old = old.reindex(index=top.index, columns=top.columns, fill_value=0)
    print("\nOld binning -> new binning (networks out of 250):")
    print((old.astype(str) + " -> " + top.astype(str)).to_string())

by_hash = edges.groupby(["pair", "method", "hash"]).size().unstack("hash", fill_value=0)
lo = (by_hash.min(axis=1) / 50).round(2).astype(str)
hi = (by_hash.max(axis=1) / 50).round(2).astype(str)
print("\nShare of networks with the connection, lowest to highest across the 5 hash seeds:")
print((lo + " to " + hi).unstack("method").loc[top.index])

