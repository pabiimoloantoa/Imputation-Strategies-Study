import pandas as pd

from load_pima import PROJECT_ROOT

RES = PROJECT_ROOT / "results" / "rq2"

parts = []
for h in range(5):
    e = pd.read_csv(RES / f"edge_counts_hash{h}.csv")
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

N_POOLED, N_PER_HASH = 250, 50
freq = edges.groupby(["pair", "method"]).size().unstack(fill_value=0)
top = freq.loc[freq.max(axis=1).sort_values(ascending=False).index].head(15)
print(f"\nNetworks (out of {N_POOLED}) in which each connection appears, pooled over 5 hash seeds:")
print(top)

by_hash = edges.groupby(["pair", "method", "hash"]).size().unstack("hash", fill_value=0)
lo = (by_hash.min(axis=1) / N_PER_HASH).round(2).astype(str)
hi = (by_hash.max(axis=1) / N_PER_HASH).round(2).astype(str)
spread = (lo + " to " + hi).unstack("method")
print("\nShare of networks with the connection, lowest to highest across the 5 hash seeds:")
print(spread.loc[top.index])