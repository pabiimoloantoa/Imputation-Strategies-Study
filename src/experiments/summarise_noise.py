"""
summarise_noise.py -- how big is Hill Climbing's wobble compared with the
differences between methods?

Reads results/rq1/noise_floor.csv and results/rq1/augmented_results.csv.

Run from src/experiments:
    python summarise_noise.py
"""

from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[2] / "results" / "rq1"
n = pd.read_csv(root / "noise_floor.csv")
main = pd.read_csv(root / "augmented_results.csv")

print(f"Noise-floor runs: {len(n)}  |  hash seeds: {n.hash_seed.nunique()}  |  "
      f"seeds: {n.seed.nunique()}")

# 1. Does hash seed 0 reproduce the main results exactly?
h0 = n[n.hash_seed == 0].merge(main, on=["seed", "sparsity", "method"],
                               suffixes=("_noise", "_main"))
same = ((h0.kl_16_noise - h0.kl_16_main).abs() < 1e-6) & \
       (h0.cpdag_shd_16_noise == h0.cpdag_shd_16_main)
print(f"Hash seed 0 reproduces the main results in {int(same.sum())} of {len(h0)} datasets")
print()

# 2. The wobble: how much one dataset's scores move across hash seeds
COLS = ["cpdag_shd_16", "kl_16", "r_edges_extra", "r_edges_found"]
for (method, pct), grp in n.groupby(["method", "sparsity"]):
    per_seed = grp.groupby("seed")[COLS]
    sd = per_seed.std().mean()
    rng = per_seed.agg(lambda s: s.max() - s.min()).mean()
    print(f"Wobble of '{method}' at {pct}% (average over seeds, across hash seeds)")
    print(f"  CPDAG SHD 16 nodes : SD {sd.cpdag_shd_16:.2f}, typical range (max - min) {rng.cpdag_shd_16:.1f}")
    print(f"  KL 16 nodes        : SD {sd.kl_16:.3f}, typical range {rng.kl_16:.3f}")
    print(f"  extra R-links      : SD {sd.r_edges_extra:.2f}, typical range {rng.r_edges_extra:.1f}")
    print(f"  R-edges found      : SD {sd.r_edges_found:.2f}, typical range {rng.r_edges_found:.1f}")
    print()

# 3. Compare the method gaps (from the main results) with the baseline's wobble
for pct in sorted(n.sparsity.unique()):
    p = n[(n.method == "perfect") & (n.sparsity == pct)]
    if p.empty:
        continue
    w = p.groupby("seed")[["cpdag_shd_16", "kl_16"]].std().mean()
    base = main[(main.method == "perfect") & (main.sparsity == pct)].set_index("seed")
    print(f"Method gaps at {pct}% (mean difference from the perfect baseline, same seeds)")
    print(f"  baseline wobble: SHD SD {w.cpdag_shd_16:.2f}, KL SD {w.kl_16:.3f}")
    for m in ["em", "mice", "soft"]:
        mm = main[(main.method == m) & (main.sparsity == pct)].set_index("seed")
        d_shd = (mm.cpdag_shd_16 - base.cpdag_shd_16).mean()
        d_kl = (mm.kl_16 - base.kl_16).mean()
        r_shd = d_shd / w.cpdag_shd_16 if w.cpdag_shd_16 > 0 else float("nan")
        r_kl = d_kl / w.kl_16 if w.kl_16 > 0 else float("nan")
        print(f"  {m:<5} SHD {d_shd:+5.1f} ({r_shd:+.1f} x the wobble)   KL {d_kl:+.3f} ({r_kl:+.1f} x the wobble)")
    print()
    