"""
make_figures_2.py  -  can the structure learner still "see" what caused the missingness?

  fig_ownlinks_vs_redges.png     (Option A) per method: own-links vs true R-edges found, one bubble per cell
  fig_rnode_outcomes.png         (Option B) what happened to each R-node, by method and sparsity
  r_node_outcomes.csv            one row per R-node per dataset (results/rq1)

Option B needs the true network of every seed (model_with_r_nodes_seed<N>.bif) and the learned_edges
column of augmented_results.csv.  Everything is treated as ADJACENCY (direction ignored), the same way
"true connections recovered (any direction)" was counted earlier.

Place in <project>/src/experiments/ and run:   python src/experiments/make_figures_2.py
"""
from pathlib import Path
import ast
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG - change names here only
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "rq1"
MAIN_CSV = RESULTS / "augmented_results.csv"
OUT_DIR = RESULTS / "figures_report"
LONG_CSV = RESULTS / "r_node_outcomes.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "synthetic" / "models"   # searched recursively if the file is not here
MODEL_FILE = "model_with_r_nodes_seed{seed}.bif"

COL_SEED = "seed"
COL_SPARSITY = "sparsity"
COL_METHOD = "method"
COL_R_FOUND = "r_edges_found"
COL_OWN = "r_own_links"
COL_EDGES = "learned_edges"

R_NODE_RE = re.compile(r"^R_(.+)_(\d+)pct$")    # R_<variable>_<pct>pct
# ----------------------------------------------------------------------------

METHODS = ["perfect", "mice", "em", "soft"]
LABELS = {"perfect": "Perfect (baseline)", "mice": "MICE", "em": "EM", "soft": "Soft-Impute"}
COLORS = {"perfect": "#555555", "mice": "#1b9e77", "em": "#d95f02", "soft": "#7570b3"}   # same palette as make_figures.py

OUTCOMES = [  # (key, legend label, colour) - stacking order bottom -> top
    ("clean", "True edge found, no own-link", "#2a9d8f"),
    ("found_own", "True edge found + own-link", "#f4c27a"),
    ("own_only", "True edge missed, own-link instead", "#d1495b"),
    ("neither", "True edge missed, no own-link", "#b8bcc4"),
]


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    })


# ----------------------------------------------------------------------------
# loading / parsing
# ----------------------------------------------------------------------------
def load_main():
    if not MAIN_CSV.exists():
        sys.exit(f"Cannot find {MAIN_CSV}. The script expects to sit in <project>/src/experiments/.")
    df = pd.read_csv(MAIN_CSV)
    need = [COL_SEED, COL_SPARSITY, COL_METHOD, COL_R_FOUND, COL_OWN, COL_EDGES]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"{MAIN_CSV.name} has no column(s) {missing}.\nIt has: {list(df.columns)}\n"
                 f"Edit the CONFIG block at the top of this script.")
    df[COL_METHOD] = df[COL_METHOD].astype(str).str.strip().str.lower()
    df[COL_SPARSITY] = pd.to_numeric(df[COL_SPARSITY].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df


def parse_edges(cell):
    """learned_edges cell -> set of frozenset({a, b}) (undirected adjacency).
    Accepts JSON / python-literal lists of pairs, or 'A>B;C>D' / 'A->B;C->D' style strings."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    obj = cell
    if isinstance(cell, str):
        s = cell.strip()
        obj = None
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(s)
                break
            except Exception:
                pass
        if obj is None:
            obj = [p for p in re.split(r"[;|\n]+", s) if p.strip()]
    pairs = set()
    for e in obj:
        if isinstance(e, str):
            parts = [p.strip(" '\"()[]") for p in re.split(r"->|→|>|,", e)]
        else:
            parts = [str(p).strip() for p in e]
        parts = [p for p in parts if p]
        if len(parts) == 2 and parts[0] != parts[1]:
            pairs.add(frozenset(parts))
    return pairs


_truth_cache = {}


def true_rnodes(seed, pct):
    """{R-node name: (its variable, [its true parents])} for one seed and sparsity level."""
    if seed not in _truth_cache:
        try:
            from pgmpy.readwrite import BIFReader
        except ImportError:
            sys.exit("pgmpy is needed to read the .bif files (you already use it in hillclimb.py).")
        name = MODEL_FILE.format(seed=seed)
        path = MODEL_DIR / name
        if not path.exists():
            hits = list(PROJECT_ROOT.rglob(name))
            if not hits:
                sys.exit(f"Cannot find {name} under {PROJECT_ROOT}. Set MODEL_DIR in the CONFIG block.")
            path = hits[0]
        _truth_cache[seed] = BIFReader(str(path)).get_model()
    model = _truth_cache[seed]
    out = {}
    for node in model.nodes():
        m = R_NODE_RE.match(str(node))
        if m and int(m.group(2)) == int(pct):
            out[str(node)] = (m.group(1), [str(p) for p in model.get_parents(node)])
    if not out:
        sys.exit(f"No R-nodes for the {pct}% level in seed {seed}'s network.\n"
                 f"Node names there: {sorted(map(str, model.nodes()))}\n"
                 f"Adjust R_NODE_RE in the CONFIG block.")
    return out


def build_long(main):
    rows = []
    n_edges_total = 0
    for r in main.itertuples(index=False):
        edges = parse_edges(getattr(r, COL_EDGES))
        n_edges_total += len(edges)
        for rn, (var, parents) in true_rnodes(int(getattr(r, COL_SEED)), getattr(r, COL_SPARSITY)).items():
            own = frozenset({rn, var}) in edges
            found = bool(parents) and all(frozenset({rn, p}) in edges for p in parents)
            key = ("found_own" if own else "clean") if found else ("own_only" if own else "neither")
            rows.append(dict(seed=getattr(r, COL_SEED), sparsity=getattr(r, COL_SPARSITY),
                             method=getattr(r, COL_METHOD), r_node=rn, variable=var,
                             true_parents="|".join(parents), true_edge_found=found,
                             own_link=own, outcome=key))
    if n_edges_total == 0:
        sample = main[COL_EDGES].iloc[0]
        sys.exit(f"Could not read any edges from the '{COL_EDGES}' column.\n"
                 f"First cell looks like: {str(sample)[:300]!r}\nTell me the format and I'll adjust parse_edges().")
    return pd.DataFrame(rows)


def sanity_check(main, long):
    """Compare my per-R-node counts with the r_edges_found / r_own_links already in the CSV."""
    agg = long.groupby(["seed", "sparsity", "method"]).agg(
        found=("true_edge_found", "sum"), own=("own_link", "sum")).reset_index()
    m = main.merge(agg, left_on=[COL_SEED, COL_SPARSITY, COL_METHOD],
                   right_on=["seed", "sparsity", "method"], how="left", suffixes=("", "_re"))
    ok_f = (m[COL_R_FOUND] == m["found"]).mean()
    ok_o = (m[COL_OWN] == m["own"]).mean()
    print(f"Sanity check vs the CSV: r_edges_found matches in {ok_f:.0%} of datasets, "
          f"r_own_links matches in {ok_o:.0%}.")
    if ok_f < 0.95 or ok_o < 0.95:
        print("  (Not an error, but my adjacency rule differs a little from the one used when the CSV was built; "
              "the figures still use my per-R-node counts.)")


# ----------------------------------------------------------------------------
# Option A: own-links vs true R-edges found
# ----------------------------------------------------------------------------
def fig_scatter(main):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, m in zip(axes, METHODS):
        sub = main[main[COL_METHOD] == m]
        cnt = sub.groupby([COL_OWN, COL_R_FOUND]).size().reset_index(name="n")
        ax.scatter(cnt[COL_OWN], cnt[COL_R_FOUND], s=cnt["n"] * 7, color=COLORS[m],
                   alpha=0.65, edgecolor="white", linewidth=0.8, zorder=2)
        for r in cnt.itertuples(index=False):
            if r.n >= 5:      # count printed beside the bubble so the mean marker cannot hide it
                ax.annotate(str(r.n), (getattr(r, COL_OWN), getattr(r, COL_R_FOUND)),
                            xytext=(11, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=8, color="#333333", zorder=3)
        ax.scatter(sub[COL_OWN].mean(), sub[COL_R_FOUND].mean(), marker="X", s=90,
                   color="black", zorder=4)
        ax.set_title(f"{LABELS[m]}\nmean: {sub[COL_OWN].mean():.1f} own-links, {sub[COL_R_FOUND].mean():.1f} found", fontsize=9)
        ax.set_xticks(range(0, 6))
        ax.set_yticks(range(0, 6))
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.9)
    axes[0].set_ylabel("True R-edges found (of 5)")
    fig.supxlabel("Own-links per dataset (R-nodes linked to their own variable)", fontsize=10)
    fig.suptitle("Bubble size = number of datasets in that cell (all seeds and levels pooled; counts shown for 5 or more); black X = mean", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_ownlinks_vs_redges.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Option B: outcome of every R-node
# ----------------------------------------------------------------------------
def fig_outcomes(long):
    levels = sorted(long["sparsity"].dropna().unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(3.3 * len(levels), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(METHODS))
    for ax, lv in zip(axes, levels):
        sub = long[long["sparsity"] == lv]
        bottom = np.zeros(len(METHODS))
        for key, label, color in OUTCOMES:
            share = np.array([(sub[sub["method"] == m]["outcome"] == key).mean() for m in METHODS])
            ax.bar(x, share, bottom=bottom, color=color, width=0.7, label=label, zorder=2)
            bottom += share
        ax.set_title(f"{int(lv)}% missing")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of R-nodes")
    axes[0].set_yticks(np.linspace(0, 1, 6))
    axes[0].set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    n_per_bar = int((long["method"] == "perfect").groupby(long["sparsity"]).sum().min())
    fig.suptitle("What happened to each R-node after imputation", y=1.02)
    fig.text(0.5, -0.12, f"Each bar covers {n_per_bar} R-nodes (all seeds at that level). Edge direction is ignored.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_rnode_outcomes.png", bbox_inches="tight")
    plt.close(fig)


def print_hypothesis_table(long):
    """Is the true edge missed more often when the R-node also has an own-link?"""
    print("\nTrue edge missed, split by whether the R-node has an own-link (all seeds and levels pooled):")
    print(f"{'method':<17}{'n with own-link':>16}{'% edge missed':>15}{'n without':>12}{'% edge missed':>15}")
    for m in METHODS:
        s = long[long["method"] == m]
        a, b = s[s["own_link"]], s[~s["own_link"]]
        fa = f"{(~a['true_edge_found']).mean():.0%}" if len(a) else "n/a"
        fb = f"{(~b['true_edge_found']).mean():.0%}" if len(b) else "n/a"
        print(f"{LABELS[m]:<17}{len(a):>16}{fa:>15}{len(b):>12}{fb:>15}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    main_df = load_main()
    long = build_long(main_df)
    long.to_csv(LONG_CSV, index=False)
    sanity_check(main_df, long)
    fig_scatter(main_df)
    fig_outcomes(long)
    print_hypothesis_table(long)
    print(f"\nSaved 2 figures to {OUT_DIR} and {LONG_CSV.name} to {RESULTS}")


if __name__ == "__main__":
    main()