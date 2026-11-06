"""
draw_graphs.py -- draw the true network next to a learned network.

Left panel : the true 16-node network (11 variables + that level's R-nodes).
Right panel: the learned network, edges coloured by how they compare
             with the truth:
               green  = correct edge, correct direction
               orange = right connection, opposite direction
               red    = extra edge (not in the true network)
               grey dashed = true edge the learned network is missing

python draw_graphs.py 0 20 perfect/em/mice/soft

Layout (new): variables sit in layers (parents above children) and the R-nodes in a bottom row.
Within each row the order is chosen to reduce crossings, and an edge that spans several layers is
routed AROUND the nodes in between (through reserved gaps), so no arrow runs behind a node.
Both panels use the same positions, so the two pictures can be compared node by node.
"""

import re
import textwrap
from pathlib import Path

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.path import Path as MPath
import networkx as nx

GREEN, ORANGE, RED, GREY = "#2e8b57", "#e08a00", "#d62728", "#8c8c8c"
# same palette as plot_replicate_structure.py: Vo blue, Vm coral, R-parent edges purple
VO_FILL, VO_EDGE = "#B5D4F4", "#185FA5"
VM_FILL, VM_EDGE = "#F5C4B3", "#D85A30"
R_FILL, R_EDGE = "#E6E4FA", "#534AB7"          # R-nodes: light purple, to match the purple R-parent edges
R_PARENT_EDGE = "#534AB7"
PLAIN_EDGE = "#9A9A96"

# layout sizes, in "layout units" (one real node is about 1 unit wide)
BOX_W, BOX_H = 1.0, 0.62      # node box
DUMMY_W = 0.18                # gap reserved for an edge passing through a layer
PAD = 0.22                    # minimum space between neighbours in a row
DY = 1.75                     # vertical distance between layers
SCALE = 0.8                   # inches per layout unit


def is_r(name):
    return name.startswith("R_")


def pretty(name):
    """Readable node label: 'NumberOfCreditInquiries' -> 'Number Of Credit Inquiries'."""
    if is_r(name):
        name = re.sub(r"_\d+pct$", "", name[2:])
        return "R: " + textwrap.fill(re.sub(r"(?<!^)(?=[A-Z])", " ", name), 12)
    return textwrap.fill(re.sub(r"(?<!^)(?=[A-Z])", " ", name), 12)


def parse_edges(text):
    """'A>B;C>D' -> [('A','B'), ('C','D')]"""
    return [tuple(e.split(">")) for e in text.split(";")] if text else []


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
def _isotonic(vals):
    """Pool-adjacent-violators: closest non-decreasing sequence (least squares)."""
    blocks = []
    for v in vals:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out += [s / c] * c
    return out


def make_layout(nodes, true_edges, learned_edges=()):
    """
    Positions for both panels.  Returns a dict with
      pos    : {node: (x, y)}
      layer  : {node: layer index}
      chains : {frozenset({a, b}): [(x, y), ...]}  points an edge passes through, top to bottom
      extent : (xmin, xmax, ymin, ymax)
    Layout uses the true edges AND the learned edges, so every edge in either panel is routed.
    """
    orig = [n for n in nodes if not is_r(n)]
    r_nodes = [n for n in nodes if is_r(n)]

    g = nx.DiGraph()
    g.add_nodes_from(orig)
    g.add_edges_from(e for e in true_edges if not is_r(e[0]) and not is_r(e[1]))
    rows = [sorted(layer) for layer in nx.topological_generations(g)]
    if r_nodes:
        rows.append(sorted(r_nodes))
    n_rows = len(rows)
    layer_of = {n: i for i, row in enumerate(rows) for n in row}

    # every connection (either panel) gets one chain of items: real node, gap items, real node
    pairs = set()
    for e in list(true_edges) + list(learned_edges):
        if e[0] in layer_of and e[1] in layer_of and e[0] != e[1]:
            pairs.add(frozenset(e))
    up, down, chains = {}, {}, {}
    dummies = [[] for _ in range(n_rows)]
    for idx, pair in enumerate(sorted(pairs, key=lambda p: sorted(p))):
        u, v = sorted(pair, key=lambda n: (layer_of[n], n))     # u is the upper node
        a, b = layer_of[u], layer_of[v]
        if a == b:
            continue                                            # same row: drawn as an arc
        chain = [u]
        for k in range(a + 1, b):
            d = ("~", idx, k)
            dummies[k].append(d)
            chain.append(d)
        chain.append(v)
        chains[pair] = chain
        for c1, c2 in zip(chain, chain[1:]):
            down.setdefault(c1, []).append(c2)
            up.setdefault(c2, []).append(c1)

    # ---- order within each row: barycentre sweeps, keep the arrangement with fewest crossings
    def frac(i, n):
        return i / (n - 1) if n > 1 else 0.5

    def sweep(order, downward):
        ks = range(1, n_rows) if downward else range(n_rows - 2, -1, -1)
        nbrs = up if downward else down
        for k in ks:
            ref = order[k - 1] if downward else order[k + 1]
            im = {it: i for i, it in enumerate(ref)}
            cur = order[k]

            def key(it, cur=cur, im=im, ref=ref):
                ns = [im[x] for x in nbrs.get(it, []) if x in im]
                if ns:
                    return frac(sum(ns) / len(ns), len(ref))
                return frac(cur.index(it), len(cur))
            order[k] = sorted(cur, key=key)

    def crossings(order):
        pos = [{it: i for i, it in enumerate(layer)} for layer in order]
        total = 0
        for k in range(n_rows - 1):
            segs = [(pos[k][a], pos[k + 1][b]) for a in order[k] for b in down.get(a, [])
                    if b in pos[k + 1]]
            for i in range(len(segs)):
                for j in range(i + 1, len(segs)):
                    if (segs[i][0] - segs[j][0]) * (segs[i][1] - segs[j][1]) < 0:
                        total += 1
        return total

    rng = np.random.default_rng(0)
    best, best_c = None, None
    for attempt in range(12):
        order = [list(row) + list(dummies[k]) for k, row in enumerate(rows)]
        if attempt:
            for layer in order:
                rng.shuffle(layer)
        for _ in range(12):
            sweep(order, True)
            sweep(order, False)
        c = crossings(order)
        if best is None or c < best_c:
            best, best_c = [list(layer) for layer in order], c
    order = best

    # ---- horizontal positions: pull each item towards its neighbours, never closer than PAD
    def width(it):
        return BOX_W if isinstance(it, str) else DUMMY_W

    def place(layer, desired):
        w = [width(it) for it in layer]
        off = [0.0]
        for j in range(1, len(layer)):
            off.append(off[-1] + (w[j - 1] + w[j]) / 2 + PAD)
        fitted = _isotonic([d - o for d, o in zip(desired, off)])
        return [f + o for f, o in zip(fitted, off)]

    x = {}
    for layer in order:
        for it, xv in zip(layer, place(layer, [0.0] * len(layer))):
            x[it] = xv
    for _ in range(40):
        for k in list(range(n_rows)) + list(range(n_rows - 2, -1, -1)):
            des = []
            for it in order[k]:
                ns = [x[n] for n in up.get(it, []) + down.get(it, [])]
                des.append(sum(ns) / len(ns) if ns else x[it])
            for it, xv in zip(order[k], place(order[k], des)):
                x[it] = xv
    shift = min(x[it] - width(it) / 2 for it in x) - 0.4
    x = {it: xv - shift for it, xv in x.items()}

    def y_of(k):
        return -k * DY - (0.4 if (r_nodes and k == n_rows - 1) else 0.0)

    pos = {n: (x[n], y_of(layer_of[n])) for n in layer_of}

    def where(it):
        k = layer_of[it] if isinstance(it, str) else it[2]
        return (x[it], y_of(k))
    chain_pts = {pair: [where(it) for it in chain] for pair, chain in chains.items()}

    xs = [x[it] + s * width(it) / 2 for it in x for s in (-1, 1)]
    ys = [p[1] for p in pos.values()]
    extent = (min(xs) - 0.4, max(xs) + 0.4, min(ys) - 1.0, max(ys) + 1.0)
    return dict(pos=pos, layer=layer_of, chains=chain_pts, extent=extent, n_rows=n_rows)


# ----------------------------------------------------------------------------
# edges
# ----------------------------------------------------------------------------
def classify(true_edges, learned_edges):
    t, l = set(true_edges), set(learned_edges)
    t_pairs = {frozenset(e) for e in t}
    correct = [e for e in l if e in t]
    reversed_ = [e for e in l if e not in t and (e[1], e[0]) in t]
    extra = [e for e in l if frozenset(e) not in t_pairs]
    missed = [e for e in t if e not in l and (e[1], e[0]) not in l]
    return correct, reversed_, extra, missed


def _clip(centre, towards):
    """Where the line from a node's centre towards another point leaves the node's box."""
    c, p = np.array(centre, float), np.array(towards, float)
    d = p - c
    t = min((BOX_W / 2) / abs(d[0]) if d[0] else np.inf, (BOX_H / 2) / abs(d[1]) if d[1] else np.inf)
    return c + d * t


def _smooth(points):
    """Smooth curve through the points (Catmull-Rom converted to cubic Beziers)."""
    p = [np.array(q, float) for q in points]
    if len(p) == 2:
        return MPath(p, [MPath.MOVETO, MPath.LINETO])
    ext = [p[0]] + p + [p[-1]]
    verts, codes = [p[0]], [MPath.MOVETO]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        verts += [p1 + (p2 - p0) / 6.0, p2 - (p3 - p1) / 6.0, p2]
        codes += [MPath.CURVE4] * 3
    return MPath(verts, codes)


def edge_path(src, dst, layout):
    pos, layer = layout["pos"], layout["layer"]
    if layer[src] == layer[dst]:                       # same row: arc over (or under, in the last row) the nodes
        (x1, y), (x2, _) = pos[src], pos[dst]
        sign = -1 if layer[src] == layout["n_rows"] - 1 else 1
        y0 = y + sign * BOX_H / 2
        h = 0.55 + 0.16 * abs(x2 - x1)
        return MPath([(x1, y0), (x1 + 0.25 * (x2 - x1), y0 + sign * h),
                      (x1 + 0.75 * (x2 - x1), y0 + sign * h), (x2, y0)],
                     [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    pts = list(layout["chains"][frozenset((src, dst))])        # top to bottom
    if layer[src] > layer[dst]:                                # edge points upwards
        pts = pts[::-1]
    pts[0] = tuple(_clip(pts[0], pts[1]))
    pts[-1] = tuple(_clip(pts[-1], pts[-2]))
    return _smooth(pts)


def draw_panel(ax, nodes, layout, edge_groups, title):
    pos = layout["pos"]
    # Vm variables are the ones that have an R-node (R_<variable>_<pct>pct)
    vm = {re.match(r"R_(.+)_\d+pct$", n).group(1) for n in nodes if is_r(n) and re.match(r"R_(.+)_\d+pct$", n)}
    for n in nodes:
        x, y = pos[n]
        if is_r(n):
            fill, edge = R_FILL, R_EDGE
        elif n in vm:
            fill, edge = VM_FILL, VM_EDGE
        else:
            fill, edge = VO_FILL, VO_EDGE
        ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                                    boxstyle="round,pad=0,rounding_size=0.16",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=3))
        ax.text(x, y, pretty(n), ha="center", va="center", fontsize=6.5, zorder=4)
    for edges, colour, style, width in edge_groups:
        for src, dst in edges:
            ax.add_patch(FancyArrowPatch(path=edge_path(src, dst, layout), arrowstyle="-|>",
                                         mutation_scale=13, color=colour, linestyle=style,
                                         linewidth=width, zorder=2, shrinkA=0, shrinkB=0))
    xmin, xmax, ymin, ymax = layout["extent"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def draw_comparison(nodes, true_edges, learned_edges, out_path, title="", layout_edges=None):
    """layout_edges: extra edges to reserve room for (e.g. the learned edges of the other methods),
    so the same variable sits in the same place in every picture of one seed and level."""
    layout = make_layout(nodes, true_edges, list(learned_edges) + list(layout_edges or []))
    correct, reversed_, extra, missed = classify(true_edges, learned_edges)

    xmin, xmax, ymin, ymax = layout["extent"]
    pw, ph = (xmax - xmin) * SCALE, (ymax - ymin) * SCALE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * pw + 0.6, ph + 2.0))
    r_parent_edges = [e for e in true_edges if is_r(e[1])]
    plain_edges = [e for e in true_edges if not is_r(e[1])]
    draw_panel(ax1, nodes, layout, [(plain_edges, PLAIN_EDGE, "solid", 1.3),
                                    (r_parent_edges, R_PARENT_EDGE, "solid", 1.9)], "True network")
    draw_panel(ax2, nodes, layout,
               [(missed, GREY, "dashed", 1.3),
                (correct, GREEN, "solid", 2.0),
                (reversed_, ORANGE, "solid", 2.0),
                (extra, RED, "solid", 2.0)],
               f"Learned network  |  {len(correct)} correct, {len(reversed_)} reversed, "
               f"{len(extra)} extra, {len(missed)} missing")

    legend = [Line2D([0], [0], color=GREEN, lw=2, label="correct edge"),
              Line2D([0], [0], color=ORANGE, lw=2, label="reversed direction"),
              Line2D([0], [0], color=RED, lw=2, label="extra edge"),
              Line2D([0], [0], color=GREY, lw=1.5, ls="--", label="true edge missing"),
              Patch(facecolor=VO_FILL, edgecolor=VO_EDGE, label="Vo (fully observed)"),
              Patch(facecolor=VM_FILL, edgecolor=VM_EDGE, label="Vm (has missingness)"),
              Patch(facecolor=R_FILL, edgecolor=R_EDGE, label="R-node (missingness)"),
              Line2D([0], [0], color=R_PARENT_EDGE, lw=2, label="true R-parent edge (left panel)")]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_dataset(seed, pct, method):
    """
    Draw one dataset's learned network against the truth. Uses the learned edges
    saved in augmented_results.csv when they exist (so the picture matches the
    saved scores); otherwise learns the network again (needs PYTHONHASHSEED).
    """
    from augmented_16 import load_trimmed_truth, score_dataset, PROJECT_ROOT

    results_path = PROJECT_ROOT / "results" / "rq1" / "augmented_results.csv"
    truth = load_trimmed_truth(seed, pct)

    scores = None
    layout_edges = []          # learned edges of every method on this seed/level -> shared positions
    if results_path.exists():
        df = pd.read_csv(results_path)
        same = df[(df.seed == seed) & (df.sparsity == pct)]
        for text in same["learned_edges"].dropna():
            layout_edges += parse_edges(text)
        row = same[same.method == method]
        if len(row):
            scores = row.iloc[0].to_dict()
    if scores is None:
        if "PYTHONHASHSEED" not in os.environ:
            raise SystemExit("Set PYTHONHASHSEED first, e.g.  $env:PYTHONHASHSEED = '0'")
        scores = score_dataset(seed, pct, method)

    n_r = len([n for n in truth.nodes() if is_r(n)])
    title = (f"seed {seed}, {pct}% missing, {method}   |   "
             f"CPDAG SHD: {int(scores['cpdag_shd_16'])} (16 nodes), "
             f"{int(scores['cpdag_shd_11'])} (11 variables)   |   KL {float(scores['kl_16']):.3f}   |   "
             f"R-edges found {int(scores['r_edges_found'])}/{n_r}, "
             f"extra R-links {int(scores['r_edges_extra'])}")
    out = PROJECT_ROOT / "results" / "rq1" / "graphs" / f"seed{seed}_{pct}pct_{method}.png"
    draw_comparison(list(truth.nodes()), list(truth.edges()),
                    parse_edges(scores["learned_edges"]), out, title, layout_edges=layout_edges)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3:
        print("Saved", draw_dataset(int(args[0]), int(args[1]), args[2]))
    elif args == ["all"]:
        from augmented_16 import PROJECT_ROOT
        df = pd.read_csv(PROJECT_ROOT / "results" / "rq1" / "augmented_results.csv")
        for r in df.itertuples():
            print("Saved", draw_dataset(int(r.seed), int(r.sparsity), r.method))
    else:
        print("Usage:  python draw_graphs.py <seed> <sparsity> <method>")
        print("   e.g. python draw_graphs.py 2 20 em")
        print("        python draw_graphs.py all      (every row in the results file)")

        