"""
generate_replicate.py

Runs one full, self-contained replicate of the RQ1 data-generation pipeline:

    1. Generate a FRESH random ground-truth BN structure + CPDs (seed-specific)
    2. Sample a fresh 5000-row dataset from it
    3. Automatically select Vm (gets missingness) and Vo (stays observed)
       variables -- Vm must have at least one parent, so an eligible R-parent
       exists
    4. Automatically assign each Vm variable's R-node an R-parent, drawn from
       its own direct parents where at least one is in Vo; falls back to a
       random Vo variable (with a warning) if none of its parents qualify
    5. Automatically derive each R-node's CPT: a generic rank-based ratio
       (1x, 1.5x, 2x, ... across the parent's states, in state order) solved
       against the parent's REAL marginal distribution (from this replicate's
       own sampled data) to hit each target sparsity level (10/20/30%)
    6. Inject missingness via per-row weighted coin flips
    7. Save the ground truth, the missing-data versions, and a JSON audit log
       of every automated choice made for this replicate (for transparency /
       methodology reporting)

Usage:
    python generate_replicate.py --seed 0
    python generate_replicate.py --seed 0 --n_vm 5
    time python generate_replicate.py --seed 0

Everything is self-contained per seed: a different seed gives a different
random BN, a different sampled dataset, different Vm/Vo/R-parent choices,
and different (freshly re-derived) CPTs -- but the SAME general procedure.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from random_ground_truth_bn import (
    NODE_STATES,
    VARIABLE_STATE_LABELS,
    build_random_ground_truth_bn,
    relabel_states,
    check_validity,
    sample_synthetic_dataset,
)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "synthetic" / "ground_truth_per_seed"
MISSING_DIR = PROJECT_ROOT / "data" / "synthetic" / "missing"
AUDIT_DIR = PROJECT_ROOT / "data" / "synthetic" / "audit_logs"
MODELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "models"

# --- Fixed experimental design choices (apply identically every replicate) ---
N_SAMPLES = 5000
SPARSITY_LEVELS = [0.10, 0.20, 0.30]
DEFAULT_N_VM = 5
MAX_PROBABILITY = 0.95  # safety ceiling; a CPT entry should never realistically hit 1.0


def select_vm_vo(model, rng: np.random.Generator, n_vm: int):
    """
    Selects which variables become Vm (missingness targets), restricted to
    variables that have at least one parent (so an R-parent is available).
    Returns (vm_vars, vo_vars).
    """
    eligible = [n for n in model.nodes() if len(model.get_parents(n)) > 0]
    ineligible_roots = [n for n in model.nodes() if n not in eligible]

    n_vm_actual = min(n_vm, len(eligible))
    if n_vm_actual < n_vm:
        warnings.warn(
            f"Only {len(eligible)} variables have a parent (roots excluded: "
            f"{ineligible_roots}); requested n_vm={n_vm}, using {n_vm_actual} instead."
        )

    vm_vars = [str(v) for v in rng.choice(eligible, size=n_vm_actual, replace=False)]
    vo_vars = [n for n in model.nodes() if n not in vm_vars]
    return vm_vars, vo_vars


def select_r_parent(model, vm_var: str, vo_vars: list, rng: np.random.Generator) -> str:
    """
    Picks the R-parent for a given Vm variable: prefers one of vm_var's own
    direct parents that is in Vo (clean, defensible MAR story). Falls back to
    a random Vo variable if none of its parents qualify (e.g. all of this
    variable's parents happen to also be in Vm).
    """
    direct_parents = model.get_parents(vm_var)
    eligible_direct = [p for p in direct_parents if p in vo_vars]

    if eligible_direct:
        return str(rng.choice(eligible_direct))

    warnings.warn(
        f"'{vm_var}' has no direct parent in Vo (its parents are {direct_parents}); "
        f"falling back to a random Vo variable for its R-node."
    )
    return str(rng.choice(vo_vars))


def derive_cpt(marginal: pd.Series, target: float) -> dict:
    """
    Given a parent's real marginal distribution (state -> proportion, in a
    fixed state order) and a target sparsity level, derives the per-state
    missingness probability using a generic rank-based ratio: the i-th state
    (in its natural state order) gets a multiplier of 1 + 0.5*i, i.e.
    1x, 1.5x, 2x, 2.5x, ... This is a fixed, content-neutral rule applied
    identically regardless of which variable or replicate it's used on.
    """
    states = list(marginal.index)
    multipliers = [1 + 0.5 * i for i in range(len(states))]

    weighted_sum = sum(marginal[s] * m for s, m in zip(states, multipliers))
    k = target / weighted_sum

    cpt = {}
    for s, m in zip(states, multipliers):
        p = k * m
        if p > MAX_PROBABILITY:
            warnings.warn(
                f"Derived P(missing)={p:.3f} for state '{s}' at target={target:.0%} "
                f"exceeds the {MAX_PROBABILITY} safety ceiling; clipping."
            )
            p = MAX_PROBABILITY
        cpt[s] = round(p, 4)

    return cpt


def inject_missingness(df: pd.DataFrame, vm_vars: list, r_parents: dict,
                        cpts: dict, sparsity: float, rng: np.random.Generator) -> pd.DataFrame:
    df_missing = df.copy()
    for vm_var in vm_vars:
        parent_col = r_parents[vm_var]
        cpt = cpts[vm_var][sparsity]

        p_missing = df[parent_col].map(cpt)
        if p_missing.isna().any():
            unmapped = df.loc[p_missing.isna(), parent_col].unique()
            raise ValueError(f"CPT for '{vm_var}' has no entry for state(s) {unmapped}.")

        coin_flips = rng.random(len(df))
        is_missing = coin_flips < p_missing.values
        df_missing.loc[is_missing, vm_var] = np.nan

    return df_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                         help="Drives structure generation, CPDs, sampling, Vm/Vo/R-parent selection, and missingness coin flips for this replicate.")
    parser.add_argument("--n_vm", type=int, default=DEFAULT_N_VM,
                         help="Target number of Vm variables (may be reduced if too few eligible variables exist).")
    args = parser.parse_args()

    seed = args.seed
    rng = np.random.default_rng(seed)

    # 1. Fresh random structure + CPDs for this replicate
    model = build_random_ground_truth_bn(
        node_states=NODE_STATES, max_indegree=3, edge_prob=0.4, seed=seed
    )
    model = relabel_states(model)
    validity = check_validity(model)
    if not validity["is_valid_bn"]:
        raise RuntimeError(f"Seed {seed} produced an invalid BN: {validity}")

    # 2. Fresh sampled dataset from this replicate's own structure
    df = sample_synthetic_dataset(model, n_samples=N_SAMPLES, seed=seed)

    # 3. Automatic Vm/Vo selection
    vm_vars, vo_vars = select_vm_vo(model, rng, n_vm=args.n_vm)

    # 4. Automatic R-parent selection per Vm variable
    r_parents = {vm: select_r_parent(model, vm, vo_vars, rng) for vm in vm_vars}

    # 5. Automatic CPT derivation per Vm variable, per sparsity level
    cpts = {}
    for vm_var in vm_vars:
        parent_col = r_parents[vm_var]
        marginal = df[parent_col].value_counts(normalize=True)
        # enforce the parent's canonical state order (not frequency order)
        canonical_order = VARIABLE_STATE_LABELS.get(parent_col, list(marginal.index))
        marginal = marginal.reindex(canonical_order).dropna()

        cpts[vm_var] = {target: derive_cpt(marginal, target) for target in SPARSITY_LEVELS}

    # 6. Inject missingness at each sparsity level
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    MISSING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"model_seed{seed}.bif"
    model.save(str(model_path), filetype="bif")
    print(f"Saved full network (structure + CPDs) -> {model_path}")

    gt_path = GROUND_TRUTH_DIR / f"ground_truth_seed{seed}.csv"
    df.to_csv(gt_path, index=False)
    print(f"Saved fresh ground truth -> {gt_path}")

    print(f"\n=== Replicate seed={seed} ===")
    print(f"Vm (missingness targets): {vm_vars}")
    print(f"Vo (fully observed):      {vo_vars}")
    print(f"R-parent assignments:     {r_parents}\n")

    achieved_summary = {}
    for sparsity in SPARSITY_LEVELS:
        df_missing = inject_missingness(df, vm_vars, r_parents, cpts, sparsity, rng)

        print(f"--- Target sparsity: {sparsity:.0%} ---")
        achieved_summary[sparsity] = {}
        for vm_var in vm_vars:
            achieved = df_missing[vm_var].isna().mean()
            achieved_summary[sparsity][vm_var] = round(achieved, 4)
            print(f"  {vm_var:<20s} achieved: {achieved:.4f} ({achieved:.1%})")

        out_path = MISSING_DIR / f"missing_{int(sparsity*100)}pct_seed{seed}.csv"
        df_missing.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

    # 7. Audit log -- every automated choice made for this replicate
    audit = {
        "seed": seed,
        "edges": list(model.edges()),
        "vm_vars": vm_vars,
        "vo_vars": vo_vars,
        "r_parents": r_parents,
        "cpts": {vm: {str(t): cpt for t, cpt in tgt_cpts.items()} for vm, tgt_cpts in cpts.items()},
        "achieved_rates": {str(t): rates for t, rates in achieved_summary.items()},
        "validity_checks": {k: v for k, v in validity.items() if k != "in_degrees"},
    }
    audit_path = AUDIT_DIR / f"audit_seed{seed}.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nSaved audit log -> {audit_path}")


if __name__ == "__main__":
    main()