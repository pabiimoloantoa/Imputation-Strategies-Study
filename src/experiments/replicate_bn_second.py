"""
THIS IS THE ONE YOU HAD AFTER THE VERY FIRST ONE(THE ONE THAT PRODUCED ONE R PARENT FOR MANY MISSING VARIABLES)
generate_replicate.py

Runs one full, self-contained replicate of the RQ1 data-generation pipeline.
See project methodology notes for the full step-by-step description.

Validity enforcement: each candidate network is checked against the same
pre-registered criteria used for the original fixed network -- (i) valid
BN/CPDs, (ii) a single connected component, (iii) no fully isolated nodes.
If a seed's network fails any of these, generation is retried with a fixed,
logged offset (seed + 1000 * attempt) rather than being silently accepted
or hand-picked -- this is a mechanical, pre-specified substitution rule,
applied identically every time, not a case-by-case judgment call.

Usage:
    python replicate_bn.py --seed 0
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from random_bn import (
    NODE_STATES,
    VARIABLE_STATE_LABELS,
    SINK_NODES,
    build_random_ground_truth_bn,
    relabel_states,
    check_validity,
    sample_synthetic_dataset,
)

# --- Paths (script lives in src/experiment/, so project root is 3 levels up) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "synthetic" / "ground_truth_per_seed"
MISSING_DIR = PROJECT_ROOT / "data" / "synthetic" / "missing"
AUDIT_DIR = PROJECT_ROOT / "data" / "synthetic" / "audit_logs"
MODELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "models"

N_SAMPLES = 5000
SPARSITY_LEVELS = [0.10, 0.20, 0.30]
DEFAULT_N_VM = 5
MAX_PROBABILITY = 0.95
MAX_RETRY_ATTEMPTS = 5
RETRY_OFFSET = 1000


def build_valid_network(seed: int, max_attempts: int = MAX_RETRY_ATTEMPTS):
    """
    Builds a network for the given seed, enforcing the pre-registered
    validity criteria (valid BN, single connected component, no isolated
    nodes). If the criteria fail, retries with a fixed, logged offset
    (seed + 1000, +2000, ...) rather than hand-picking a "nicer" seed.
    Returns (model, validity_dict, actual_seed_used).
    """
    for attempt in range(max_attempts):
        trial_seed = seed + RETRY_OFFSET * attempt
        model = build_random_ground_truth_bn(
            node_states=NODE_STATES, max_indegree=3, edge_prob=0.4,
            seed=trial_seed, sink_nodes=SINK_NODES,
        )
        model = relabel_states(model)
        validity = check_validity(model)

        is_ok = (
            validity["is_valid_bn"]
            and validity["is_single_connected_component"]
            and len(validity["isolated_nodes"]) == 0
        )
        if is_ok:
            if attempt > 0:
                warnings.warn(
                    f"Seed {seed} failed validity criteria; used fallback "
                    f"seed {trial_seed} (attempt {attempt}) instead."
                )
            return model, validity, trial_seed

        warnings.warn(
            f"Seed {trial_seed} failed validity criteria "
            f"(connected={validity['is_single_connected_component']}, "
            f"isolated={validity['isolated_nodes']}); retrying with offset seed."
        )

    raise RuntimeError(
        f"Could not generate a valid network for seed {seed} within {max_attempts} attempts."
    )


def select_vm_vo(model, rng: np.random.Generator, n_vm: int):
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_vm", type=int, default=DEFAULT_N_VM)
    args = parser.parse_args()

    requested_seed = args.seed
    rng = np.random.default_rng(requested_seed)

    # 1. Fresh, VALIDATED random structure + CPDs for this replicate
    model, validity, actual_seed = build_valid_network(requested_seed)

    # 2. Fresh sampled dataset (sampling uses the seed that actually produced
    #    a valid network, for full reproducibility of this replicate)
    df = sample_synthetic_dataset(model, n_samples=N_SAMPLES, seed=actual_seed)

    # 3. Automatic Vm/Vo selection
    vm_vars, vo_vars = select_vm_vo(model, rng, n_vm=args.n_vm)

    # 4. Automatic R-parent selection per Vm variable
    r_parents = {vm: select_r_parent(model, vm, vo_vars, rng) for vm in vm_vars}

    # 5. Automatic CPT derivation per Vm variable, per sparsity level
    cpts = {}
    for vm_var in vm_vars:
        parent_col = r_parents[vm_var]
        marginal = df[parent_col].value_counts(normalize=True)
        canonical_order = VARIABLE_STATE_LABELS.get(parent_col, list(marginal.index))
        marginal = marginal.reindex(canonical_order).dropna()
        cpts[vm_var] = {target: derive_cpt(marginal, target) for target in SPARSITY_LEVELS}

    # 6. Save the network, inject missingness, save everything
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    MISSING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Files are named by the ORIGINAL requested seed, for consistent
    # referencing, even if a fallback seed was used internally.
    model_path = MODELS_DIR / f"model_seed{requested_seed}.bif"
    model.save(str(model_path), filetype="bif")
    print(f"Saved full network (structure + CPDs) -> {model_path}")

    gt_path = GROUND_TRUTH_DIR / f"ground_truth_seed{requested_seed}.csv"
    df.to_csv(gt_path, index=False)
    print(f"Saved fresh ground truth -> {gt_path}")

    print(f"\n=== Replicate seed={requested_seed} (actual generation seed: {actual_seed}) ===")
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

        out_path = MISSING_DIR / f"missing_{int(sparsity*100)}pct_seed{requested_seed}.csv"
        df_missing.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

    audit = {
        "requested_seed": requested_seed,
        "actual_generation_seed": actual_seed,
        "fallback_used": actual_seed != requested_seed,
        "edges": list(model.edges()),
        "vm_vars": vm_vars,
        "vo_vars": vo_vars,
        "r_parents": r_parents,
        "cpts": {vm: {str(t): cpt for t, cpt in tgt_cpts.items()} for vm, tgt_cpts in cpts.items()},
        "achieved_rates": {str(t): rates for t, rates in achieved_summary.items()},
        "validity_checks": {k: v for k, v in validity.items() if k != "in_degrees"},
    }
    audit_path = AUDIT_DIR / f"audit_seed{requested_seed}.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nSaved audit log -> {audit_path}")


if __name__ == "__main__":
    main()