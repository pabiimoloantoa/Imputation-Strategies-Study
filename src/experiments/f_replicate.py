"""
FIXED ONE. THE ONE I USED IN THIS EXPERIMENT.
generate_replicate.py  (R-nodes embedded in the pgmpy network)

Runs one full, self-contained replicate of the RQ1 data-generation pipeline.

CHANGE FROM THE PREVIOUS VERSION
    Missingness is no longer injected by an external coin-flip loop after
    sampling. Instead, for every Vm variable and every sparsity level
    (10/20/30%) a real indicator node R_<var>_<pct>pct is added to the pgmpy
    Bayesian Network:
        - a real edge   R-parent (a Vo variable) -> R_<var>_<pct>pct
        - a real TabularCPD  P(R | R-parent), states ["observed", "missing"]
    The target sparsity is encoded directly in the CPD parameters. The rows
    still differ by parent state (so the mechanism stays MAR, not MCAR) and
    are calibrated so that  sum_s P(parent=s) * P(missing | s) == target,
    where P(parent) comes from EXACT inference on the original network
    (not from the sampled data).
    The augmented network (11 + 3 * |Vm| nodes) is forward-sampled ONCE.
    The original 11 columns are the complete ground truth; each missing CSV
    blanks a Vm variable wherever its R-node samples "missing".

Validity enforcement (unchanged): a candidate replicate (network + Vm/Vo
split + R-parent assignments) is accepted only if it satisfies ALL of:
  (i)   valid BN / consistent CPDs
  (ii)  single connected component, no isolated nodes
  (iii) no single Vo variable serves as R-parent for more than
        MAX_R_PARENT_SHARE Vm variables
  (iv)  every Vm variable's R-parent is one of its OWN direct parents in Vo
If any criterion fails, generation is retried with a fixed offset seed
(seed + 1000, +2000, ...).

Usage:
    python generate_replicate.py --seed 0
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

from random_bn import (
    NODE_STATES,
    VARIABLE_STATE_LABELS,
    SINK_NODES,
    build_random_ground_truth_bn,
    relabel_states,
    check_validity,
)

# --- Paths (script lives in src/experiment/, so project root is 3 levels up) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "synthetic" / "ground_truth_per_seed"
MISSING_DIR = PROJECT_ROOT / "data" / "synthetic" / "missing"
AUDIT_DIR = PROJECT_ROOT / "data" / "synthetic" / "audit_logs"
MODELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "models"

N_SAMPLES = 5000
SPARSITY_LEVELS = [0.10, 0.20, 0.30, 0.40]
DEFAULT_N_VM = 5
MAX_PROBABILITY = 0.95
MAX_RETRY_ATTEMPTS = 8
RETRY_OFFSET = 1000
MAX_R_PARENT_SHARE = 2

R_STATES = ["observed", "missing"]  # index 0 = observed, index 1 = missing


def r_node_name(vm_var: str, sparsity: float) -> str:
    return f"R_{vm_var}_{int(round(sparsity * 100))}pct"


# ----------------------------------------------------------------------------
# Replicate construction (unchanged logic)
# ----------------------------------------------------------------------------

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


def select_r_parent(model, vm_var: str, vo_vars: list, rng: np.random.Generator,
                    used_parents: set):
   
    direct_parents = model.get_parents(vm_var)
    eligible_direct = [p for p in direct_parents if p in vo_vars]

    if not eligible_direct:
        return None, False

    unused = [p for p in eligible_direct if p not in used_parents]
    if unused:
        return str(rng.choice(unused)), True

    chosen = str(rng.choice(eligible_direct))
    warnings.warn(
        f"'{vm_var}': all of its eligible direct parents {eligible_direct} "
        f"are already used as another Vm variable's R-parent; reusing '{chosen}'."
    )
    return chosen, True


def build_valid_replicate(seed: int, n_vm: int, max_attempts: int = MAX_RETRY_ATTEMPTS):
   
    for attempt in range(max_attempts):
        trial_seed = seed + RETRY_OFFSET * attempt

        model = build_random_ground_truth_bn(
            node_states=NODE_STATES, max_indegree=3, edge_prob=0.4,
            seed=trial_seed, sink_nodes=SINK_NODES,
        )
        model = relabel_states(model)
        validity = check_validity(model)

        structurally_ok = (
            validity["is_valid_bn"]
            and validity["is_single_connected_component"]
            and len(validity["isolated_nodes"]) == 0
        )
        if not structurally_ok:
            warnings.warn(
                f"Seed {trial_seed} failed structural validity "
                f"(connected={validity['is_single_connected_component']}, "
                f"isolated={validity['isolated_nodes']}); retrying with offset seed."
            )
            continue

        rng = np.random.default_rng(trial_seed)
        vm_vars, vo_vars = select_vm_vo(model, rng, n_vm=n_vm)

        r_parents = {}
        used_parents = set()
        all_have_real_parent = True
        for vm in vm_vars:
            parent, ok = select_r_parent(model, vm, vo_vars, rng, used_parents)
            if not ok:
                warnings.warn(
                    f"Seed {trial_seed} rejected: '{vm}' has no direct parent in Vo "
                    f"(all of its real parents are themselves in Vm); retrying with offset seed."
                )
                all_have_real_parent = False
                break
            r_parents[vm] = parent
            used_parents.add(parent)

        if not all_have_real_parent:
            continue

        parent_counts = Counter(r_parents.values())
        max_share = max(parent_counts.values()) if parent_counts else 0
        if max_share > MAX_R_PARENT_SHARE:
            warnings.warn(
                f"Seed {trial_seed} rejected: R-parent '{parent_counts.most_common(1)[0][0]}' "
                f"is used by {max_share} Vm variables (max allowed: {MAX_R_PARENT_SHARE}); "
                f"retrying with offset seed."
            )
            continue

        if attempt > 0:
            warnings.warn(
                f"Seed {seed} required fallback; used seed {trial_seed} "
                f"(attempt {attempt}) instead."
            )
        return model, validity, vm_vars, vo_vars, r_parents, rng, trial_seed

    raise RuntimeError(
        f"Could not generate a valid replicate for seed {seed} within {max_attempts} attempts."
    )


# ----------------------------------------------------------------------------
# Missingness CPDs (new)
# ----------------------------------------------------------------------------

def exact_parent_marginal(model, parent: str) -> pd.Series:
    """P(parent) from exact inference on the ORIGINAL 11-node network."""
    infer = VariableElimination(model)
    factor = infer.query([parent], show_progress=False)
    return pd.Series(factor.values, index=factor.state_names[parent], dtype=float)


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


def build_r_node_cpd(model, vm_var: str, parent: str, sparsity: float,
                     cpt: dict) -> TabularCPD:
   
    parent_states = model.get_cpds(parent).state_names[parent]
    missing_row = []
    for s in parent_states:
        if s not in cpt:
            raise KeyError(f"No derived P(missing) for parent '{parent}' state '{s}'.")
        missing_row.append(cpt[s])
    observed_row = [1.0 - p for p in missing_row]

    name = r_node_name(vm_var, sparsity)
    return TabularCPD(
        variable=name,
        variable_card=2,
        values=[observed_row, missing_row],
        evidence=[parent],
        evidence_card=[len(parent_states)],
        state_names={name: R_STATES, parent: list(parent_states)},
    )


def embed_r_nodes(model, vm_vars, r_parents):
    
    aug = model.copy()
    cpts, expected = {}, {s: {} for s in SPARSITY_LEVELS}

    for vm in vm_vars:
        parent = r_parents[vm]
        marginal = exact_parent_marginal(model, parent)
        canonical = VARIABLE_STATE_LABELS.get(parent, list(marginal.index))
        marginal = marginal.reindex(canonical).dropna()
        marginal = marginal / marginal.sum()

        cpts[vm] = {}
        for sparsity in SPARSITY_LEVELS:
            cpt = derive_cpt(marginal, sparsity)
            cpts[vm][sparsity] = cpt
            expected[sparsity][vm] = round(
                float(sum(marginal[s] * cpt[s] for s in marginal.index)), 4
            )

            name = r_node_name(vm, sparsity)
            aug.add_node(name)
            aug.add_edge(parent, name)
            aug.add_cpds(build_r_node_cpd(model, vm, parent, sparsity, cpt))

    if not aug.check_model():
        raise RuntimeError("Augmented network failed check_model().")
    return aug, cpts, expected


def sample_augmented(aug_model, n_samples: int, seed: int) -> pd.DataFrame:
    sampler = BayesianModelSampling(aug_model)
    return sampler.forward_sample(size=n_samples, seed=seed, show_progress=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_vm", type=int, default=DEFAULT_N_VM)
    args = parser.parse_args()

    requested_seed = args.seed

    model, validity, vm_vars, vo_vars, r_parents, rng, actual_seed = build_valid_replicate(
        requested_seed, n_vm=args.n_vm
    )
    original_nodes = list(model.nodes())

    aug_model, cpts, expected_rates = embed_r_nodes(model, vm_vars, r_parents)
    print(f"Augmented network: {len(model.nodes())} -> {len(aug_model.nodes())} nodes, "
          f"{len(model.edges())} -> {len(aug_model.edges())} edges")

    sample = sample_augmented(aug_model, N_SAMPLES, seed=actual_seed)

    # Complete data = the original variables only (R columns are not data).
    df = sample[original_nodes].copy()

    for d in (GROUND_TRUTH_DIR, MISSING_DIR, AUDIT_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"model_seed{requested_seed}.bif"
    model.save(str(model_path), filetype="bif")
    print(f"Saved original network -> {model_path}")

    aug_path = MODELS_DIR / f"model_with_r_nodes_seed{requested_seed}.bif"
    aug_model.save(str(aug_path), filetype="bif")
    print(f"Saved augmented network (with R-nodes) -> {aug_path}")

    gt_path = GROUND_TRUTH_DIR / f"ground_truth_seed{requested_seed}.csv"
    df.to_csv(gt_path, index=False)
    print(f"Saved ground truth -> {gt_path}")

    print(f"\n=== Replicate seed={requested_seed} (actual generation seed: {actual_seed}) ===")
    print(f"Vm (missingness targets): {vm_vars}")
    print(f"Vo (fully observed):      {vo_vars}")
    print(f"R-parent assignments:     {r_parents}\n")

    achieved_summary = {}
    for sparsity in SPARSITY_LEVELS:
        df_missing = df.copy()
        achieved_summary[sparsity] = {}
        print(f"--- Target sparsity: {sparsity:.0%} ---")
        for vm_var in vm_vars:
            r_col = r_node_name(vm_var, sparsity)
            is_missing = (sample[r_col] == "missing").values
            df_missing.loc[is_missing, vm_var] = np.nan

            achieved = float(df_missing[vm_var].isna().mean())
            achieved_summary[sparsity][vm_var] = round(achieved, 4)
            print(f"  {vm_var:<20s} expected: {expected_rates[sparsity][vm_var]:.4f}  "
                  f"achieved: {achieved:.4f}")

        out_path = MISSING_DIR / f"missing_{int(round(sparsity * 100))}pct_seed{requested_seed}.csv"
        df_missing.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

    audit = {
        "requested_seed": requested_seed,
        "actual_generation_seed": actual_seed,
        "fallback_used": actual_seed != requested_seed,
        "edges": list(model.edges()),
        "r_node_edges": [(r_parents[vm], r_node_name(vm, s))
                         for vm in vm_vars for s in SPARSITY_LEVELS],
        "vm_vars": vm_vars,
        "vo_vars": vo_vars,
        "r_parents": r_parents,
        "cpts": {vm: {str(t): cpt for t, cpt in tgt.items()} for vm, tgt in cpts.items()},
        "expected_rates": {str(t): r for t, r in expected_rates.items()},
        "achieved_rates": {str(t): r for t, r in achieved_summary.items()},
        "validity_checks": {k: v for k, v in validity.items() if k != "in_degrees"},
    }
    audit_path = AUDIT_DIR / f"audit_seed{requested_seed}.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nSaved audit log -> {audit_path}")


if __name__ == "__main__":
    main()


    