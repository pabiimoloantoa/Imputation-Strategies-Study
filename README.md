# Bayesian Network Structure Learning under Missing Data

Imputation strategy evaluation on a synthetic credit risk dataset (RQ1) and
the Pima Indians Diabetes dataset (RQ2).

Paballo Moloantoa (2451669) — Wits Honours, supervised by Prof. Ritesh Ajoodha.

---

## Important: this replaces an earlier, hand-designed version

An earlier version of this project used a **manually designed** ground-truth
Bayesian Network (hand-placed edges, hand-tuned CPD probabilities). Prof.
Ajoodha flagged that this introduces **selection bias**: a human-designed
"ground truth" tends to encode the designer's own assumptions about what
the relationships *should* look like, which can quietly bias the evaluation
toward finding what was expected.

This version replaces that approach entirely. The ground-truth network's
**structure and CPD parameters are generated algorithmically at random**,
given only high-level constraints. No edge was placed and no probability
was chosen by hand anywhere in the generation code.

**A further revision (see "Independently generated replicates" below):**
the initial version of this redesign validated the approach on a single
fixed random network (`seed=42`). Following further discussion with Prof.
Ajoodha, the design was extended so that **each of the study's 20
replicates is built on its own independently and randomly generated
network**, rather than reusing one fixed network across all replicates.
The single-network version is retained in the repository as a worked
pilot example (see `src/pilot/`), since it remains useful for illustrating
the missingness mechanism concretely, but it is no longer what the final
RQ1 results are generated from.

---

## How the ground-truth BN is generated

**Step 1 — research real-world characteristics.** Before generating
anything, we researched credit-risk datasets that have genuine missing
data (HMEQ, Give Me Some Credit, Australian Credit Approval — see summary
below) to ground the generator's constraints in evidence rather than
guesswork.

**Step 2 — fix the variable list and state counts (before any generation).**
11 named variables (10 predictors + `Default`), decided upfront so names
can never be reverse-fitted to whatever structure the algorithm produces:

| Variable | States | Type |
|---|---|---|
| Age | 3 | continuous-like |
| EmploymentStatus | 3 | categorical (employed/unemployed/self_employed) |
| Income | 3 | continuous-like |
| DebtRatio | 3 | continuous-like |
| CreditUtilization | 3 | continuous-like |
| PaymentHistory | 4 | categorical (poor/fair/good/excellent) |
| NumberOfDefaults | 3 | count (0/1/2plus) |
| LoanAmount | 3 | continuous-like |
| CreditHistoryLength | 3 | continuous-like |
| NumberOfCreditInquiries | 3 | count |
| **Default** (target) | 2 | binary |

State counts reflect each variable's real-world type (continuous → more
bins, true categories/counts → their natural number of states) — this is
faithful representation, not a lever used to shape results.

**Step 3 — generate structure + parameters in one call, at random.**
`build_random_ground_truth_bn()`:
1. Randomly shuffles the node order (this IS the topological order, so
   acyclicity is guaranteed by construction — edges only ever point from
   an earlier node to a later one). **One exception:** `Default` (the
   target variable) is always placed last in this order, so it can be
   *caused by* other variables but can never causally influence them —
   consistent with its role as an outcome rather than a predictor. This
   is the only constraint placed on top of full randomization; everything
   else about the structure remains unconstrained.
2. For each node, randomly selects up to `max_indegree` parents from the
   nodes before it (`generate_random_dag()`).
3. Calls pgmpy's `get_random_cpds()` to auto-generate fully valid, random
   CPDs for the resulting structure — no probability is chosen by hand.

**Step 4 — pre-registered validity checks, enforced automatically for
every replicate**, to avoid the temptation to "reroll" a seed until
something looks nicer (which would just reintroduce selection bias one
level removed):
- Must be a single connected component (no isolated sub-groups of nodes)
- No fully isolated nodes
- Observed max in-degree must not exceed the constraint
- No single fully-observed variable may serve as the R-parent (see below)
  for more than 2 of the missingness-target variables, so that one "hub"
  variable doesn't end up driving most of the study's missingness by chance

If a candidate replicate fails any of these checks, generation is
automatically retried with a fixed, logged offset seed (`seed + 1000`,
then `+2000`, ...) — a mechanical, pre-specified substitution rule applied
identically every time, never a manual choice of which result "looks
better." Every substitution is recorded in that replicate's audit log
(see below).

`seed=42` was used for the original single-network pilot and passed all
checks on the first attempt.

### Generator constraints, and why

| Constraint | Value | Justification |
|---|---|---|
| `n_nodes` | 11 | Matches Give Me Some Credit (10+1) closely; HMEQ has 12+1 |
| `max_indegree` | 3 | Standard for small/medium benchmark BNs (e.g. ASIA/ALARM-style networks typically use 2–4) |
| `edge_prob` | 0.4 | Moderate density — sparse enough for structure learning to have real work to do |
| Variable type mix | ~7 continuous-like : 3 categorical/count : 1 binary target | Mirrors HMEQ's ~10:2 continuous-to-categorical ratio, scaled to 11 nodes |
| `Default` position | always last in topological order (sink) | Keeps the target variable from causally influencing its own predictors |

### What we found researching real credit-risk datasets with missing data

| Dataset | Variables | Continuous | Categorical | Missing rate | Notes |
|---|---|---|---|---|---|
| HMEQ | 12+1 | ~10 | 2 | ~44% of rows have ≥1 missing (cell-level ~6%) | DEBTINC alone reaches ~21% — one variable dominates |
| Give Me Some Credit | 10+1 | ~9 | ~1 | MonthlyIncome ~19.8% missing; rest low | Same "one dominant variable" pattern as HMEQ |
| Australian Credit Approval (UCI) | 14+1 | 6 | 8 | ~5% of rows | Missingness thinly scattered across 7 of 14 attributes |
| Taiwan Credit Card Default | 23–24 | 14 | 8+2 binary | 0% | Clean benchmark; useful only for type-mix reference |
| German Credit | 20+1 | 7 | 14 | 0% | Clean benchmark, opposite type-mix to HMEQ |
| Lending Club | 75–145+ | mixed | mixed | >25% overall, many columns near-100% | Extreme outlier; missingness is *temporal* (fields not collected before a certain date) — a real-world example of genuine MAR (depends on an observed variable, issue date, not the value itself) |

This review directly motivated adopting an explicit, structurally-grounded
MAR mechanism (below) rather than simple uniform (MCAR) deletion.

---

## MAR missing data injection mechanism

Rather than deleting values uniformly at random (MCAR), missingness is
injected using a structurally-grounded Missing At Random (MAR) mechanism
based on the missingness graph (m-graph) framework of Mohan, Pearl, and
Tian (2013). The mechanism is part of the Bayesian Network itself: the
missingness indicators are real nodes with real edges and CPDs, not a
rule applied to the data after sampling.

For each variable selected to receive missing values (a **Vm** variable),
and for each sparsity level, an auxiliary binary response-indicator node,
`R_X`, is added to the network. `R_X` determines, per row, whether `X`'s
true value is observed (`R_X = 0`, state "observed") or hidden
(`R_X = 1`, state "missing"). The column actually presented to the
imputation methods — the **proxy**, `X*` — equals `X` when observed and
`NA` when hidden; the true, complete values are retained separately for
evaluation but are never visible to any imputation method.

To preserve MAR, each `R_X` node has exactly one parent (its "R-parent"),
which is restricted to a variable that stays fully observed throughout
(a **Vo** variable) and is one of `X`'s own direct parents in the graph,
so that `R_X ⊥ X | Pa(R_X)`: missingness may depend on other observed
variables, but never on the value being hidden, nor on another
partially-observed variable. Each `R_X` is a leaf node (edge in from its
R-parent, none out), so adding the indicators leaves the distribution of
the original 11 variables unchanged.

Each `R_X` node's conditional probability table (CPT) is derived
analytically: given the R-parent's exact marginal distribution (obtained
by inference on that replicate's own network) and rank-based multipliers
across the parent's states, the per-state probabilities are scaled so
their marginal-weighted average equals the target sparsity level (10%,
20%, or 30%, applied **per Vm column** — confirmed with Prof. Ajoodha).
Because the multipliers differ across parent states, missingness depends
on the R-parent (MAR rather than MCAR). All probabilities are capped at a
0.95 safety ceiling.

The augmented network (original variables plus all `R_X` nodes) is
forward-sampled once. The original columns form the ground truth, and
each `missing_<pct>pct` dataset blanks `X` wherever `R_X` is "missing".
The expected missingness rate implied by each CPT equals the target; the
achieved rate in a finite sample approximates it, with both logged in the
audit file.

### Independently generated replicates

Following further discussion with Prof. Ajoodha, each of the study's 20
replicates is built on its **own** randomly generated network (structure
+ CPDs + sampled data), rather than one fixed network reused across all
replicates. This tests whether an imputation method's apparent advantage
holds up across genuinely distinct random structures, not just across
different missingness patterns applied to one structure.

This required automating steps that were done by hand for the original
pilot network:

- **Vm/Vo selection** — variables are chosen automatically for missingness
  injection, restricted to those with at least one parent (so an R-parent
  is available). 5 variables are selected per replicate.
- **R-parent selection** — each Vm variable's R-parent is chosen from its
  own direct parents that lie in Vo (preserving a defensible causal
  rationale), preferring a parent not already assigned to another Vm
  variable. If a Vm variable has no direct parent in Vo, the replicate is
  rejected and regenerated with the next offset seed, rather than
  assigning an unrelated Vo variable with no edge to it.
- **CPT derivation** — since variable identity/semantics aren't stable
  across random structures, a fixed, content-neutral rank-based ratio
  (1×, 1.5×, 2×, ... across a parent's states in natural order) is applied
  uniformly, rather than the substantive per-variable "stories" used in
  the pilot network (e.g. "higher debt ratio → less income disclosure").
  The resulting CPTs are embedded in the network as the `R_X` nodes' CPDs.

Every replicate's full set of automated choices — network edges, R-node
edges, Vm/Vo membership, R-parent assignments, derived CPTs, expected and
achieved missingness rates, and any retry triggered — is written to a
per-replicate JSON audit log. Alongside it, two networks are saved in
`.bif` format with structure and CPDs preserved: the generated network
(`model_seed<N>.bif`) and the augmented network with R-nodes
(`model_with_r_nodes_seed<N>.bif`), plus the ground-truth and
missingness-injected datasets. This allows every replicate's construction
to be verified after the fact without manual re-inspection.

**Limitation.** This generalisation trades the domain-specific narrative
plausibility of the pilot design (a stated real-world rationale for each
missingness dependency) for a uniform, reproducible rule applicable to
arbitrary structures. The rank-based multiplier scheme does not correspond
to a claimed real-world mechanism; its role is to impose a consistent,
non-trivial MAR dependency structure across replicates.

---

## Repo structure

```
bn-imputation-study/
├── README.md                          — this file
├── requirements.txt
├── .gitignore
├── src/
│   ├── pilot/                         — original hand-worked single-network example (seed=42)
│   │   ├── generate_ground_truth_bn.py
│   │   ├── compute_marginals.py
│   │   ├── inject_mar_missingness.py
│   │   ├── check_missingness.py
│   │   └── plot_*.py                  — pilot figures (mechanism diagram, CPT curves, marginals, highlighted structure)
│   └── experiment/                    — automated per-replicate pipeline (used for final RQ1 results)
│       ├── generate_ground_truth_bn.py    — network generator (with Default-as-sink constraint)
│       ├── generate_replicate.py          — full per-seed pipeline: network, Vm/Vo, R-parents, CPTs, missingness
│       ├── plot_bn_from_bif.py            — plain structure diagram for any saved replicate
│       └── plot_replicate_structure.py    — structure diagram with Vm/Vo/R-parent edges highlighted
├── data/
│   ├── synthetic/
│   │   ├── ground_truth_complete.csv          — pilot: sampled 5000-row complete dataset (seed=42)
│   │   ├── missing/                           — pilot + per-replicate missingness-injected CSVs
│   │   ├── ground_truth_per_seed/             — per-replicate fresh ground-truth CSVs
│   │   ├── models/                            — per-replicate saved networks (.bif, structure + CPDs)
│   │   └── audit_logs/                        — per-replicate JSON audit logs
│   ├── raw/                            — empty, for the Pima Indians Diabetes dataset (Phase 4)
│   └── processed/                      — empty, for post-imputation datasets
├── figures/
│   ├── random_bn_structure.png             — pilot ground-truth DAG
│   ├── mar_missingness_mechanism.png       — R-node mechanism diagram (pilot)
│   ├── r_node_cpt_curves.png               — derived CPTs by parent state (pilot)
│   ├── r_parent_marginals.png              — R-parent marginal distributions (pilot)
│   ├── ground_truth_bn_highlighted.png     — pilot structure, Vm highlighted
│   └── replicate_structure_seed*.png       — per-replicate structure diagrams
├── results/
│   ├── rq1/                            — empty, for SHD/KL results (Phase 3)
│   └── rq2/                            — empty, for AUC/F1/etc. results (Phase 5)
├── notebooks/                          — empty, for exploratory work
└── configs/                            — empty, for hyperparameter configs (Table 3.2 values)
```

## Status

- [x] Literature review of real credit-risk-with-missing-data dataset characteristics
- [x] Random ground-truth BN generator built (structure + CPDs, no manual tuning)
- [x] Pre-registered validity checks defined (connectivity, isolated nodes, in-degree, R-parent concentration), with automatic retry on failure
- [x] Pilot: complete 5000-row synthetic dataset sampled and saved (seed=42)
- [x] MAR missingness injection mechanism designed (m-graph / R-node framework) and validated on the pilot network
- [x] Automated per-replicate pipeline built: independent random network + Vm/Vo selection + R-parent assignment + CPT derivation per seed
- [ ] Full 20-replicate batch run
- [ ] Pima Indians Diabetes dataset download + preprocessing (RQ2, Phase 4)
- [ ] Imputation methods: EM, MICE, Soft-Impute, BN-Gibbs
- [ ] Structure learning (Hill Climbing + BIC) and evaluation (SHD, KL-divergence) — RQ1
- [ ] Predictive evaluation (AUC-ROC, F1, precision, recall) — RQ2

## How to reproduce

**Pilot (single fixed network, seed=42):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd src/pilot
python generate_ground_truth_bn.py
python inject_mar_missingness.py
```

**Full experiment (independent replicates):**
```powershell
cd src/experiment
python generate_replicate.py --seed 0
python plot_replicate_structure.py --seed 0
```
Repeat for each seed (0–19) to produce the full replicate set.