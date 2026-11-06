# Bayesian Network Structure Learning under Missing Data

How imputation methods distort the learned network and downstream prediction.

Code, data and results for the paper *Bayesian Network Structure Learning under Missing Data: How Imputation Methods Distort the Learned Network and Downstream Prediction* (submitted to Applied Soft Computing).

Paballo Moloantoa (2451669), BSc Honours, University of the Witwatersrand. Supervisor: Prof. Ritesh Ajoodha, School of Computer Science and Applied Mathematics.

Repository: https://github.com/pabiimoloantoa/Imputation-Strategies-Study

---

## What this project does

Three imputation methods are compared in three studies:

- **EM** (Gaussian expectation maximisation with the full M-step)
- **MICE** (chained equations with decision trees, single imputation)
- **Soft-Impute** (low rank matrix completion by singular value shrinkage)

Each method fills the blanks, a Bayesian network is learned from the completed data by Hill Climbing with the BIC score, and the result is scored. A baseline is added in each study: the true values (the "perfect" baseline) in Study 1 and the median fill in Studies 2 and 3.

| Study | Research question | Data | Why this data |
|---|---|---|---|
| 1 | RQ1 | 20 synthetic credit risk networks, 5,000 records each, missingness at 10%, 20%, 30%, 40% | True structure and true missingness mechanism are known, so the learned network can be scored against the truth |
| 2 | RQ2 (i) | Pima Indians Diabetes (768 patients) | The missing values are real, so methods can only be judged through prediction |
| 3 | RQ2 (ii) | Taiwan credit card default (30,000 customers, 10 predictors), missingness injected at 10% to 40% | The true values are known, so filled value accuracy and prediction can both be measured |

### Main findings

- **Study 1.** MICE was indistinguishable from perfect imputation. EM and Soft-Impute raised the KL divergence of the learned network 12 to 17 times at 30% to 40% missingness and left extra links between the missingness indicators and the variables they filled (the "footprint").
- **Study 2.** EM and MICE gave a slightly higher AUC than the median fill (0.814 and 0.810 against 0.804). The differences are small relative to fold to fold variation and to the run to run variability of Hill Climbing.
- **Study 3.** All three methods roughly halved the error of the filled values relative to the median fill, yet AUC stayed within 0.014 of the complete data score (0.752) and gains over the median fill were at most 0.007.

Accurate imputation therefore does not guarantee a faithful network or better prediction.

---

## Study 1: synthetic networks with a known missingness mechanism

### How the ground truth networks are generated

Each replicate has its own network, generated at random. No edge was placed and no probability was chosen by hand.

**Variables.** 11 variables (10 predictors and the target `Default`), with names and state counts fixed before any network was generated:

| Variable | States | Type |
|---|---|---|
| Age | 3 (young / middle / senior) | continuous like |
| EmploymentStatus | 3 (employed / unemployed / self_employed) | categorical |
| Income | 3 (low / medium / high) | continuous like |
| DebtRatio | 3 | continuous like |
| CreditUtilization | 3 | continuous like |
| PaymentHistory | 4 (poor / fair / good / excellent) | categorical |
| NumberOfDefaults | 3 (0 / 1 / 2plus) | count |
| LoanAmount | 3 | continuous like |
| CreditHistoryLength | 3 (short / medium / long) | continuous like |
| NumberOfCreditInquiries | 3 | count |
| **Default** (target) | 2 | binary |

State counts reflect each variable's real world type. They are not a lever for shaping results.

**Structure.** A uniformly random permutation of the ten predictors gives a topological order, and `Default` is always placed last, so it can be caused by other variables but cannot influence them. Every earlier variable is a candidate parent and is included with probability `edge_prob = 0.4`; if more than `max_indegree = 3` parents are chosen, a random subset of three is kept. Edges only point from earlier to later positions, so the graph is acyclic by construction.

**Parameters.** For each variable and parent configuration, a probability vector is drawn by taking one uniform random number per state and normalising (pgmpy `get_random_cpds`).

**Constraints and why.**

| Constraint | Value | Justification |
|---|---|---|
| `n_nodes` | 11 | Matches Give Me Some Credit (10 predictors and a target) |
| `max_indegree` | 3 | Typical of small to medium benchmark networks (two to four) |
| `edge_prob` | 0.4 | Moderate density, sparse enough for structure learning to have real work to do |
| `Default` position | last in the topological order | Keeps the target from influencing its own predictors |

`max_indegree` and `edge_prob` cannot be read from tabular data, so they were set by convention. This is listed as a limitation in the paper. An empirical review of six credit risk datasets (HMEQ, Give Me Some Credit, Australian Credit Approval, German Credit, Taiwan Credit Card Default and Lending Club) informed the choice of 11 variables.

### Replicates and validity checks

Twenty replicates were built, each with its own network, probability tables and 5,000 sampled records. A candidate replicate is accepted only if all of these hold, fixed in advance:

- the network is valid and a single connected component, with no isolated variable
- its maximum in degree does not exceed three
- every incomplete variable has a valid R-parent (defined below)
- no fully observed variable is the R-parent of more than two incomplete variables

If a candidate fails, generation is retried with a fixed offset to the seed (seed + 1000, then + 2000, and so on), applied identically to every replicate and logged. Eleven of the 20 replicates were built from a retry seed.

### Missingness mechanism

Missingness is represented inside the network, following the missingness graph framework of Mohan, Pearl and Tian (2013). In each replicate, five variables with at least one parent are drawn at random to be incomplete (**Vm**) and the rest are fully observed (**Vo**).

Each incomplete variable `X` gets an indicator node `R_X` (one per sparsity level) with a single parent, its **R-parent**, chosen from the direct parents of `X` that lie in Vo. A replicate in which an incomplete variable has no such parent is rejected and regenerated. R-parents are spread over different Vo variables where possible. Because `R_X` depends only on a fully observed variable, whether a value is missing never depends on the missing value itself, so the mechanism is missing at random (MAR). Each `R_X` is a leaf node, so adding the indicators does not change the joint distribution of the original 11 variables.

The probability that a value is missing rises with the order of the R-parent's states in the ratio 1 : 1.5 : 2 : ..., scaled so that its average over the R-parent's marginal distribution equals the target sparsity (10%, 20%, 30% or 40%, per incomplete column), with a safety cap of 0.95. The R-parent's marginal comes from exact inference on that replicate's own network. The augmented network (11 variables plus the indicators) is forward sampled once; the complete columns are kept as ground truth and each incomplete dataset blanks a variable wherever its indicator says it is missing. The sparsity levels share the same complete data but use independent indicators.

The column shown to an imputation method is the proxy `X*`, equal to `X` when observed and blank when hidden. The true values are retained for scoring and are never visible to any method.

### Imputation, learning and evaluation

- Each method fills every blank with a single value. Study 1 data are categorical, so they are one hot encoded for EM and Soft-Impute and decoded by argmax (one reference column per variable is dropped for EM and restored afterwards). After each imputation every observed cell is checked to confirm it was returned unchanged.
- **EM** stops when the observed data log likelihood changes by less than 1e-4 or after 100 iterations. The M-step includes the conditional covariance of the filled values (see "Changes to the analysis" below).
- **MICE** uses fully grown decision tree classifiers on one hot predictors, ten rounds, most likely state as the fill.
- **Soft-Impute** uses a fixed shrinkage of 1.0 on the one hot matrix.
- Networks are learned by Hill Climbing with BIC, with no limit on the number of parents.
- The learning table holds the 11 variables plus the five indicator columns for the sparsity level, rebuilt from the blanks, and the learned graph is compared with the true 16 node network (11 variables plus 5 indicators).

Measures (mean and standard error over 20 replicates):

- Kullback-Leibler divergence in nats between the true and learned networks (parameters fitted on the scored data with a Bayesian estimator, pseudo count 1, joint distribution enumerated exactly)
- structural Hamming distance on the CPDAGs of the two graphs
- true edges from an observed parent to an indicator that were recovered (out of five)
- extra indicator links, meaning links between an indicator and a variable other than its true parent; links to its own variable are called own links

**Hash seed wobble.** Hill Climbing breaks ties in an order that depends on Python's `PYTHONHASHSEED`, so the learned graph can differ between runs on identical data. All final runs fix `PYTHONHASHSEED = 0`. The perfect baseline at 20% missingness was run on all 20 replicates under hash seeds 0 to 9 to measure the spread (the "wobble": about 0.028 nats for KL and 2.11 for SHD).

### Audit trail

For every replicate the network edges, the choice of Vm and Vo, the R-parents, the derived probability tables, the expected and achieved missingness rates and any retries are written to a JSON audit log. The generated network and the augmented network with indicators are saved in `.bif` format, together with the complete and incomplete datasets.

---

## Studies 2 and 3: prediction

Both studies use stratified 10 fold cross validation with every method evaluated on exactly the same folds. In each fold the imputer learns from the training records only, using the predictors and never the outcome; it then fills the training and test records without re-learning from the test records. Numeric variables are cut into three groups (low, medium, high) at the one third and two thirds quantiles of the **observed (non missing) training values**, so the binning is identical for every imputation method. A Bayesian network with at most three parents per variable and a BDeu prior (equivalent sample size 10) is learned on the training records, and the probability of the outcome given a test record's other variables is obtained by inference.

The main metric is AUC. F1, precision and recall at a threshold of 0.5 are also recorded. Each method is compared with the median fill on the same folds using the mean difference, the number of folds better, tied and worse, and a Wilcoxon signed rank test.

### Study 2: Pima Indians Diabetes

768 female patients, eight measurements and a binary outcome (diabetes in 268 patients, 34.9%). In five columns a value of zero is physiologically impossible and these zeros were recoded as missing. Almost all missing values are in Insulin (374) and SkinThickness (227), with 35 in BloodPressure, 11 in BMI and 5 in Glucose. There is no ground truth for these missing values.

Ten fold cross validation was repeated five times (seeds 42 to 46), giving 50 test folds per method. Because the structure search depends on the hash seed, the whole experiment was run under hash seeds 0 to 4 and each fold's score averaged over them.

### Study 3: Taiwan credit card data

30,000 customers, 23 predictors in the source data and a binary outcome (default, 22.1%). Ten predictors were kept (LIMIT_BAL, AGE, PAY_0, PAY_2, PAY_3, BILL_AMT1 to 3, PAY_AMT1 to 2), chosen before any results were seen to mix strongly and weakly related variables. SEX, EDUCATION and MARRIAGE were excluded because they are categories stored as numbers.

Missingness was injected in the same style as Study 1. For each of five replicate seeds (42 to 46), five of the ten predictors were drawn at random to be incomplete and the other five stayed fully observed. Each incomplete variable was given an R-parent drawn at random from the fully observed variables, cut into up to three ordered groups, with the missing probability rising across the groups in the ratio 1 : 1.5 : 2, scaled to the target rate at 10%, 20%, 30% and 40%. The outcome was never incomplete or an R-parent. Unlike Study 1, the R-parent need not be a parent of the incomplete variable, because real data have no known network. The three repayment status variables use fixed groups (no delay or paid early; one month late; two or more months late), because quantile cuts merged their tied values.

Hash seed 0 was used. A complete data ceiling is evaluated alongside, and the accuracy of the filled values is also measured (mean absolute error over the erased cells, divided by the column's standard deviation).

---

## Key results

KL divergence from the true network (mean over 20 replicates, 16 node evaluation):

| Method | 10% | 20% | 30% | 40% |
|---|---|---|---|---|
| Perfect baseline | 0.152 | 0.127 | 0.125 | 0.133 |
| MICE | 0.118 | 0.110 | 0.138 | 0.136 |
| EM | 0.407 | 0.914 | 1.537 | 2.053 |
| Soft-Impute | 0.500 | 1.012 | 1.671 | 2.230 |

EM and Soft-Impute learn about 6 to 8 extra indicator links per dataset on average. Across all levels, 387 of 400 indicator nodes under EM and 398 of 400 under Soft-Impute have an own link. The perfect baseline and MICE learn none.

Mean AUC, Pima (50 folds, hash averaged): median fill 0.804, EM 0.814, MICE 0.810, Soft-Impute 0.804.

Mean AUC, Taiwan: complete data 0.752. Differences between methods are at most 0.007 at any level.

The full tables, figures and supplementary material are in the paper and its supplement.

---

## Changes to the analysis during the work

The reported results are those after the following corrections, which are documented in the supplementary material:

1. **EM in Study 1.** The first version of the EM imputer recomputed the covariance from the filled table as if the filled values were observed, omitting the conditional covariance term of the M-step. Study 1 was rerun for EM with the full M-step. All other methods were unchanged. The old EM results are kept in `results/rq1/original_em_backup/`.
2. **Soft-Impute for new records (Studies 2 and 3).** The first version filled test records without applying the shrinkage. It was corrected before the reported results.
3. **Binning (Studies 2 and 3).** The first analyses computed the cut points after imputation, which made the binning depend on the imputer. The cut points are now computed from observed training values.
4. **Hash seed variability.** The Pima study was repeated under five hash seeds and averaged.

---

## Repository layout

The three studies live in their own folders. Check the paths below against your clone, since a few early scripts remain from the development history.

```
Imputation-Strategies-Study/
├── README.md
├── requirements.txt
├── src/
│   ├── experiments_rq1/            Study 1 (RQ1)
│   │   ├── generate_replicate.py         
│   │   ├── random_bn.py 
│   │   ├── em_impute.py, mice_impute.py, soft_impute.py   the three imputers
│   │   ├── run_em.py, run_mice.py, run_soft_impute.py     run each method on all replicates
│   │   ├── run_em_full.py, evaluate_em_full.py, merge_em_full.py   EM rerun with the full M-step
│   │   ├── evaluate_augmented.py, augmented_16.py   16 node evaluation
│   │   ├── evaluate.py             evaluation on the 11 original variables only (supplement)
│   │   ├── noise_floor.py          hash seed wobble
│   │   ├── check_mice_fills.py     accuracy of the filled cells
│   │   ├── run_mice_shallow.py     depth limited MICE check
│   │   ├── make_example_network.py figure of one replicate (Graphviz)
│   │   └── make_figures*.py        figures
|   |   └──soft_impute.py
|   |   └──run_soft_impute.py
│   ├── experiments_rq2/        Study 2 (Pima)
│   │   ├── load_pima.py, bn_classifier.py
│   │   ├── run_rq2_2.py            10 fold x 5 repeats, run once per hash seed
│   │   ├── hash_ave.py             average the five hash seeds
│   │   └── analyse_*.py, edge_counts.py, check_blanket.py, fig*.py
|   |   └──bn_classifier.py
|   |   └──em_impute.py
|   |   └──soft_impute.py
|   |   └──mice_impute.py
|   |   └──test_imputers.py
│   └── experiments_taiwan/     Study 3 (Taiwan)
│       ├── build_taiwan.py, inject_mar.py, bn_classifier.py
│       ├── run_taiwan.py           run with PYTHONHASHSEED = 0
│       └── analyse_taiwan*.py, f1.py, fig*.py
|       └──inject_mar.py
|       └──mice_impute.py
|       └──em_impute.py
|       └──soft_impute.py
├── data/
│   ├── synthetic/              replicates: complete data, incomplete data, .bif networks, audit logs
│   ├── pima/                   Pima Indians Diabetes data
│   └── taiwan/                 Taiwan credit card default data
├── results/
│   ├── rq1/                    Study 1 results (CSV)
│   ├── rq2/                    Study 2 results (CSV)
│   └── taiwan/                 Study 3 results (CSV)
└── figures/                    figures used in the paper
```

---

## How to reproduce

Commands are for Windows PowerShell. Set the hash seed in the same session before any run that learns a structure.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Study 1 (RQ1).** Run from `src/experiments`.

```powershell
# 1. generate the 20 replicates (seeds 0 to 19)
python f_replicate.py --seed 0       # repeat for each seed 0 to 19

# 2. impute (EM with the full M-step), then score
python run_em_full.py
python run_mice.py
python run_soft_impute.py
$env:PYTHONHASHSEED = "0"
python evaluate_augmented.py          # 16 node evaluation
python evaluate.py                    # 11 variable evaluation (supplement)

# 3. hash seed wobble (perfect baseline, 20% missing, hash seeds 0 to 9)
python noise_floor.py
```

**Study 2 (Pima).** Run from `src/experiments_rq2`, once for each hash seed 0 to 4, then average.

```powershell
foreach ($h in 0..4) { $env:PYTHONHASHSEED = "$h"; python run_rq2_2.py }
python hash_ave.py
```

**Study 3 (Taiwan).** Run from `src/experiments_taiwan`.

```powershell
python build_taiwan.py
$env:PYTHONHASHSEED = "0"
python run_taiwan.py
```

All randomness is seeded. Replicate networks, datasets, indicator choices and the full set of generation choices are logged and serialised, and the hash seed is fixed for the structure search.

---

## Software

Python with pgmpy 1.1.2 (network learning and inference), scikit-learn, SciPy, NumPy, pandas and Graphviz for network figures. See `requirements.txt`.

## Citation

If you use this code, please cite the paper (reference to be added once published).