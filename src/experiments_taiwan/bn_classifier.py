import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=r"(?s).*deprecated.*")
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator

try:
    from pgmpy.estimators import BIC as BicScore
except ImportError:
    from pgmpy.estimators import BicScore

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork


class Discretizer:
    """Cuts each numeric column into groups using the TRAINING data only.
    Columns listed in fixed_edges use those cut points instead of quantiles."""

    def __init__(self, n_bins=3, fixed_edges=None):
        self.n_bins = n_bins
        self.fixed_edges = fixed_edges or {}

    def fit(self, df):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = {}
        for c in df.columns:
            if c in self.fixed_edges:
                self.edges_[c] = np.array(self.fixed_edges[c], dtype=float)
            else:
                self.edges_[c] = np.unique(np.quantile(df[c], qs))
        self.n_states_ = {c: len(e) + 1 for c, e in self.edges_.items()}
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c, e in self.edges_.items():
            out[c] = np.searchsorted(e, df[c].to_numpy(), side="right")
        return out


class BNClassifier:
    """Learns a Bayesian Network (structure + probability tables) and predicts Outcome."""

    def __init__(self, target="Outcome", ess=10, max_indegree=3):
        self.target = target
        self.ess = ess                  # smoothing strength for the probability tables
        self.max_indegree = max_indegree  # max parents per variable

    def fit(self, data, state_names):
        # 1) structure: Hill Climbing with the BIC score
        dag = HillClimbSearch(data).estimate(
            scoring_method=BicScore(data),
            max_indegree=self.max_indegree,
            show_progress=False,
        )
        self.model_ = BayesianNetwork(dag.edges())
        self.model_.add_nodes_from(data.columns)   # keep variables with no arrows
        # 2) parameters: probability tables, smoothed so no cell is exactly zero
        est = BayesianEstimator(self.model_, data, state_names=state_names)
        cpds = est.get_parameters(prior_type="BDeu", equivalent_sample_size=self.ess)
        self.model_.add_cpds(*cpds)
        return self

    def predict_proba(self, X):
        """Probability that Outcome = 1 for each row of X (X has no Outcome column)."""
        probs = self.model_.predict_probability(X)
        return probs[f"{self.target}_1"].to_numpy()