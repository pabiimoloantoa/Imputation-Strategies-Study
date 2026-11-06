import pandas as pd
from sklearn.metrics import roc_auc_score

from load_pima import load_pima, get_folds, PROJECT_ROOT
from bn_classifier import Discretizer, BNClassifier

OLD = PROJECT_ROOT / "results" / "rq2_old_softimpute"
NEW = PROJECT_ROOT / "results" / "rq2"

first = pd.read_csv(OLD / "fold_scores.csv")
first = first[first.method == "Median"].set_index("fold")["auc"]
old_rep = pd.read_csv(OLD / "fold_scores_repeated.csv")
old_rep = old_rep[(old_rep.method == "Median") & (old_rep.seed == 42)].set_index("fold")["auc"]
new_rep = pd.read_csv(NEW / "fold_scores_repeated.csv")
new_rep = new_rep[(new_rep.method == "Median") & (new_rep.seed == 42)].set_index("fold")["auc"]

# today: the same Median pipeline, run now on the seed-42 folds
df = load_pima()
X, y = df.drop(columns="Outcome"), df["Outcome"]
today = {}
for k, (tr, te) in enumerate(get_folds(df, seed=42), 1):
    med = X.iloc[tr].median()
    tr_f, te_f = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)
    disc = Discretizer(3).fit(tr_f)
    D_tr, D_te = disc.transform(tr_f), disc.transform(te_f)
    D_tr["Outcome"] = y.iloc[tr].to_numpy()
    names = {c: list(range(n)) for c, n in disc.n_states_.items()}
    names["Outcome"] = [0, 1]
    p = BNClassifier().fit(D_tr, names).predict_proba(D_te)
    today[k] = roc_auc_score(y.iloc[te], p)

t = pd.DataFrame({"first_run": first, "old_repeated": old_rep,
                  "new_repeated": new_rep, "today": pd.Series(today)}).round(4)
print(t)
print("\nmean:", t.mean().round(4).to_dict())
print("\nlargest per-fold difference:")
print("  first_run vs old_repeated   :", round((t.first_run - t.old_repeated).abs().max(), 4))
print("  old_repeated vs new_repeated:", round((t.old_repeated - t.new_repeated).abs().max(), 4))
print("  new_repeated vs today       :", round((t.new_repeated - t.today).abs().max(), 4))