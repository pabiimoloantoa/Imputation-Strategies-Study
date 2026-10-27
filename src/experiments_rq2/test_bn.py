from sklearn.metrics import roc_auc_score

from load_pima import load_pima, get_folds
from soft_impute import SoftImputer
from bn_classifier import Discretizer, BNClassifier

df = load_pima()
X = df.drop(columns="Outcome")
y = df["Outcome"]
tr, te = get_folds(df)[0]

# fill and bin (same as the last test)
imp = SoftImputer()
train_filled = imp.fit_transform(X.iloc[tr])
test_filled = imp.transform(X.iloc[te])
disc = Discretizer(n_bins=3).fit(train_filled)
D_tr = disc.transform(train_filled)
D_te = disc.transform(test_filled)
D_tr["Outcome"] = y.iloc[tr].to_numpy()      # Outcome goes back in for the network only

state_names = {c: list(range(n)) for c, n in disc.n_states_.items()}
state_names["Outcome"] = [0, 1]

# learn the network, then predict the test patients
bn = BNClassifier().fit(D_tr, state_names)

print("Arrows learned:")
for edge in sorted(bn.model_.edges()):
    print("  ", edge)

p = bn.predict_proba(D_te)
print("First 10 probabilities of diabetes:", p[:10].round(3))
print("AUC on this fold:", round(roc_auc_score(y.iloc[te], p), 3))