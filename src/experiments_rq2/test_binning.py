from load_pima import load_pima, get_folds
from em_impute import EMImputer
from bn_classifier import Discretizer

df = load_pima()
X = df.drop(columns="Outcome")
tr, te = get_folds(df)[0]

imp = EMImputer()
train_filled = imp.fit_transform(X.iloc[tr])
test_filled = imp.transform(X.iloc[te])

disc = Discretizer(n_bins=3).fit(train_filled)   # cut points come from training only
D_tr = disc.transform(train_filled)
D_te = disc.transform(test_filled)

print(disc.n_states_)
print(D_tr.apply(lambda c: c.value_counts().sort_index()).T)   # group sizes, training
print(D_te.apply(lambda c: c.value_counts().sort_index()).T)   # group sizes, test