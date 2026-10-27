from load_pima import load_pima, get_folds
from soft_impute import SoftImputer

df = load_pima()
X = df.drop(columns="Outcome")
tr, te = get_folds(df)[0]
imp = SoftImputer().fit(X.iloc[tr])
print("directions kept:", imp.Vt_.shape[0], "of", X.shape[1], "| shrinkage:", round(imp.lambda_, 2))