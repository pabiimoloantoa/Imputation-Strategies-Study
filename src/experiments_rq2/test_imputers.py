from load_pima import load_pima, get_folds
from soft_impute import SoftImputer

df = load_pima()
X = df.drop(columns="Outcome")        # the 8 measurements, no Outcome
tr, te = get_folds(df)[0]             # fold 1 only

imp = SoftImputer()
train_filled = imp.fit_transform(X.iloc[tr])
test_filled = imp.transform(X.iloc[te])

print("blanks left in train:", train_filled.isna().sum().sum())
print("blanks left in test:", test_filled.isna().sum().sum())
print(train_filled.describe().round(1))