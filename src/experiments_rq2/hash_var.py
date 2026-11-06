import os
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] == "child":
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from load_pima import load_pima, get_folds
    from bn_classifier import Discretizer, BNClassifier

    df = load_pima()
    X, y = df.drop(columns="Outcome"), df["Outcome"]
    aucs = []
    for tr, te in get_folds(df, seed=42):
        med = X.iloc[tr].median()
        tr_f, te_f = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)
        disc = Discretizer(3).fit(tr_f)
        D_tr, D_te = disc.transform(tr_f), disc.transform(te_f)
        D_tr["Outcome"] = y.iloc[tr].to_numpy()
        names = {c: list(range(k)) for c, k in disc.n_states_.items()}
        names["Outcome"] = [0, 1]
        p = BNClassifier().fit(D_tr, names).predict_proba(D_te)
        aucs.append(roc_auc_score(y.iloc[te], p))
    print(round(float(np.mean(aucs)), 4))
else:
    for hs in ["0", "1", "2", "3", "4"]:
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, __file__, "child"], env=env,
                             capture_output=True, text=True)
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
        print(f"PYTHONHASHSEED={hs}: mean AUC over 10 folds = {last}")