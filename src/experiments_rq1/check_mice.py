from pathlib import Path

import pandas as pd

from encode_decode import encode
from mice_impute import _normalize_state_dtypes
from random_bn import VARIABLE_STATE_LABELS

MISSING_DIR = Path("../../data/synthetic/missing")
VAR = "Default"


def main():
    files = sorted(MISSING_DIR.glob("missing_*pct_seed*.csv"))
    if not files:
        raise SystemExit(f"No missing_*pct_seed*.csv files found in {MISSING_DIR.resolve()}")

    at_risk = 0
    broken = []
    for f in files:
        df = pd.read_csv(f)
        if VAR not in df.columns:
            raise SystemExit(f"{f.name} has no {VAR} column.")
        n_blank = int(df[VAR].isna().sum())
        raw_dtype = str(df[VAR].dtype)

        norm = _normalize_state_dtypes(df, VARIABLE_STATE_LABELS)
        # mice_impute() encodes AFTER its initial fill, so no blanks reach encode().
        # Stand-in for that fill: use each variable's first state label.
        for v, states in VARIABLE_STATE_LABELS.items():
            if v in norm.columns:
                norm[v] = norm[v].fillna(states[0])
        X = encode(norm, VARIABLE_STATE_LABELS)
        cols = [c for c in X.columns if str(c).startswith(VAR)]
        if not cols:
            raise SystemExit(f"encode() produced no column starting with '{VAR}'. Columns: {list(X.columns)[:12]}...")
        rows_ok = int((X[cols].sum(axis=1) == 1).sum())
        rows_bad = len(X) - rows_ok

        if n_blank == 0:
            at_risk += 1
        if rows_bad > 0:
            broken.append((f.name, raw_dtype, str(norm[VAR].dtype), n_blank, rows_bad))

    print(f"Checked {len(files)} files. {at_risk} have no blanks in {VAR} (the at-risk case).")
    if not broken:
        print(f"RESULT: {VAR} is encoded correctly in every file. The dtype guard in your current mice_impute.py handles every file.")
    else:
        print(f"RESULT: {VAR} is NOT encoded correctly in {len(broken)} file(s):")
        print(f"{'file':<32}{'raw dtype':<12}{'after guard':<14}{'blanks':>7}{'bad rows':>10}")
        for name, raw, after, nb, bad in broken[:40]:
            print(f"{name:<32}{raw:<12}{after:<14}{nb:>7}{bad:>10}")
        if len(broken) > 40:
            print(f"... and {len(broken) - 40} more")


if __name__ == "__main__":
    main()






