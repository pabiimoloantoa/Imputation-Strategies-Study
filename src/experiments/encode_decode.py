"""
encode_decode.py

Shared one-hot encode/decode utility for the RQ1 imputation pipeline.

Rationale: the ground-truth BN's CPDs are generated via
model.get_random_cpds() (see random_bn.py) -- fully random probability
mass per state-combination, with no enforced ordering between neighboring
states. This makes every variable effectively NOMINAL at the CPT level,
regardless of whether its labels look ordinal (e.g. Income's
low/medium/high). One-hot encoding is used for ALL variables accordingly,
rather than ordinal encoding, to avoid imposing a false numeric ordering
that the data-generating process never actually built in.

Categories are fixed from VARIABLE_STATE_LABELS (random_bn.py), NOT
inferred from a given missing_Xpct_seedY.csv. This guarantees the one-hot
column set is identical across every seed and sparsity level for a given
variable, even if a rare state doesn't happen to appear in one particular
sample.

Missing cells: a NaN in the original named-states column becomes NaN
across ALL of that variable's one-hot columns (not 0), so imputation
methods can distinguish "needs to be filled" from "genuinely absent/zero".

Decoding: argmax per variable's one-hot column group -- deterministic,
consistent with single-imputation methods producing point estimates
rather than posterior draws. See project methodology notes for the
argmax-vs-proportional-sampling discussion.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from random_bn import VARIABLE_STATE_LABELS


def _onehot_column_name(var: str, state: str) -> str:
    return f"{var}__{state}"


def encode(df: pd.DataFrame, labels_dict: Dict[str, List[str]] = VARIABLE_STATE_LABELS) -> pd.DataFrame:
    """
    Named-states (possibly with NaN) -> one-hot expanded numeric dataframe.

    For each variable Xj with states [s0, s1, ...]:
      - adds columns Xj__s0, Xj__s1, ... (float, 0.0/1.0)
      - a row with Xj missing gets NaN across ALL of Xj's one-hot columns
      - a row with Xj = sk gets 1.0 in Xj__sk, 0.0 in Xj's other columns

    Column order is fixed by labels_dict, so the output is identical in
    shape/order regardless of which states happen to appear in df.
    """
    encoded_parts = []
    for var, states in labels_dict.items():
        if var not in df.columns:
            continue
        raw = df[var]
        if pd.api.types.is_float_dtype(raw):
            raw = raw.astype("Int64")
        col = raw.astype("string")
        for state in states:
            colname = _onehot_column_name(var, state)
            values = np.where(
                col.isna(), np.nan,
                np.where((col == state).fillna(False), 1.0, 0.0),
            )
            encoded_parts.append(pd.Series(values, index=df.index, name=colname))
            
    return pd.concat(encoded_parts, axis=1)


def decode(df_encoded: pd.DataFrame, labels_dict: Dict[str, List[str]] = VARIABLE_STATE_LABELS) -> pd.DataFrame:
    """
    One-hot expanded (completed, no NaNs expected post-imputation) ->
    named-states dataframe, one column per original variable.

    Decoding is via argmax: for each variable, the state whose one-hot
    column holds the highest reconstructed value is selected -- ties are
    broken by earliest state in labels_dict's order (via numpy argmax
    default behaviour).
    """
    decoded_parts = {}
    for var, states in labels_dict.items():
        cols = [_onehot_column_name(var, s) for s in states]
        if not all(c in df_encoded.columns for c in cols):
            continue
        sub = df_encoded[cols]
        best_idx = sub.to_numpy().argmax(axis=1)
        decoded_parts[var] = [states[i] for i in best_idx]

    return pd.DataFrame(decoded_parts, index=df_encoded.index)


if __name__ == "__main__":
    # Minimal smoke test using a couple of variables from VARIABLE_STATE_LABELS
    sample = pd.DataFrame({
        "Income": ["low", "medium", None, "high"],
       # "LoanAmount": ["high", None, "low", "medium"],
        "PaymentHistory": ["excellent", "poor", "fair", None],
    })

    enc = encode(sample)
    print("Encoded:\n", enc, "\n")

    # Fake "imputed" values: fill NaNs with plausible soft values, leave
    # observed one-hot rows untouched, to check decode round-trips observed
    # cells correctly.
    enc_filled = enc.fillna(0.0)
    # simulate a soft imputed guess for the two NaN cells
    enc_filled.loc[2, ["Income__low", "Income__medium", "Income__high"]] = [0.1, 0.7, 0.2]
   # enc_filled.loc[1, ["LoanAmount__low", "LoanAmount__medium", "LoanAmount__high"]] = [0.2, 0.3, 0.5]
    enc_filled.loc[3, ["PaymentHistory__poor", "PaymentHistory__fair", "PaymentHistory__good", "PaymentHistory__excellent"]] = [0.1, 0.1, 0.3, 0.5]


    dec = decode(enc_filled)
    print("Decoded:\n", dec)