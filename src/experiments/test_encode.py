import pandas as pd
from encode_decode import encode, decode

df = pd.read_csv("../../data/synthetic/missing/missing_10pct_seed0.csv")

print(df.shape)
print(df.isna().sum())

encoded = encode(df)
print(encoded.shape)

decoded = decode(encoded.fillna(0))
for col in df.columns:
    mask = df[col].notna()
    original = df.loc[mask, col]
    if pd.api.types.is_float_dtype(original):
        original = original.astype("Int64").astype("string")
    matches = (decoded.loc[mask, col] == original).all()
    print(col, matches)