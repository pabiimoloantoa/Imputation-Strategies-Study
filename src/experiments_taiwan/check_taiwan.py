from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "taiwan"

files = sorted(DATA_DIR.glob("*.xls*"))
print("files found:", [f.name for f in files])
path = files[0]

df = pd.read_excel(path, header=1)          # real column names are on the second row
if "LIMIT_BAL" not in df.columns:           # in case this copy has them on the first row
    df = pd.read_excel(path, header=0)

print(df.shape)
print(df.columns.tolist())
print("missing values:", int(df.isna().sum().sum()))

target = df.columns[-1]
print(df[target].value_counts())
print("default rate:", round(df[target].mean(), 3))

for col in ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_1"]:
    if col in df.columns:
        print(df[col].value_counts().sort_index())