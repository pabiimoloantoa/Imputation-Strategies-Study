from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "taiwan"

PREDICTORS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "PAY_AMT1", "PAY_AMT2",
]
TARGET_RAW = "default payment next month"

raw = sorted(DATA_DIR.glob("*.xls*"))[0]
df = pd.read_excel(raw, header=1)

clean = df[PREDICTORS + [TARGET_RAW]].rename(columns={TARGET_RAW: "Outcome"})
clean.to_csv(DATA_DIR / "taiwan_clean.csv", index=False)

print("saved:", DATA_DIR / "taiwan_clean.csv")
print(clean.shape)
print("missing values:", int(clean.isna().sum().sum()))
print("default rate:", round(clean["Outcome"].mean(), 3))
print(clean[PREDICTORS].corr().round(2))