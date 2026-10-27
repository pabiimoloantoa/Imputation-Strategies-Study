import pandas as pd

# change this to your actual file name
PATH = r"C:\Users\pabal\Downloads\bn-imputation-study-v2\data\pima\diabetes.csv"

df = pd.read_csv(PATH)
print(df.shape)
print(df.columns.tolist())

# columns where a zero is physiologically impossible
zero_means_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_means_missing:
    n = (df[col] == 0).sum()
    print(f"{col}: {n} zeros ({100 * n / len(df):.1f}%)")

print(df["Outcome"].value_counts())