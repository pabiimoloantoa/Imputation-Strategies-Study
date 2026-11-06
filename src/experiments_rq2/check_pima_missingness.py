from scipy.stats import fisher_exact

from load_pima import load_pima

df = load_pima()
print("overall diabetes rate:", round(df["Outcome"].mean(), 3), "\n")
for col in ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"]:
    miss = df[col].isna()
    a, b = df.loc[miss, "Outcome"], df.loc[~miss, "Outcome"]
    _, p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    print(f"{col:<14s} missing n={miss.sum():<4d} diabetes rate when missing {a.mean():.3f}  "
          f"when observed {b.mean():.3f}  (p={p:.3f})")