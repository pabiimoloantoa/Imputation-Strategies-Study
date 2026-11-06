from augmented_16 import load_trimmed_truth, build_learning_table

truth = load_trimmed_truth(2, 20)
em = build_learning_table(2, 20, "em", truth)
perfect = build_learning_table(2, 20, "perfect", truth)

blank = em["R_Income_20pct"] == "missing"
print("True Income values in the blanked rows:")
print(perfect.loc[blank, "Income"].value_counts(normalize=True).round(2))
print("EM's fills in those same rows:")
print(em.loc[blank, "Income"].value_counts(normalize=True).round(2))


