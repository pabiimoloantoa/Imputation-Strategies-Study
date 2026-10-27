from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[2] / "results" / "rq1" / "augmented_results.csv"
d = pd.read_csv(path)
print(d.drop(columns=["learned_edges", "r_extra_pairs"]).to_string(index=False))