from pathlib import Path

import pandas as pd

d = Path(__file__).resolve().parents[2] / "results" / "taiwan"
for f in sorted(d.glob("*_seed*.csv")):
    print(f.name, len(pd.read_csv(f)))