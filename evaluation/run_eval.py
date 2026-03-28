from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_json(ROOT / "data" / "eval_set.json")
print(df.head(5))
