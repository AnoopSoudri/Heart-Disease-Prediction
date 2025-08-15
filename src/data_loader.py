import pandas as pd
from pathlib import Path

def load_raw(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)
