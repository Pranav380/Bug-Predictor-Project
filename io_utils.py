from __future__ import annotations
import os
import json
import pandas as pd

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_json(obj, path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
