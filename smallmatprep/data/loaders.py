"""Data loading utilities."""
from pathlib import Path
import json
import pandas as pd


def load_csv(path: str | Path, id_column: str | None = None) -> pd.DataFrame:
    """Load a CSV file and optionally set the index column."""
    df = pd.read_csv(path)
    if id_column and id_column in df.columns:
        df = df.set_index(id_column)
    return df


def load_config(path: str | Path) -> dict:
    """Load a JSON config file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
