"""Data loading utilities."""
from pathlib import Path
import json

import pandas as pd


def load_csv(path: str | Path, id_column: str | None = None) -> pd.DataFrame:
    """Load a CSV file and optionally set the index column.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    id_column : str, optional
        Column name to use as the DataFrame index.

    Returns
    -------
    pd.DataFrame
        Loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CSV is empty or *id_column* is not found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    if id_column:
        if id_column not in df.columns:
            raise ValueError(
                f"id_column '{id_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.set_index(id_column)

    return df


def load_config(path: str | Path) -> dict:
    """Load a JSON config file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON config file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
