"""Missing-data diagnostics."""
import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing count and rate per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Columns: "missing_count", "missing_rate_%". Sorted by rate descending.

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("Cannot generate missing report for an empty DataFrame")

    total = len(df)
    miss = df.isnull().sum()
    rate = (miss / total * 100).round(2)
    return pd.DataFrame({
        "missing_count": miss,
        "missing_rate_%": rate,
    }).sort_values("missing_rate_%", ascending=False)
