"""Missing-data diagnostics."""
import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing count and missing rate per column."""
    total = len(df)
    miss = df.isnull().sum()
    rate = (miss / total * 100).round(2)
    return pd.DataFrame({"missing_count": miss, "missing_rate_%": rate}).sort_values(
        "missing_rate_%", ascending=False
    )
