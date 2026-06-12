"""Material-aware imputation (group-based median with domain context)."""
import pandas as pd

from .baseline import impute_median


def impute_group_median(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
) -> pd.DataFrame:
    """Fill missing values using group-level medians, falling back to global median.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    feature_cols : list[str]
        Columns to impute.
    group_col : str
        Column to group by (e.g. material family, solvent type).

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.

    Raises
    ------
    ValueError
        If *group_col* is not in the DataFrame columns.
    """
    out = df.copy()

    if group_col not in out.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in DataFrame. "
            f"Available columns: {list(out.columns)}"
        )

    for col in feature_cols:
        if col not in out.columns:
            continue
        # Group-level median fill, then global median as safety net
        group_medians = out.groupby(group_col)[col].transform("median")
        out[col] = out[col].fillna(group_medians)
        out[col] = out[col].fillna(out[col].median())

    return out
