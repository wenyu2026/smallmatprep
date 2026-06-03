"""Baseline imputation methods."""
import pandas as pd
import numpy as np


def impute_median(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Impute missing values with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    feature_cols : list[str]
        Columns to impute. Columns not in *df* are silently skipped.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with imputed values.

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("Cannot impute an empty DataFrame")
    out = df.copy()
    for col in feature_cols:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def impute_knn(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_neighbors: int = 3,
) -> pd.DataFrame:
    """KNN imputation using sklearn KNNImputer.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    feature_cols : list[str]
        Columns to impute.
    n_neighbors : int
        Number of neighbors. Must be <= number of rows.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with imputed values.

    Raises
    ------
    ValueError
        If *n_neighbors* exceeds the number of available samples.
    """
    from sklearn.impute import KNNImputer

    out = df.copy()
    cols = [c for c in feature_cols if c in out.columns]

    if not cols:
        return out

    if n_neighbors > len(out):
        raise ValueError(
            f"n_neighbors ({n_neighbors}) cannot exceed number of samples "
            f"({len(out)}). Reduce n_neighbors or collect more data."
        )

    imputer = KNNImputer(n_neighbors=n_neighbors)
    out[cols] = imputer.fit_transform(out[cols])
    return out
