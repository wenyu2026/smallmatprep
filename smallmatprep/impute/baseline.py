"""Baseline imputation methods."""
import pandas as pd
import numpy as np


def impute_median(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Impute missing values with column medians."""
    out = df.copy()
    for col in feature_cols:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def impute_knn(df: pd.DataFrame, feature_cols: list[str], n_neighbors: int = 3) -> pd.DataFrame:
    """Simple KNN imputation using sklearn KNNImputer."""
    from sklearn.impute import KNNImputer

    out = df.copy()
    cols = [c for c in feature_cols if c in out.columns]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    out[cols] = imputer.fit_transform(out[cols])
    return out
