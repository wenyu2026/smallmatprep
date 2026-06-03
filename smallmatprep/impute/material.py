"""Material-aware imputation placeholders."""
import pandas as pd


def impute_group_median(df: pd.DataFrame, feature_cols: list[str], group_col: str) -> pd.DataFrame:
    """Impute with group-level medians, then global median fallback."""
    out = df.copy()
    if group_col not in out.columns:
        return impute_median(out, feature_cols)
    for col in feature_cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(group_col)[col].transform(lambda s: s.fillna(s.median()))
        out[col] = out[col].fillna(out[col].median())
    return out


def impute_median(df, feature_cols):
    from .baseline import impute_median as _base
    return _base(df, feature_cols)
