"""Compact report helpers."""
import pandas as pd


def build_summary(missing_df: pd.DataFrame, evaluation: dict, model_name: str) -> str:
    """Return a plain-text summary string."""
    lines = [
        "== SmallMatPrep summary ==",
        f"Model: {model_name}",
        f"MAE: {evaluation.get('MAE', 'NA')}",
        f"RMSE: {evaluation.get('RMSE', 'NA')}",
        f"R2: {evaluation.get('R2', 'NA')}",
        "",
        "Top missing columns:",
        missing_df.head(5).to_string(),
    ]
    return "\n".join(lines)
