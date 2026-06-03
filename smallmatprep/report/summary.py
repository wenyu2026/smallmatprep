"""Compact report helpers."""
import pandas as pd


def build_summary(
    missing_df: pd.DataFrame,
    evaluation: dict,
    model_name: str,
) -> str:
    """Return a formatted plain-text summary string.

    Parameters
    ----------
    missing_df : pd.DataFrame
        Output of :func:`smallmatprep.inspect.missingness.missing_report`.
    evaluation : dict
        Output of :func:`smallmatprep.evaluate.metrics.evaluate_model`.
    model_name : str
        Name of the model used.

    Returns
    -------
    str
        Human-readable summary.
    """
    def _fmt(key: str) -> str:
        val = evaluation.get(key, "NA")
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    lines = [
        "== SmallMatPrep summary ==",
        f"Model: {model_name}",
        f"MAE:  {_fmt('MAE')}",
        f"RMSE: {_fmt('RMSE')}",
        f"R²:   {_fmt('R2')}",
        "",
        "Top missing columns:",
        missing_df.head(5).to_string(),
    ]
    return "\n".join(lines)
