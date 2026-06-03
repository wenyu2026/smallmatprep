"""Model evaluation utilities."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and R² as a dict of floats.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values. Must be the same length as *y_true*.

    Returns
    -------
    dict
        Keys: "MAE", "RMSE", "R2".

    Raises
    ------
    ValueError
        If inputs have different lengths or contain NaN/Inf.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or Inf values")
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
