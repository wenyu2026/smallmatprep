"""Model recommendation heuristics for small-sample data."""
import pandas as pd


def recommend_model(n_samples: int, n_features: int) -> list[dict]:
    """Return a ranked list of recommended models and reasons."""
    recs = []
    recs.append({
        "model": "RandomForestRegressor",
        "params": {"n_estimators": 200, "random_state": 0},
        "reason": "Stable nonlinear baseline for small-sample tabular data."
    })
    if n_features <= n_samples:
        recs.append({
            "model": "Ridge",
            "params": {},
            "reason": "Low-variance linear baseline when feature count is moderate."
        })
    recs.append({
        "model": "SVR",
        "params": {"kernel": "rbf"},
        "reason": "Flexible nonlinear model; useful for sensitivity comparison."
    })
    return recs
