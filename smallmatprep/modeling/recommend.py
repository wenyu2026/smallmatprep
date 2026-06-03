"""Model recommendation heuristics for small-sample data."""


def recommend_model(n_samples: int, n_features: int) -> list[dict]:
    """Return a ranked list of recommended models and reasons.

    Parameters
    ----------
    n_samples : int
        Number of training samples. Must be > 0.
    n_features : int
        Number of feature columns. Must be >= 0.

    Returns
    -------
    list[dict]
        Each dict has keys: "model", "params", "reason".

    Raises
    ------
    ValueError
        If n_samples <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_features < 0:
        raise ValueError(f"n_features must be >= 0, got {n_features}")

    recs: list[dict] = [
        {
            "model": "RandomForestRegressor",
            "params": {"n_estimators": 200, "random_state": 0},
            "reason": "Stable nonlinear baseline for small-sample tabular data.",
        }
    ]

    # Ridge is best when features outnumber samples (regularization prevents overfitting)
    if n_features >= n_samples:
        recs.append({
            "model": "Ridge",
            "params": {},
            "reason": (
                f"Regularized linear model for high-dimensional data "
                f"({n_features} features ≥ {n_samples} samples)."
            ),
        })
    else:
        recs.append({
            "model": "Ridge",
            "params": {},
            "reason": "Simple linear baseline for comparison.",
        })

    recs.append({
        "model": "SVR",
        "params": {"kernel": "rbf"},
        "reason": "Flexible nonlinear model; useful for sensitivity comparison.",
    })
    return recs
