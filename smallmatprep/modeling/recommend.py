"""Model recommendation heuristics for small-sample data.

Supports both regression and classification, with awareness of:
- Sample size vs. feature count (degrees of freedom)
- CEP physics-prior feature availability
- Zhang & Ling (2018) bias-dominated regime insights
"""
from __future__ import annotations


def recommend_model(
    n_samples: int,
    n_features: int,
    task: str = "regression",
    cep_available: bool = False,
) -> list[dict]:
    """Return a ranked list of recommended models with parameters and reasoning.

    The recommendations are adjusted based on sample size, feature count,
    task type, and whether CEP physics-prior features are available.

    Parameters
    ----------
    n_samples : int
        Number of training samples. Must be > 0.
    n_features : int
        Number of feature columns. Must be >= 0.
    task : str, default="regression"
        Task type: ``"regression"`` or ``"classification"``.
    cep_available : bool, default=False
        Whether CEP physics-prior features are available. When ``True``,
        simpler models may be recommended because CEP features already
        encode domain knowledge.

    Returns
    -------
    list[dict]
        Each dict has keys: ``"model"``, ``"params"``, ``"reason"``.

    Raises
    ------
    ValueError
        If *n_samples* <= 0, or if *task* is not supported.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_features < 0:
        raise ValueError(f"n_features must be >= 0, got {n_features}")
    if task not in ("regression", "classification"):
        raise ValueError(f"Unsupported task '{task}'. Use 'regression' or 'classification'.")

    ratio = n_samples / max(n_features, 1)
    small_data = n_samples < 100 or ratio < 5.0
    high_dim = n_features >= n_samples

    if task == "classification":
        return _recommend_classification(n_samples, n_features, ratio, small_data, high_dim, cep_available)
    return _recommend_regression(n_samples, n_features, ratio, small_data, high_dim, cep_available)


def _recommend_regression(
    n_samples: int, n_features: int, ratio: float,
    small_data: bool, high_dim: bool, cep_available: bool,
) -> list[dict]:
    recs: list[dict] = []

    # --- Baseline: Ridge / Linear ---
    if high_dim:
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

    # --- Primary nonlinear model ---
    if cep_available and small_data:
        # CEP provides domain prior: simpler model is sufficient
        recs.append({
            "model": "RandomForestRegressor",
            "params": {
                "n_estimators": 200,
                "max_depth": 3,
                "min_samples_leaf": 5,
                "random_state": 0,
            },
            "reason": (
                "Shallow random forest with CEP features. CEP physics priors "
                "reduce bias, so simple nonlinearity suffices."
            ),
        })
    elif small_data:
        recs.append({
            "model": "RandomForestRegressor",
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_leaf": 3,
                "random_state": 0,
            },
            "reason": "Stable nonlinear baseline for small-sample tabular data.",
        })
    else:
        recs.append({
            "model": "RandomForestRegressor",
            "params": {"n_estimators": 300, "random_state": 0},
            "reason": "Flexible ensemble for moderate-sized datasets.",
        })

    # --- Third model: SVR or simple linear (depending on data size) ---
    if n_samples < 50:
        recs.append({
            "model": "Ridge",
            "params": {},
            "reason": "Second opinion: linear model (very small sample, avoid complexity).",
        })
    else:
        recs.append({
            "model": "SVR",
            "params": {"kernel": "rbf", "C": 10.0 if small_data else 1.0},
            "reason": "Flexible nonlinear model for sensitivity comparison.",
        })

    return recs


def _recommend_classification(
    n_samples: int, n_features: int, ratio: float,
    small_data: bool, high_dim: bool, cep_available: bool,
) -> list[dict]:
    recs: list[dict] = []

    # --- Baseline: LogisticRegression ---
    if high_dim:
        recs.append({
            "model": "LogisticRegression",
            "params": {"penalty": "l1", "solver": "saga", "max_iter": 1000, "random_state": 0},
            "reason": (
                "L1-regularized logistic regression for high-dimensional "
                f"data ({n_features} features ≥ {n_samples} samples)."
            ),
        })
    else:
        recs.append({
            "model": "LogisticRegression",
            "params": {"max_iter": 1000, "random_state": 0},
            "reason": "Simple linear classifier baseline.",
        })

    # --- Primary nonlinear classifier ---
    if cep_available and small_data:
        recs.append({
            "model": "RandomForestClassifier",
            "params": {
                "n_estimators": 200,
                "max_depth": 3,
                "min_samples_leaf": 5,
                "random_state": 0,
            },
            "reason": (
                "Shallow random forest with CEP features. Domain priors "
                "reduce bias, shallow trees prevent overfitting."
            ),
        })
    elif small_data:
        recs.append({
            "model": "RandomForestClassifier",
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_leaf": 3,
                "random_state": 0,
            },
            "reason": "Stable nonlinear classifier for small-sample data.",
        })
    else:
        recs.append({
            "model": "RandomForestClassifier",
            "params": {"n_estimators": 300, "random_state": 0},
            "reason": "Flexible ensemble classifier.",
        })

    # --- Third model ---
    recs.append({
        "model": "SVC",
        "params": {"kernel": "rbf", "C": 10.0 if small_data else 1.0, "probability": True, "random_state": 0},
        "reason": "Nonlinear classifier with probabilistic output; useful for comparison.",
    })

    return recs
