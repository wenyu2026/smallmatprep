"""Bias-variance decomposition via bootstrap for small-sample evaluation.

Implements the standard bootstrap-based decomposition of mean squared error
into bias² + variance components (Geman, Bienenstock & Doursat, 1992).

This is particularly important for small-sample materials datasets because
Zhang & Ling (2018) show that bias² often dominates variance in this
regime — meaning feature engineering (e.g., CEP physics priors) matters
more than tuning model complexity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bias_variance_decomposition(
    model_class: type,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model_params: dict | None = None,
    n_bootstraps: int = 50,
    random_state: int = 42,
) -> dict:
    """Decompose MSE into bias² + variance using bootstrap resampling.

    Parameters
    ----------
    model_class : type
        An sklearn-compatible regressor class (e.g.,
        ``RandomForestRegressor``, ``Ridge``). Must have a ``fit`` and
        ``predict`` method.
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray or pd.Series
        Target vector of shape ``(n_samples,)``.
    model_params : dict, optional
        Keyword arguments passed to ``model_class(**model_params)``.
    n_bootstraps : int, default=50
        Number of bootstrap rounds. Higher gives more stable estimates.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
        - **bias_squared** (*float*) — average bias² across all samples.
        - **variance** (*float*) — average variance across all samples.
        - **total_error** (*float*) — ``bias_squared + variance``.
        - **mse** (*float*) — direct MSE on OOB predictions (should be
          close to ``total_error``).
        - **fraction_bias** (*float*) — ``bias_squared / total_error``.
        - **fraction_variance** (*float*) — ``variance / total_error``.
        - **dominant_source** (*str*) — ``"bias"``, ``"variance"``, or
          ``"balanced"``.
        - **n_bootstraps_used** (*int*) — number of successful bootstrap
          rounds.
        - **n_samples** (*int*) — number of samples.
        - **n_features** (*int*) — number of features.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n_samples, n_features = X_arr.shape
    rng = np.random.default_rng(random_state)

    if model_params is None:
        model_params = {}

    # Store OOB predictions: list of (sample_index, prediction) pairs
    oob_preds: list[list[float]] = [[] for _ in range(n_samples)]
    successful = 0

    for b in range(n_bootstraps):
        seed = random_state + b * 7
        bag = rng.integers(0, n_samples, size=n_samples)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bag] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) == 0:
            continue

        X_boot = X_arr[bag]
        y_boot = y_arr[bag]

        try:
            model = model_class(**model_params)
            model.fit(X_boot, y_boot)
            preds = model.predict(X_arr[oob_idx])
        except Exception:
            continue

        for i, idx in enumerate(oob_idx):
            oob_preds[idx].append(float(preds[i]))
        successful += 1

    # Compute bias² and variance per sample
    biases: list[float] = []
    vars_: list[float] = []
    total_biases: list[float] = []

    for i in range(n_samples):
        preds_i = oob_preds[i]
        if len(preds_i) < 2:
            continue

        expected = np.mean(preds_i)
        bias_i = (expected - float(y_arr[i])) ** 2
        var_i = np.var(preds_i, ddof=1)
        biases.append(bias_i)
        vars_.append(var_i)
        total_biases.append(bias_i + var_i)

    if not biases:
        return {
            "bias_squared": float("nan"),
            "variance": float("nan"),
            "total_error": float("nan"),
            "mse": float("nan"),
            "fraction_bias": float("nan"),
            "fraction_variance": float("nan"),
            "dominant_source": "unknown",
            "n_bootstraps_used": successful,
            "n_samples": n_samples,
            "n_features": n_features,
        }

    bias_sq = float(np.mean(biases))
    variance = float(np.mean(vars_))
    total_err = bias_sq + variance

    frac_bias = bias_sq / total_err if total_err > 0 else 0.5
    frac_var = variance / total_err if total_err > 0 else 0.5

    if frac_bias > 0.55:
        dominant = "bias"
    elif frac_var > 0.55:
        dominant = "variance"
    else:
        dominant = "balanced"

    return {
        "bias_squared": bias_sq,
        "variance": variance,
        "total_error": total_err,
        "mse": total_err,
        "fraction_bias": round(frac_bias, 4),
        "fraction_variance": round(frac_var, 4),
        "dominant_source": dominant,
        "n_bootstraps_used": successful,
        "n_samples": n_samples,
        "n_features": n_features,
    }


def decomposition_summary(decomp: dict) -> str:
    """Return a human-readable summary string of the decomposition results.

    Parameters
    ----------
    decomp : dict
        Output of :func:`bias_variance_decomposition`.

    Returns
    -------
    str
        Formatted summary.
    """
    if decomp.get("dominant_source") == "unknown":
        return "Bias-variance decomposition failed (insufficient OOB samples)."

    lines = [
        "== Bias-Variance Decomposition ==",
        f"Samples: {decomp['n_samples']} | Features: {decomp['n_features']}",
        f"Bootstraps used: {decomp['n_bootstraps_used']}",
        "",
        f"MSE = bias² + variance",
        f"  bias²     = {decomp['bias_squared']:.4f}  ({decomp['fraction_bias']:.1%})",
        f"  variance  = {decomp['variance']:.4f}  ({decomp['fraction_variance']:.1%})",
        f"  total     = {decomp['total_error']:.4f}",
        "",
    ]

    source = decomp["dominant_source"]
    if source == "bias":
        lines.append(
            "→ Error is BIAS-dominated: invest in feature engineering "
            "(CEP), not model tuning."
        )
    elif source == "variance":
        lines.append(
            "→ Error is VARIANCE-dominated: simplify the model, "
            "add regularization, or collect more data."
        )
    else:
        lines.append(
            "→ Error is balanced: both feature engineering and "
            "regularization may help."
        )

    return "\n".join(lines)
