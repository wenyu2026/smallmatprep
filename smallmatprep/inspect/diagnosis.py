"""Small-sample dataset diagnosis for materials data.

Implements diagnostic heuristics based on Zhang & Ling (2018),
providing sample-size-aware recommendations for model complexity,
error sources, and the potential benefit of physics-prior (CEP) features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_diagnosis(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str | None = None,
    cep_available: bool = False,
) -> dict:
    """Diagnose a small-sample dataset and return recommendations.

    Evaluates sample size adequacy, degrees of freedom, expected error
    source (bias vs. variance), and whether CEP physics-prior features
    would be beneficial.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to diagnose.
    feature_cols : list of str, optional
        Names of feature columns. If ``None``, all numeric columns except
        *target_col* are used.
    target_col : str, optional
        Name of the target column (excluded from features).
    cep_available : bool, default=False
        Whether CEP-style physics-prior features are available for this
        dataset.

    Returns
    -------
    dict
        Keys include:
        - **n_samples** (*int*) — number of rows.
        - **n_features** (*int*) — number of feature columns.
        - **sample_size_category** (*str*) — ``"very_small"`` (< 30),
          ``"small"`` (30--100), ``"moderate"`` (100--500),
          ``"adequate"`` (>= 500).
        - **sample_to_feature_ratio** (*float*) — ``n_samples / n_features``.
        - **estimated_dof** (*int*) — approximate degrees of freedom
          (samples minus features).
        - **recommended_max_complexity** (*str*) — one of
          ``"simple_linear"``, ``"moderate_nonlinear"``,
          ``"flexible_ensemble"``.
        - **error_source_hint** (*str*) — ``"bias_dominated"``,
          ``"variance_dominated"``, or ``"balanced"`` per Zhang & Ling.
        - **cep_recommended** (*bool*) — whether CEP features are
          suggested.
        - **warnings** (*list of str*) — specific concerns.
        - **advice** (*str*) — human-readable summary recommendation.

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("Cannot diagnose an empty DataFrame")

    n_samples = len(df)

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        feature_cols = numeric_cols

    n_features = len(feature_cols)
    if n_features == 0:
        return {
            "n_samples": n_samples,
            "n_features": 0,
            "sample_size_category": _categorize_size(n_samples),
            "sample_to_feature_ratio": float("inf"),
            "estimated_dof": n_samples,
            "recommended_max_complexity": "simple_linear",
            "error_source_hint": "bias_dominated",
            "cep_recommended": False,
            "warnings": ["No feature columns found for diagnosis."],
            "advice": "Add feature columns to enable modeling.",
        }

    ratio = n_samples / n_features
    est_dof = max(n_samples - n_features, 1)
    size_cat = _categorize_size(n_samples)
    complexity = _recommend_complexity(n_samples, n_features, ratio)
    error_hint = _estimate_error_source(n_samples, n_features, ratio)
    cep_rec = _recommend_cep(n_samples, n_features, ratio, size_cat, cep_available)
    warnings = _generate_warnings(
        n_samples, n_features, ratio, size_cat, error_hint, cep_rec,
    )
    advice = _generate_advice(
        size_cat, complexity, error_hint, cep_rec, n_samples, n_features,
    )

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "sample_size_category": size_cat,
        "sample_to_feature_ratio": round(ratio, 2),
        "estimated_dof": est_dof,
        "recommended_max_complexity": complexity,
        "error_source_hint": error_hint,
        "cep_recommended": cep_rec,
        "warnings": warnings,
        "advice": advice,
    }


def _categorize_size(n: int) -> str:
    if n < 30:
        return "very_small"
    if n < 100:
        return "small"
    if n < 500:
        return "moderate"
    return "adequate"


def _recommend_complexity(
    n_samples: int, n_features: int, ratio: float,
) -> str:
    """Map sample size to recommended model complexity.

    Zhang & Ling (2018) show that with very few samples, even simple
    models can overfit. The heuristic here is conservative.
    """
    if n_samples < 30 or ratio < 2.0:
        return "simple_linear"
    if n_samples < 100 or ratio < 10.0:
        return "moderate_nonlinear"
    return "flexible_ensemble"


def _estimate_error_source(
    n_samples: int, n_features: int, ratio: float,
) -> str:
    """Estimate whether error is dominated by bias or variance.

    Zhang & Ling (2018) found that with small samples, bias² dominates
    over variance — contrary to the common belief that small-sample
    error is mainly variance. This function encodes that finding.
    """
    if ratio < 1.0:
        # Ultra high-dimensional: variance dominates
        return "variance_dominated"
    if n_samples < 50:
        # Very small samples: bias dominates (Zhang & Ling key finding)
        return "bias_dominated"
    if n_samples < 200 and ratio < 5.0:
        # Small samples with limited features: bias still likely dominates
        return "bias_dominated"
    if n_samples < 500:
        # Moderate: balanced
        return "balanced"
    return "balanced"


def _recommend_cep(
    n_samples: int, n_features: int, ratio: float,
    size_cat: str, cep_available: bool,
) -> bool:
    """Recommend CEP physics-prior features.

    CEP is most beneficial when bias dominates (small samples) or
    features are few (weak signal).
    """
    if not cep_available:
        return False
    if size_cat in ("very_small", "small"):
        return True
    if n_features < 5 and n_samples < 200:
        return True
    return False


def _generate_warnings(
    n_samples: int, n_features: int, ratio: float,
    size_cat: str, error_hint: str, cep_rec: bool,
) -> list[str]:
    warnings: list[str] = []
    if size_cat == "very_small":
        warnings.append(
            f"Very small sample ({n_samples} rows): risk of unreliable "
            "generalization. Results should be validated externally."
        )
    elif size_cat == "small":
        warnings.append(
            f"Small sample ({n_samples} rows): cross-validation metrics "
            "may have high variance."
        )

    if ratio < 2.0:
        warnings.append(
            f"Features ({n_features}) almost equal or exceed samples "
            f"({n_samples}): high risk of overfitting."
        )

    if error_hint == "bias_dominated":
        warnings.append(
            "Model error is likely bias-dominated: increasing model "
            "complexity or adding physics-prior features (CEP) may help."
        )
    elif error_hint == "variance_dominated":
        warnings.append(
            "Model error is likely variance-dominated: stronger "
            "regularization or more training data is recommended."
        )

    if error_hint == "bias_dominated" and not cep_rec:
        warnings.append(
            "CEP physics-prior features could improve bias-dominated "
            "error but are not currently available."
        )

    return warnings


def _generate_advice(
    size_cat: str, complexity: str, error_hint: str,
    cep_rec: bool, n_samples: int, n_features: int,
) -> str:
    lines: list[str] = []

    if size_cat in ("very_small", "small"):
        lines.append(
            f"Dataset has only {n_samples} samples with {n_features} "
            f"features. "
        )
        if cep_rec:
            lines.append(
                "Consider using CEP (chemistry-enhanced prediction) "
                "features to inject domain knowledge and reduce bias² error."
            )
        else:
            lines.append(
                "Consider adding physics-prior features (CEP) to improve "
                "prediction in the small-sample regime."
            )

    if complexity == "simple_linear":
        lines.append(
            "Start with simple linear models (Ridge, Lasso) before "
            "attempting nonlinear approaches."
        )
    elif complexity == "moderate_nonlinear":
        lines.append(
            "Moderately complex models (random forests with shallow "
            "trees, SVR) are appropriate."
        )

    if error_hint == "bias_dominated":
        lines.append(
            "Prediction error is likely bias-dominated: focus on feature "
            "engineering rather than hyperparameter tuning."
        )
    elif error_hint == "variance_dominated":
        lines.append(
            "Prediction error is likely variance-dominated: consider "
            "simplifying the model or collecting more data."
        )

    return " ".join(lines)


def missing_pattern_report(df: pd.DataFrame) -> dict:
    """Analyze co-occurrence of missing values.

    Reports overlap ratios between missing columns — which pairs tend
    to be missing together — and identifies any columns that are always
    missing simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Keys:
        - **pair_overlap** (*list of dict*) — each entry has ``col_a``,
          ``col_b``, ``both_missing``, ``overlap_ratio`` (fraction of
          time they are missing together).
        - **top_pairs** (*list of dict*) — top-5 highest-overlap pairs.
        - **perfect_overlap** (*list of tuple*) — column pairs that are
          always missing together.
    """
    if df.empty:
        return {"pair_overlap": [], "top_pairs": [], "perfect_overlap": []}

    mask = df.isna()
    missing_cols = [col for col in df.columns if mask[col].any()]
    if len(missing_cols) < 2:
        return {"pair_overlap": [], "top_pairs": [], "perfect_overlap": []}

    pairs = []
    perfect_pairs = []
    for i in range(len(missing_cols)):
        for j in range(i + 1, len(missing_cols)):
            a, b = missing_cols[i], missing_cols[j]
            both = (mask[a] & mask[b]).sum()
            overlap = both / max(mask[a].sum(), mask[b].sum())
            pairs.append({
                "col_a": a,
                "col_b": b,
                "both_missing": int(both),
                "overlap_ratio": round(float(overlap), 4),
            })
            if both > 0 and overlap >= 0.99:
                perfect_pairs.append((a, b))

    pairs.sort(key=lambda x: x["overlap_ratio"], reverse=True)
    return {
        "pair_overlap": pairs,
        "top_pairs": pairs[:5],
        "perfect_overlap": perfect_pairs,
    }
