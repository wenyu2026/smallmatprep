"""Configuration schema and validation."""
from typing import Any

REQUIRED_KEYS = ["target"]
VALID_IMPUTE_METHODS = {"median", "knn", "zero", "group_median", "mean", "none"}
VALID_MODELS = {
    "RandomForestRegressor", "RandomForestClassifier",
    "Ridge", "SVR", "XGBoost",
}
ALLOWED_KEYS = {
    "target", "id_column", "feature_groups", "blank_as_zero_columns",
    "categorical_columns", "drop_columns", "impute", "model", "task",
    "test_size", "random_state", "prefer_interpretable",
}


def validate_config(cfg: dict[str, Any], columns: list[str] | None = None) -> list[str]:
    """Validate a SmallMatPrep config and return a list of issues.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary to validate.
    columns : list[str], optional
        If provided, validate that referenced column names actually exist.

    Returns
    -------
    list[str]
        List of issues found. Empty list means the config is valid.
    """
    issues: list[str] = []

    # -- 1. Required fields --
    for k in REQUIRED_KEYS:
        if k not in cfg:
            issues.append(f"Missing required key: '{k}'")

    # -- 2. Unknown fields --
    for k in cfg:
        if k not in ALLOWED_KEYS:
            issues.append(
                f"Unknown key: '{k}'. Allowed keys: {sorted(ALLOWED_KEYS)}"
            )

    # -- 3. target --
    target = cfg.get("target")
    if target is not None and not isinstance(target, str):
        issues.append(f"'target' must be a string, got {type(target).__name__}")

    # -- 4. id_column --
    id_col = cfg.get("id_column")
    if id_col is not None and not isinstance(id_col, str):
        issues.append(f"'id_column' must be a string, got {type(id_col).__name__}")

    # -- 5. feature_groups --
    fg = cfg.get("feature_groups")
    if fg is not None:
        if not isinstance(fg, dict):
            issues.append(f"'feature_groups' must be a dict, got {type(fg).__name__}")
        else:
            for group_name, group_cols in fg.items():
                if not isinstance(group_cols, list):
                    issues.append(
                        f"feature_groups['{group_name}'] must be a list"
                    )
                elif columns:
                    for col in group_cols:
                        if col not in columns:
                            issues.append(
                                f"feature_groups['{group_name}'] references "
                                f"unknown column '{col}'"
                            )

    # -- 6. List-type fields --
    for field in ("blank_as_zero_columns", "categorical_columns", "drop_columns"):
        val = cfg.get(field)
        if val is not None:
            if not isinstance(val, list):
                issues.append(f"'{field}' must be a list, got {type(val).__name__}")
            elif columns:
                for col in val:
                    if col not in columns:
                        issues.append(f"'{field}' references unknown column '{col}'")

    # -- 7. impute --
    impute = cfg.get("impute")
    if impute is not None:
        if not isinstance(impute, dict):
            issues.append(f"'impute' must be a dict, got {type(impute).__name__}")
        else:
            method = impute.get("method")
            if method is not None and method not in VALID_IMPUTE_METHODS:
                issues.append(
                    f"impute.method must be one of {sorted(VALID_IMPUTE_METHODS)}, "
                    f"got '{method}'"
                )
            imp_cols = impute.get("columns")
            if imp_cols is not None:
                if isinstance(imp_cols, list):
                    if columns:
                        for col in imp_cols:
                            if col not in columns:
                                issues.append(
                                    f"impute.columns references unknown column '{col}'"
                                )
                elif isinstance(imp_cols, dict):
                    if columns:
                        for col in imp_cols:
                            if col not in columns:
                                issues.append(
                                    f"impute.columns references unknown column '{col}'"
                                )
                else:
                    issues.append(
                        "impute.columns must be a list or dict, "
                        f"got {type(imp_cols).__name__}"
                    )

    # -- 8. model --
    model = cfg.get("model")
    if model is not None:
        if isinstance(model, str) and model not in VALID_MODELS:
            issues.append(
                f"Unknown model '{model}'. Known models: {sorted(VALID_MODELS)}"
            )
        elif isinstance(model, dict):
            model_name = model.get("name")
            if model_name is not None and model_name not in VALID_MODELS:
                issues.append(
                    f"Unknown model name '{model_name}'. "
                    f"Known models: {sorted(VALID_MODELS)}"
                )

    # -- 9. Consistency: target not in drop_columns --
    if target and isinstance(cfg.get("drop_columns"), list):
        if target in cfg["drop_columns"]:
            issues.append(f"'target' column '{target}' is also in drop_columns")

    # -- 10. Consistency: blank_as_zero vs impute columns --
    blank_cols = cfg.get("blank_as_zero_columns")
    if isinstance(blank_cols, list) and isinstance(impute, dict):
        imp_cols = impute.get("columns", [])
        if isinstance(imp_cols, list):
            overlap = set(blank_cols) & set(imp_cols)
            if overlap:
                issues.append(
                    f"Columns in both blank_as_zero_columns and impute.columns: "
                    f"{sorted(overlap)}. These don't need imputation."
                )

    return issues
