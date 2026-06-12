"""Iterative KNN + ExtraTrees imputation for small materials datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KNNExtraTreesImputer:
    """Lightweight MatImpute-inspired tabular imputer.

    The imputer first initializes missing values with ``KNNImputer``. It then
    refines originally-missing entries by repeatedly fitting one
    ``ExtraTreesRegressor`` per incomplete column, using the other columns as
    predictors.
    """

    feature_cols: list[str]
    n_neighbors: int = 3
    max_iter: int = 10
    tol: float = 1e-3
    n_estimators: int = 100
    random_state: int = 42
    max_depth: int | None = None
    min_training_samples: int = 3

    def fit(self, df: pd.DataFrame) -> "KNNExtraTreesImputer":
        """Fit the imputer on a training DataFrame."""
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import KNNImputer

        if df.empty:
            raise ValueError("Cannot impute an empty DataFrame")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        self.columns_ = _numeric_existing_columns(df, self.feature_cols)
        self.models_ = {}
        self.iter_history_ = []
        self.converged_ = True
        self.n_iter_ = 0

        if not self.columns_:
            self.effective_n_neighbors_ = 0
            self.knn_imputer_ = None
            self.train_imputed_ = None
            return self

        values = df[self.columns_].astype(float).to_numpy()
        _validate_numeric_matrix(values, self.columns_)
        _raise_on_all_missing(values, self.columns_)

        n_samples, n_features = values.shape
        self.effective_n_neighbors_ = min(self.n_neighbors, n_samples)
        self.knn_imputer_ = KNNImputer(n_neighbors=self.effective_n_neighbors_)
        working = self.knn_imputer_.fit_transform(values)

        missing_mask = np.isnan(values)
        if not missing_mask.any() or n_features < 2:
            self.train_imputed_ = working
            return self

        depth = self.max_depth
        if depth is None and n_samples < 20:
            depth = 3

        for iter_idx in range(self.max_iter):
            previous = working.copy()
            models_this_iter = {}

            for target_idx in range(n_features):
                target_missing = missing_mask[:, target_idx]
                if not target_missing.any():
                    continue

                observed = ~np.isnan(values[:, target_idx])
                if observed.sum() < self.min_training_samples:
                    continue

                other_idx = [
                    idx for idx in range(n_features)
                    if idx != target_idx
                ]
                model = ExtraTreesRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=depth,
                    random_state=self.random_state + iter_idx * 100 + target_idx,
                    n_jobs=1,
                )
                model.fit(
                    working[observed][:, other_idx],
                    values[observed, target_idx],
                )
                working[target_missing, target_idx] = model.predict(
                    working[target_missing][:, other_idx]
                )
                models_this_iter[target_idx] = model

            if not models_this_iter:
                break

            max_change = _max_missing_change(previous, working, missing_mask)
            self.iter_history_.append(float(max_change))
            self.models_ = models_this_iter
            self.n_iter_ = iter_idx + 1

            if max_change <= self.tol:
                break
        else:
            self.converged_ = False

        self.train_imputed_ = working
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in a new DataFrame using the fitted state."""
        if not hasattr(self, "columns_"):
            raise ValueError("KNNExtraTreesImputer must be fitted first")

        out = df.copy()
        if not self.columns_:
            return out

        missing_cols = [col for col in self.columns_ if col not in out.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in transform data: {missing_cols}")

        values = out[self.columns_].astype(float).to_numpy()
        _validate_numeric_matrix(values, self.columns_)
        original_missing = np.isnan(values)

        working = self.knn_imputer_.transform(values)
        for target_idx, model in self.models_.items():
            target_missing = original_missing[:, target_idx]
            if not target_missing.any():
                continue
            other_idx = [
                idx for idx in range(len(self.columns_))
                if idx != target_idx
            ]
            working[target_missing, target_idx] = model.predict(
                working[target_missing][:, other_idx]
            )

        out.loc[:, self.columns_] = working
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the imputer and return the imputed training DataFrame."""
        self.fit(df)
        out = df.copy()
        if self.columns_:
            out.loc[:, self.columns_] = self.train_imputed_
        return out

    def info(self) -> dict:
        """Return convergence metadata."""
        return {
            "columns": list(getattr(self, "columns_", [])),
            "effective_n_neighbors": getattr(self, "effective_n_neighbors_", 0),
            "n_iter": getattr(self, "n_iter_", 0),
            "converged": getattr(self, "converged_", True),
            "history": list(getattr(self, "iter_history_", [])),
        }


def impute_knn_extratrees(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_neighbors: int = 3,
    max_iter: int = 10,
    tol: float = 1e-3,
    n_estimators: int = 100,
    random_state: int = 42,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Fill missing numeric feature values with KNN initialization and trees.

    This is a lightweight, sklearn-only imputer inspired by MatImpute's
    two-stage idea: initialize with nearest-neighbor information, then refine
    with nonlinear feature relationships.
    """
    imputer = KNNExtraTreesImputer(
        feature_cols=feature_cols,
        n_neighbors=n_neighbors,
        max_iter=max_iter,
        tol=tol,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    out = imputer.fit_transform(df)
    if return_info:
        return out, imputer.info()
    return out


def _numeric_existing_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    return [
        col for col in feature_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def _validate_numeric_matrix(values: np.ndarray, columns: list[str]) -> None:
    if np.isinf(values).any():
        raise ValueError(f"Cannot impute Inf values in columns: {columns}")


def _raise_on_all_missing(values: np.ndarray, columns: list[str]) -> None:
    all_missing = [
        columns[idx] for idx in range(values.shape[1])
        if np.isnan(values[:, idx]).all()
    ]
    if all_missing:
        raise ValueError(
            "Cannot impute columns with all values missing: "
            f"{all_missing}. Provide at least one observed value."
        )


def _max_missing_change(
    previous: np.ndarray,
    current: np.ndarray,
    missing_mask: np.ndarray,
) -> float:
    if not missing_mask.any():
        return 0.0
    diff = np.abs(current[missing_mask] - previous[missing_mask])
    if diff.size == 0:
        return 0.0
    return float(np.nanmax(diff))
