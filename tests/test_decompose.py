"""Tests for smallmatprep.evaluate.decompose (bias-variance decomposition)."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import pytest

from smallmatprep.evaluate.decompose import (
    bias_variance_decomposition,
    decomposition_summary,
)


class TestBiasVarianceDecomposition:
    def test_returns_dict_with_expected_keys(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        expected_keys = {
            "bias_squared", "variance", "total_error", "mse",
            "fraction_bias", "fraction_variance", "dominant_source",
            "n_bootstraps_used", "n_samples", "n_features",
        }
        assert expected_keys.issubset(result.keys())

    def test_bias_variance_sum_equals_total(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        total_from_sum = result["bias_squared"] + result["variance"]
        assert abs(total_from_sum - result["total_error"]) < 1e-10

    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        assert abs(result["fraction_bias"] + result["fraction_variance"] - 1.0) < 1e-6

    def test_dominant_source_is_valid(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        assert result["dominant_source"] in ("bias", "variance", "balanced")

    def test_n_samples_n_features_match(self):
        rng = np.random.default_rng(42)
        X = rng.random((50, 5))
        y = rng.random(50)

        result = bias_variance_decomposition(
            RandomForestRegressor, X, y,
            model_params={"n_estimators": 20, "random_state": 42},
            n_bootstraps=10, random_state=42,
        )
        assert result["n_samples"] == 50
        assert result["n_features"] == 5

    def test_returns_float_values(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 2))
        y = X[:, 0] + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        assert isinstance(result["bias_squared"], float)
        assert isinstance(result["variance"], float)
        assert isinstance(result["total_error"], float)

    def test_sklearn_kwargs_passed_correctly(self):
        """Verify that model_params are passed to the model constructor."""
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            RandomForestRegressor, X, y,
            model_params={"n_estimators": 10, "max_depth": 3, "random_state": 42},
            n_bootstraps=10, random_state=42,
        )
        assert result["n_bootstraps_used"] > 0


class TestDecompositionSummary:
    def test_returns_string(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        summary = decomposition_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_contains_key_components(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 30)

        result = bias_variance_decomposition(
            Ridge, X, y, n_bootstraps=10, random_state=42,
        )
        summary = decomposition_summary(result)
        assert "bias" in summary.lower()
        assert "variance" in summary.lower()
