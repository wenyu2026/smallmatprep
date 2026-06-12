"""Tests for smallmatprep.inspect.diagnosis and missing_pattern_report."""
import numpy as np
import pandas as pd
import pytest

from smallmatprep.inspect.diagnosis import (
    sample_diagnosis,
    missing_pattern_report,
    _categorize_size,
    _recommend_complexity,
    _estimate_error_source,
)


# ══════════════════════════════════════════════════════════════════════════════
#  _categorize_size
# ══════════════════════════════════════════════════════════════════════════════

class TestCategorizeSize:
    def test_very_small(self):
        assert _categorize_size(10) == "very_small"
        assert _categorize_size(29) == "very_small"

    def test_small(self):
        assert _categorize_size(30) == "small"
        assert _categorize_size(50) == "small"
        assert _categorize_size(99) == "small"

    def test_moderate(self):
        assert _categorize_size(100) == "moderate"
        assert _categorize_size(250) == "moderate"
        assert _categorize_size(499) == "moderate"

    def test_adequate(self):
        assert _categorize_size(500) == "adequate"
        assert _categorize_size(1000) == "adequate"


# ══════════════════════════════════════════════════════════════════════════════
#  _recommend_complexity
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendComplexity:
    def test_very_small_sample(self):
        assert _recommend_complexity(20, 5, 4.0) == "simple_linear"

    def test_low_ratio(self):
        assert _recommend_complexity(10, 10, 1.0) == "simple_linear"
        assert _recommend_complexity(40, 30, 1.33) == "simple_linear"

    def test_moderate(self):
        assert _recommend_complexity(50, 5, 10.0) == "moderate_nonlinear"
        assert _recommend_complexity(80, 5, 16.0) == "moderate_nonlinear"

    def test_flexible(self):
        assert _recommend_complexity(500, 10, 50.0) == "flexible_ensemble"


# ══════════════════════════════════════════════════════════════════════════════
#  _estimate_error_source
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateErrorSource:
    def test_variance_dominated_ultra_high_dim(self):
        assert _estimate_error_source(30, 50, 0.6) == "variance_dominated"

    def test_bias_dominated_very_small(self):
        assert _estimate_error_source(30, 5, 6.0) == "bias_dominated"
        assert _estimate_error_source(49, 10, 4.9) == "bias_dominated"

    def test_bias_dominated_small_ratio(self):
        assert _estimate_error_source(80, 30, 2.67) == "bias_dominated"

    def test_balanced(self):
        assert _estimate_error_source(300, 10, 30.0) == "balanced"
        assert _estimate_error_source(500, 20, 25.0) == "balanced"


# ══════════════════════════════════════════════════════════════════════════════
#  sample_diagnosis — integration
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleDiagnosis:
    def test_returns_dict_with_expected_keys(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
        result = sample_diagnosis(df)
        expected_keys = {
            "n_samples", "n_features", "sample_size_category",
            "sample_to_feature_ratio", "estimated_dof",
            "recommended_max_complexity", "error_source_hint",
            "cep_recommended", "warnings", "advice",
        }
        assert expected_keys.issubset(result.keys())

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            sample_diagnosis(pd.DataFrame())

    def test_very_small_classification(self):
        df = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6]})
        result = sample_diagnosis(df)
        assert result["n_samples"] == 5
        assert result["n_features"] == 2
        assert result["sample_size_category"] == "very_small"
        assert result["recommended_max_complexity"] == "simple_linear"

    def test_target_col_excluded(self):
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "target": [0.1, 0.2, 0.3],
        })
        result = sample_diagnosis(df, target_col="target")
        assert result["n_features"] == 2

    def test_feature_cols_explicit(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
            "target": [0.1, 0.2, 0.3],
        })
        result = sample_diagnosis(df, feature_cols=["a", "b", "target"])
        assert result["n_features"] == 3

    def test_cep_recommended_for_small_sample(self):
        df = pd.DataFrame({"x1": range(20), "x2": range(20)})
        result = sample_diagnosis(df, cep_available=True)
        assert result["cep_recommended"] is True

    def test_cep_not_recommended_when_unavailable(self):
        df = pd.DataFrame({"x1": range(20), "x2": range(20)})
        result = sample_diagnosis(df, cep_available=False)
        assert result["cep_recommended"] is False

    def test_no_feature_columns(self):
        df = pd.DataFrame({"label": ["a", "b", "c"]})
        result = sample_diagnosis(df)
        assert result["n_features"] == 0
        assert "No feature columns" in result["warnings"][0]

    def test_large_dataset_no_warnings(self):
        rng = np.random.default_rng(42)
        data = {f"f{i}": rng.random(500) for i in range(5)}
        data["target"] = rng.random(500)
        df = pd.DataFrame(data)
        result = sample_diagnosis(df, target_col="target")
        assert result["sample_size_category"] == "adequate"
        assert result["error_source_hint"] == "balanced"
        assert len(result["warnings"]) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  missing_pattern_report
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingPatternReport:
    def test_empty_df(self):
        result = missing_pattern_report(pd.DataFrame())
        assert result["pair_overlap"] == []
        assert result["perfect_overlap"] == []

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = missing_pattern_report(df)
        assert result["pair_overlap"] == []

    def test_single_missing_column(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        result = missing_pattern_report(df)
        assert result["pair_overlap"] == []

    def test_pair_overlap_detected(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, 3.0, 4.0],
            "b": [np.nan, np.nan, 3.5, np.nan],
            "c": [1.0, 2.0, 3.0, 4.0],
        })
        result = missing_pattern_report(df)
        assert len(result["pair_overlap"]) >= 1
        assert len(result["top_pairs"]) >= 1

    def test_perfect_overlap(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, 3.0],
            "b": [np.nan, np.nan, 4.0],
            "c": [1.0, 2.0, 3.0],
        })
        result = missing_pattern_report(df)
        assert ("a", "b") in result["perfect_overlap"] or ("b", "a") in result["perfect_overlap"]
