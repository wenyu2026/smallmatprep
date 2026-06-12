"""Tests for smallmatprep core modules (data, inspect, impute, modeling, evaluate, config)."""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import smallmatprep as smp
from smallmatprep.config_schema import validate_config
from smallmatprep.data.loaders import load_csv, load_config
from smallmatprep.inspect.missingness import missing_report
from smallmatprep.impute.baseline import impute_median, impute_knn
from smallmatprep.impute.material import impute_group_median
from smallmatprep.modeling.recommend import recommend_model
from smallmatprep.evaluate.metrics import evaluate_model


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _make_df(n_rows: int = 50) -> pd.DataFrame:
    """Create a standard test DataFrame with controlled missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        "sample_id": [f"S{i:03d}" for i in range(n_rows)],
        "solvent_A": np.random.uniform(0, 100, n_rows),
        "solvent_B": np.random.uniform(0, 100, n_rows),
        "salt_S": np.random.uniform(0.1, 2.0, n_rows),
        "additive_C": np.random.uniform(0, 5, n_rows),
        "temperature": np.random.uniform(20, 80, n_rows),
        "target": np.random.uniform(0.5, 10, n_rows),
    })


def _inject_missing(df: pd.DataFrame, col: str, frac: float = 0.1) -> pd.DataFrame:
    """Set a fraction of values in a column to NaN."""
    mask = np.random.random(len(df)) < frac
    df = df.copy()
    df.loc[mask, col] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  data/loaders
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaders(unittest.TestCase):
    def test_load_csv_returns_dataframe(self):
        df = load_csv("examples/sample_electrolyte.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_load_csv_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("nonexistent_file.csv")

    def test_load_config_valid_json(self):
        cfg = load_config("examples/config_template.json")
        self.assertIsInstance(cfg, dict)
        self.assertIn("target", cfg)

    def test_load_config_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent.json")


# ══════════════════════════════════════════════════════════════════════════════
#  inspect/missingness
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingness(unittest.TestCase):
    def test_missing_report_returns_dataframe(self):
        df = _make_df()
        report = missing_report(df)
        self.assertIsInstance(report, pd.DataFrame)
        self.assertIn("missing_count", report.columns)
        self.assertIn("missing_rate_%", report.columns)

    def test_missing_report_no_missing(self):
        df = _make_df()
        report = missing_report(df)
        self.assertEqual(report["missing_count"].sum(), 0)

    def test_missing_report_with_missing(self):
        df = _inject_missing(_make_df(), "temperature", frac=0.2)
        report = missing_report(df)
        self.assertGreater(report.loc["temperature", "missing_count"], 0)

    def test_missing_report_empty_df_raises(self):
        df = pd.DataFrame()
        with self.assertRaises(ValueError):
            missing_report(df)


# ══════════════════════════════════════════════════════════════════════════════
#  impute/baseline
# ══════════════════════════════════════════════════════════════════════════════

class TestImpute(unittest.TestCase):
    def setUp(self):
        self.df = _inject_missing(_make_df(30), "temperature", frac=0.3)

    def test_impute_median_fills_nan(self):
        result = impute_median(self.df, feature_cols=["temperature"])
        self.assertFalse(result["temperature"].isna().any())

    def test_impute_knn_fills_nan(self):
        result = impute_knn(self.df, feature_cols=["temperature", "solvent_A"], n_neighbors=3)
        self.assertFalse(result["temperature"].isna().any())

    def test_impute_knn_n_neighbors_too_large(self):
        with self.assertRaises(ValueError):
            impute_knn(self.df, feature_cols=["temperature"], n_neighbors=999)

    def test_impute_median_empty_df_raises(self):
        with self.assertRaises(ValueError):
            impute_median(pd.DataFrame(), feature_cols=["x"])

    def test_impute_median_preserves_non_feature_cols(self):
        result = impute_median(self.df, feature_cols=["temperature"])
        self.assertEqual(list(result.columns), list(self.df.columns))

    def test_impute_group_median_fills_nan(self):
        result = impute_group_median(self.df, feature_cols=["temperature"], group_col="salt_S")
        self.assertFalse(result["temperature"].isna().any())

    def test_impute_group_median_invalid_group_col_raises(self):
        with self.assertRaises(ValueError):
            impute_group_median(self.df, feature_cols=["temperature"], group_col="nonexistent")


# ══════════════════════════════════════════════════════════════════════════════
#  modeling/recommend
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommend(unittest.TestCase):
    def test_recommend_returns_list_of_dicts(self):
        recs = recommend_model(n_samples=50, n_features=8)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIn("model", r)
            self.assertIn("reason", r)

    def test_small_sample_recommends_ridge(self):
        """n_features >= n_samples should recommend Ridge with high-dim reason."""
        recs = recommend_model(n_samples=5, n_features=10)
        models = [r["model"] for r in recs]
        self.assertIn("Ridge", models)

    def test_large_sample_includes_rf(self):
        recs = recommend_model(n_samples=500, n_features=8)
        models = [r["model"] for r in recs]
        self.assertIn("RandomForestRegressor", models)

    def test_invalid_n_samples_raises(self):
        with self.assertRaises(ValueError):
            recommend_model(n_samples=0, n_features=5)

    def test_invalid_n_features_raises(self):
        with self.assertRaises(ValueError):
            recommend_model(n_samples=10, n_features=-1)

    def test_classification_returns_classifiers(self):
        recs = recommend_model(n_samples=50, n_features=5, task="classification")
        models = [r["model"] for r in recs]
        self.assertIn("LogisticRegression", models)
        self.assertIn("RandomForestClassifier", models)
        self.assertIn("SVC", models)

    def test_cep_reduces_rf_depth(self):
        recs_without = recommend_model(n_samples=30, n_features=5, task="regression", cep_available=False)
        recs_with = recommend_model(n_samples=30, n_features=5, task="regression", cep_available=True)
        rf_without = [r for r in recs_without if r["model"] == "RandomForestRegressor"][0]
        rf_with = [r for r in recs_with if r["model"] == "RandomForestRegressor"][0]
        # With CEP, RF should have max_depth=3 (shallower)
        self.assertEqual(rf_with["params"]["max_depth"], 3)
        self.assertGreater(rf_without["params"]["max_depth"], 3)

    def test_classification_small_sample_logistic(self):
        recs = recommend_model(n_samples=5, n_features=10, task="classification")
        models = [r["model"] for r in recs]
        self.assertIn("LogisticRegression", models)

    def test_invalid_task_raises(self):
        with self.assertRaises(ValueError, match="Unsupported task"):
            recommend_model(n_samples=50, n_features=5, task="clustering")


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate/metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.y_true = np.random.randn(50)
        self.y_pred = self.y_true + np.random.normal(0, 0.2, 50)

    def test_evaluate_returns_dict(self):
        metrics = evaluate_model(self.y_true, self.y_pred)
        self.assertIsInstance(metrics, dict)

    def test_evaluate_contains_expected_keys(self):
        metrics = evaluate_model(self.y_true, self.y_pred)
        for key in ("MAE", "RMSE", "R2"):
            self.assertIn(key, metrics)

    def test_evaluate_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate_model(self.y_true, self.y_pred[:10])

    def test_evaluate_perfect_prediction(self):
        metrics = evaluate_model(self.y_true, self.y_true)
        self.assertAlmostEqual(metrics["RMSE"], 0.0, places=10)
        self.assertAlmostEqual(metrics["R2"], 1.0, places=10)

    def test_evaluate_nan_in_y_true_raises(self):
        bad = self.y_true.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_model(bad, self.y_pred)


# ══════════════════════════════════════════════════════════════════════════════
#  config_schema
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSchema(unittest.TestCase):
    def test_valid_config_no_issues(self):
        cfg = {"target": "y", "model": "Ridge"}
        issues = validate_config(cfg)
        self.assertEqual(issues, [])

    def test_missing_target(self):
        cfg: dict = {}
        issues = validate_config(cfg)
        self.assertTrue(any("target" in i.lower() for i in issues))

    def test_unknown_key(self):
        cfg = {"target": "y", "unknown_field": 123}
        issues = validate_config(cfg)
        self.assertTrue(any("unknown" in i.lower() for i in issues))

    def test_invalid_model_string(self):
        cfg = {"target": "y", "model": "NonExistentModel"}
        issues = validate_config(cfg)
        self.assertTrue(any("model" in i.lower() for i in issues))

    def test_valid_model_dict(self):
        cfg = {"target": "y", "model": {"name": "Ridge", "task": "regression"}}
        issues = validate_config(cfg)
        self.assertEqual(issues, [])

    def test_invalid_model_dict_name(self):
        cfg = {"target": "y", "model": {"name": "BadModel"}}
        issues = validate_config(cfg)
        self.assertTrue(any("model" in i.lower() for i in issues))

    def test_target_in_drop_columns(self):
        cfg = {"target": "y", "drop_columns": ["y"]}
        issues = validate_config(cfg, columns=["y"])
        self.assertTrue(any("drop_columns" in i for i in issues))

    def test_column_validation_with_list(self):
        cfg = {"target": "y", "feature_groups": {"g1": ["nonexistent"]}}
        issues = validate_config(cfg, columns=["y", "x1"])
        self.assertTrue(any("nonexistent" in i for i in issues))

    def test_impute_columns_as_dict(self):
        cfg = {
            "target": "y",
            "impute": {"method": "median", "columns": {"col_a": {"method": "median"}}},
        }
        issues = validate_config(cfg, columns=["y", "col_a"])
        self.assertEqual(issues, [])

    def test_impute_columns_as_list(self):
        cfg = {
            "target": "y",
            "impute": {"method": "knn", "columns": ["col_a"]},
        }
        issues = validate_config(cfg, columns=["y", "col_a"])
        self.assertEqual(issues, [])

    def test_blank_as_zero_overlap_warns(self):
        cfg = {
            "target": "y",
            "blank_as_zero_columns": ["additive_C"],
            "impute": {"method": "median", "columns": ["additive_C"]},
        }
        issues = validate_config(cfg)
        self.assertTrue(any("blank_as_zero" in i.lower() for i in issues))


if __name__ == "__main__":
    unittest.main()
