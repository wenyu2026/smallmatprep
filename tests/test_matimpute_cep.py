import numpy as np
import pandas as pd
import pytest

import smallmatprep as smp


def test_impute_knn_extratrees_fills_missing_numeric_values():
    df = pd.DataFrame({
        "temperature_K": [298.0, 303.0, 308.0, 313.0, 318.0],
        "concentration_mol_L": [0.8, np.nan, 1.2, 1.4, np.nan],
        "salt_S": [0.82, 1.02, 1.18, np.nan, 1.62],
        "label": ["a", "b", "c", "d", "e"],
    })

    out, info = smp.impute_knn_extratrees(
        df,
        ["temperature_K", "concentration_mol_L", "salt_S"],
        n_neighbors=2,
        max_iter=3,
        n_estimators=20,
        return_info=True,
    )

    assert not out[["temperature_K", "concentration_mol_L", "salt_S"]].isna().any().any()
    assert out["label"].tolist() == df["label"].tolist()
    assert info["effective_n_neighbors"] == 2


def test_impute_knn_extratrees_raises_on_all_missing_column():
    df = pd.DataFrame({
        "a": [np.nan, np.nan, np.nan],
        "b": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ValueError, match="all values missing"):
        smp.impute_knn_extratrees(df, ["a", "b"])


def test_knn_extratrees_imputer_can_transform_new_data():
    train = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],
        "c": [1.5, 3.5, 5.5, 7.5],
    })
    test = pd.DataFrame({
        "a": [np.nan, 5.0],
        "b": [10.0, np.nan],
        "c": [9.5, 11.5],
    })

    imputer = smp.KNNExtraTreesImputer(
        feature_cols=["a", "b", "c"],
        n_neighbors=2,
        max_iter=3,
        n_estimators=20,
    )
    imputer.fit(train)
    out = imputer.transform(test)

    assert not out.isna().any().any()


def test_add_cep_feature_with_simple_electrolyte_formula():
    df = pd.DataFrame({
        "temperature_K": [298.15, 310.0, 320.0],
        "concentration_mol_L": [0.5, 1.0, 1.5],
    })

    out = smp.add_cep_feature(df, smp.simple_electrolyte_cep)

    assert "cep_estimate" in out.columns
    assert np.isfinite(out["cep_estimate"]).all()
    assert (out["cep_estimate"] >= 0).all()


def test_add_cep_feature_rejects_wrong_length_formula():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="returned 2 values"):
        smp.add_cep_feature(df, lambda _: [1.0, 2.0])


def test_public_exports_available():
    assert callable(smp.impute_knn_extratrees)
    assert callable(smp.add_cep_feature)
    assert callable(smp.simple_electrolyte_cep)
    assert callable(smp.sample_diagnosis)
    assert callable(smp.missing_pattern_report)
    assert callable(smp.bias_variance_decomposition)
    assert callable(smp.decomposition_summary)
