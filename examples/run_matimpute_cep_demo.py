"""Compare median imputation, KNN+ExtraTrees imputation, and CEP features."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import smallmatprep as smp


RANDOM_STATE = 42
TARGET_COL = "conductivity"
FEATURE_COLS = [
    "temperature_K",
    "concentration_mol_L",
    "solvent_A",
    "solvent_B",
    "salt_S",
    "additive_C",
    "density_g_ml",
    "viscosity_index",
]


def make_synthetic_electrolyte(
    n_samples: int = 72,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build a deterministic small electrolyte dataset with missing features."""
    rng = np.random.default_rng(random_state)

    concentration = rng.uniform(0.35, 2.35, n_samples)
    temperature = rng.uniform(288.0, 333.0, n_samples)
    solvent_a = 82.0 - 13.0 * concentration + 0.06 * (temperature - 298.15)
    solvent_a += rng.normal(0.0, 2.0, n_samples)
    solvent_a = np.clip(solvent_a, 48.0, 88.0)
    solvent_b = 100.0 - solvent_a
    salt_s = concentration + rng.normal(0.0, 0.045, n_samples)
    additive = rng.choice([0.0, 0.05, 0.10, 0.20], size=n_samples)
    density = 0.92 + 0.082 * concentration - 0.00035 * (temperature - 298.15)
    density += rng.normal(0.0, 0.008, n_samples)
    viscosity = 1.45 + 0.42 * concentration - 0.006 * (temperature - 298.15)
    viscosity += 0.006 * solvent_b + rng.normal(0.0, 0.025, n_samples)

    complete = pd.DataFrame({
        "temperature_K": temperature,
        "concentration_mol_L": concentration,
        "solvent_A": solvent_a,
        "solvent_B": solvent_b,
        "salt_S": salt_s,
        "additive_C": additive,
        "density_g_ml": density,
        "viscosity_index": viscosity,
    })

    cep = smp.simple_electrolyte_cep(
        complete,
        lambda_0=13.5,
        debye_coeff=1.15,
        activation_energy_j_mol=8500.0,
    )
    residual = (
        0.018 * (solvent_a - 65.0)
        + 0.55 * additive
        - 0.16 * (viscosity - viscosity.mean())
        + rng.normal(0.0, 0.22, n_samples)
    )
    complete[TARGET_COL] = 0.92 * cep + residual

    df = complete.copy()
    missing_specs = {
        "concentration_mol_L": 0.18,
        "salt_S": 0.16,
        "solvent_A": 0.10,
        "solvent_B": 0.10,
        "additive_C": 0.12,
        "density_g_ml": 0.12,
        "viscosity_index": 0.12,
    }
    for col, rate in missing_specs.items():
        mask = rng.random(n_samples) < rate
        df.loc[mask, col] = np.nan

    df.insert(0, "sample_id", [f"E{i:03d}" for i in range(1, n_samples + 1)])
    return df


def evaluate_workflow(
    df: pd.DataFrame,
    imputer_name: str,
    add_cep: bool,
) -> dict:
    """Run cross-validation for one workflow."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    X_raw = df[FEATURE_COLS]
    y = df[TARGET_COL].to_numpy()

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_raw), start=1):
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        if imputer_name == "median":
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train_raw),
                columns=FEATURE_COLS,
                index=X_train_raw.index,
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test_raw),
                columns=FEATURE_COLS,
                index=X_test_raw.index,
            )
        elif imputer_name == "knn_extratrees":
            imputer = smp.KNNExtraTreesImputer(
                feature_cols=FEATURE_COLS,
                n_neighbors=5,
                max_iter=8,
                tol=1e-4,
                n_estimators=120,
                random_state=RANDOM_STATE + fold_idx,
            )
            X_train = imputer.fit_transform(X_train_raw)
            X_test = imputer.transform(X_test_raw)
        else:
            raise ValueError(f"Unknown imputer: {imputer_name}")

        if add_cep:
            cep_formula = lambda d: smp.simple_electrolyte_cep(
                d,
                lambda_0=13.5,
                debye_coeff=1.15,
                activation_energy_j_mol=8500.0,
            )
            X_train = smp.add_cep_feature(X_train, cep_formula)
            X_test = smp.add_cep_feature(X_test, cep_formula)

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=4,
            random_state=RANDOM_STATE + fold_idx,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_metrics.append({
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

    return {
        metric: float(np.mean([m[metric] for m in fold_metrics]))
        for metric in ["MAE", "RMSE", "R2"]
    } | {
        f"{metric}_std": float(np.std([m[metric] for m in fold_metrics]))
        for metric in ["MAE", "RMSE", "R2"]
    }


def main() -> None:
    df = make_synthetic_electrolyte()
    missing_rate = df[FEATURE_COLS].isna().mean().mean()

    experiments = [
        ("A baseline", "median", False),
        ("B MatImpute-lite", "knn_extratrees", False),
        ("C MatImpute-lite + CEP", "knn_extratrees", True),
    ]

    print("== MatImpute-lite + CEP demo ==")
    print(f"Rows: {len(df)} | Feature missing rate: {missing_rate:.1%}")
    print()
    print("Group                     RMSE       MAE        R2     RMSE gain")
    print("----------------------------------------------------------------")

    results = []
    baseline_rmse = None
    for name, imputer_name, use_cep in experiments:
        metrics = evaluate_workflow(df, imputer_name, use_cep)
        if baseline_rmse is None:
            baseline_rmse = metrics["RMSE"]
        gain = (baseline_rmse - metrics["RMSE"]) / baseline_rmse * 100.0
        results.append((name, metrics, gain))
        print(
            f"{name:<24} "
            f"{metrics['RMSE']:>7.3f} "
            f"{metrics['MAE']:>9.3f} "
            f"{metrics['R2']:>8.3f} "
            f"{gain:>9.1f}%"
        )

    best_name, best_metrics, best_gain = min(
        results,
        key=lambda item: item[1]["RMSE"],
    )
    print()
    print(
        f"Best: {best_name} with RMSE={best_metrics['RMSE']:.3f} "
        f"({best_gain:.1f}% lower than baseline)."
    )


if __name__ == "__main__":
    main()
