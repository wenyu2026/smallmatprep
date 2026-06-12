"""Demo: material-constrained imputation vs unconstrained baselines.

Compares:
1. Median imputation
2. KNN+ExtraTrees (MatImpute-lite)
3. KNN+ExtraTrees + Range constraints
4. KNN+ExtraTrees + Range + Similarity constraints
5. KNN+ExtraTrees + CEP features (best known baseline)
6. KNN+ExtraTrees + Constraints + CEP (this work)

All on a small synthetic electrolyte dataset (n=72, ~12% missing).
"""
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


# ── Reuse the synthetic data generator from the CEP demo ──
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
    return df, complete  # return both incomplete and ground truth


# ── Physical range constraints for electrolyte properties ──
ELECTROLYTE_RANGES = {
    "temperature_K": (288.0, 333.0),       # observed range
    "concentration_mol_L": (0.35, 2.35),   # observed range
    "solvent_A": (48.0, 88.0),             # observed range
    "solvent_B": (12.0, 52.0),             # 100 - A range
    "salt_S": (0.2, 2.8),                  # observed range
    "additive_C": (0.0, 0.20),             # [0, 20%]
    "density_g_ml": (0.85, 1.2),           # physical + observed
    "viscosity_index": (1.0, 6.0),         # observed range
}


def evaluate_workflow(
    df_incomplete: pd.DataFrame,
    df_complete: pd.DataFrame,
    method: str,
    add_cep: bool = False,
) -> dict:
    """Cross-validation evaluation for one imputation method.

    Uses the ground-truth *df_complete* to compute imputation RMSE
    (how accurately missing values were filled) and downstream RMSE
    (predictive performance after imputation).
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_impute_rmse = []
    fold_downstream = []

    X_raw = df_incomplete[FEATURE_COLS]
    y = df_complete[TARGET_COL].to_numpy()

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_raw), start=1):
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # ── Ground truth for imputation error ──
        X_complete_train = df_complete[FEATURE_COLS].iloc[train_idx].to_numpy()
        missing_train = np.isnan(X_train_raw[FEATURE_COLS].to_numpy())

        if method == "median":
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

        elif method == "knn_extratrees":
            imp = smp.KNNExtraTreesImputer(
                feature_cols=FEATURE_COLS,
                n_neighbors=5,
                max_iter=8,
                tol=1e-4,
                n_estimators=120,
                random_state=RANDOM_STATE + fold_idx,
            )
            X_train = imp.fit_transform(X_train_raw)
            X_test = imp.transform(X_test_raw)

        elif method == "constrained_range":
            imp = smp.KNNExtraTreesImputer(
                feature_cols=FEATURE_COLS,
                n_neighbors=5,
                max_iter=8,
                tol=1e-4,
                n_estimators=120,
                random_state=RANDOM_STATE + fold_idx,
            )
            X_train_imp = imp.fit_transform(X_train_raw)
            X_test_imp = imp.transform(X_test_raw)

            # Apply range constraints
            train_mask = np.isnan(X_train_raw[FEATURE_COLS].to_numpy())
            test_mask = np.isnan(X_test_raw[FEATURE_COLS].to_numpy())

            X_train = smp.apply_range_constraint(
                X_train_imp, ELECTROLYTE_RANGES,
                missing_mask={c: train_mask[:, FEATURE_COLS.index(c)]
                              for c in ELECTROLYTE_RANGES if c in FEATURE_COLS},
            )
            X_test = smp.apply_range_constraint(
                X_test_imp, ELECTROLYTE_RANGES,
                missing_mask={c: test_mask[:, FEATURE_COLS.index(c)]
                              for c in ELECTROLYTE_RANGES if c in FEATURE_COLS},
            )

        elif method == "constrained_full":
            imp = smp.KNNExtraTreesImputer(
                feature_cols=FEATURE_COLS,
                n_neighbors=5,
                max_iter=8,
                tol=1e-4,
                n_estimators=120,
                random_state=RANDOM_STATE + fold_idx,
            )
            X_train_imp = imp.fit_transform(X_train_raw)
            X_test_imp = imp.transform(X_test_raw)

            train_mask = np.isnan(X_train_raw[FEATURE_COLS].to_numpy())
            test_mask = np.isnan(X_test_raw[FEATURE_COLS].to_numpy())

            # Range + similarity constraints
            X_train = smp.apply_range_constraint(
                X_train_imp, ELECTROLYTE_RANGES,
                missing_mask={c: train_mask[:, FEATURE_COLS.index(c)]
                              for c in ELECTROLYTE_RANGES if c in FEATURE_COLS},
            )
            X_train = smp.apply_similarity_constraint(
                X_train, FEATURE_COLS, train_mask,
                n_neighbors=3, alpha=0.25,
            )
            X_test = smp.apply_range_constraint(
                X_test_imp, ELECTROLYTE_RANGES,
                missing_mask={c: test_mask[:, FEATURE_COLS.index(c)]
                              for c in ELECTROLYTE_RANGES if c in FEATURE_COLS},
            )
            X_test = smp.apply_similarity_constraint(
                X_test, FEATURE_COLS, test_mask,
                n_neighbors=3, alpha=0.25,
            )

        elif method == "constrained_uncertainty":
            # Fit a master imputer once, then use it as template
            master_imp = smp.KNNExtraTreesImputer(
                feature_cols=FEATURE_COLS,
                n_neighbors=5,
                max_iter=8,
                tol=1e-4,
                n_estimators=120,
                random_state=RANDOM_STATE + fold_idx,
            )
            master_imp.fit(X_train_raw)

            call_counter = [0]

            def _imp_fn(df, cols):
                """Impute with varying random_state for uncertainty."""
                call_counter[0] += 1
                run_seed = (RANDOM_STATE + fold_idx * 100 + call_counter[0] * 7)
                imp = smp.KNNExtraTreesImputer(
                    feature_cols=cols,
                    n_neighbors=5,
                    max_iter=8,
                    tol=1e-4,
                    n_estimators=120,
                    random_state=run_seed,
                )
                return imp.fit_transform(df)

            # Use uncertainty: mean of multiple runs
            print(f"        [fold {fold_idx+1}] 启动不确定性填充 (5 runs)...")
            train_mean, train_std = smp.impute_with_uncertainty(
                X_train_raw, FEATURE_COLS, _imp_fn,
                n_runs=10, random_state=RANDOM_STATE + fold_idx * 3,
            )
            print(f"        [fold {fold_idx+1}] 不确定性填充完成")
            # For test, use the master imputer
            X_test_imp = master_imp.transform(X_test_raw)
            test_mask = np.isnan(X_test_raw[FEATURE_COLS].to_numpy())
            X_test = smp.apply_range_constraint(
                X_test_imp, ELECTROLYTE_RANGES,
                missing_mask={c: test_mask[:, FEATURE_COLS.index(c)]
                              for c in ELECTROLYTE_RANGES if c in FEATURE_COLS},
            )

            X_train = train_mean
            # Also clip train to ranges
            X_train = smp.apply_range_constraint(
                X_train, ELECTROLYTE_RANGES,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Imputation accuracy ──
        if missing_train.any():
            impute_err = np.mean((
                X_train[FEATURE_COLS].to_numpy()[missing_train]
                - X_complete_train[missing_train]
            ) ** 2)
            fold_impute_rmse.append(np.sqrt(impute_err))

        # ── CEP features (optional) ──
        if add_cep:
            cep_fn = lambda d: smp.simple_electrolyte_cep(
                d, lambda_0=13.5, debye_coeff=1.15, activation_energy_j_mol=8500.0,
            )
            X_train = smp.add_cep_feature(X_train, cep_fn)
            X_test = smp.add_cep_feature(X_test, cep_fn)

        # ── Downstream model ──
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=4,
            random_state=RANDOM_STATE + fold_idx,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_downstream.append({
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

    result = {
        metric: float(np.mean([m[metric] for m in fold_downstream]))
        for metric in ["MAE", "RMSE", "R2"]
    }
    result.update({
        f"{metric}_std": float(np.std([m[metric] for m in fold_downstream]))
        for metric in ["MAE", "RMSE", "R2"]
    })
    if fold_impute_rmse:
        result["ImputeRMSE"] = float(np.mean(fold_impute_rmse))
    return result


def main() -> None:
    df_incomplete, df_complete = make_synthetic_electrolyte()
    missing_rate = df_incomplete[FEATURE_COLS].isna().mean().mean()

    experiments = [
        ("① Median (baseline)",          "median",             False),
        ("② MatImpute-lite",             "knn_extratrees",     False),
        ("③ + Range constraints",        "constrained_range",  False),
        ("④ + Range + Similarity",       "constrained_full",   False),
        ("⑤ MatImpute-lite + CEP",       "knn_extratrees",     True),
        ("⑥ + Constraints + CEP ★",      "constrained_full",   True),
        ("⑦ + Uncertainty + CEP",        "constrained_uncertainty", True),
    ]

    print("=" * 75)
    print("  材料约束填充对比实验 — Material-Constrained Imputation")
    print("=" * 75)
    print(f"  样本数: {len(df_incomplete)}  |  特征缺失率: {missing_rate:.1%}")
    print(f"  验证方式: 5-fold CV  |  下游模型: RandomForest")
    print()

    header = f"{'方法':<28s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'ImputeRMSE':>10s} {'RMSE提升':>9s}"
    print(header)
    print("-" * len(header))

    results = []
    baseline_rmse = None
    for name, method, use_cep in experiments:
        metrics = evaluate_workflow(
            df_incomplete, df_complete, method, add_cep=use_cep,
        )
        if baseline_rmse is None:
            baseline_rmse = metrics.get("RMSE", 0)

        gain = (baseline_rmse - metrics["RMSE"]) / baseline_rmse * 100.0
        results.append((name, metrics, gain))

        impute_rmse_str = f"{metrics.get('ImputeRMSE', 0):.4f}" if "ImputeRMSE" in metrics else "   N/A  "
        print(
            f"{name:<28s} "
            f"{metrics['RMSE']:>8.4f} "
            f"{metrics['MAE']:>8.4f} "
            f"{metrics['R2']:>8.4f} "
            f"{impute_rmse_str:>10s} "
            f"{gain:>+8.2f}%"
        )

    print()
    print("─" * 75)
    print("  解读:")
    print("  • ImputeRMSE = 填充值 vs 真实值的 RMSE（越低越好）")
    print("  • RMSE/MAE/R2 = 下游预测任务指标")
    print("  • RMSE提升 = 相对于 Median baseline 的 RMSE 降幅")
    print("  • ★ = 推荐配置：物理约束 + CEP 先验特征协同")
    print()

    # ── Summary for PROJECT.md ──
    print("=" * 75)
    print("  实验结果摘要（可直接记录到 PROJECT.md）")
    print("=" * 75)
    for name, metrics, gain in results:
        impute_str = f", ImputeRMSE={metrics.get('ImputeRMSE', 0):.4f}" if "ImputeRMSE" in metrics else ""
        print(f"  {name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}{impute_str}, gain={gain:+.2f}%")


if __name__ == "__main__":
    main()
