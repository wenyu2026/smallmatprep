"""Evaluate SmallMatPrep on real electrolyte data from v2/data."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import smallmatprep as smp


DATA_DIR = ROOT.parent / "v2" / "data"
BOSAILISI_PATH = DATA_DIR / "bosailisi.xlsx"
SUPPLEMENT_PATH = DATA_DIR / "supplement.xlsx"
COMPONENT_COLS = list("ABCDEFGHIJKLMNOPQRSTUVWX")
CATEGORICAL_COLS = ["CA", "GR", "体系方案"]
TARGET_COLS = ["P15", "P17"]
RANDOM_STATE = 42


def load_real_electrolyte_data() -> pd.DataFrame:
    """Load and align bosailisi.xlsx and supplement.xlsx."""
    raw = pd.read_excel(BOSAILISI_PATH, sheet_name="Sheet1", header=None)
    columns = [
        str(col) if not pd.isna(col) else f"blank_{idx}"
        for idx, col in enumerate(raw.iloc[1].tolist())
    ]
    bos = raw.iloc[4:].copy()
    bos.columns = columns

    supplement = pd.read_excel(SUPPLEMENT_PATH, sheet_name="Sheet1")
    keep_cols = CATEGORICAL_COLS + COMPONENT_COLS + TARGET_COLS

    frames = []
    for frame in [bos, supplement]:
        aligned = frame[[col for col in keep_cols if col in frame.columns]].copy()
        for col in keep_cols:
            if col not in aligned.columns:
                aligned[col] = np.nan
        aligned = aligned[keep_cols]
        frames.append(aligned)

    df = pd.concat(frames, ignore_index=True)
    for col in COMPONENT_COLS + TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # In both spreadsheets, system columns are written like merged cells.
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].ffill().fillna("__missing__")
    return df


def evaluate_target(
    df: pd.DataFrame,
    target_col: str,
    strategy: str,
    proxy_col: str | None = None,
) -> dict:
    """Evaluate one preprocessing strategy for one target."""
    needed_targets = [target_col]
    if proxy_col:
        needed_targets.append(proxy_col)
    data = df.dropna(subset=needed_targets).reset_index(drop=True)

    y = data[target_col].astype(float).to_numpy()
    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(data), start=1):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()

        x_train_num = train[COMPONENT_COLS].copy()
        x_test_num = test[COMPONENT_COLS].copy()

        if strategy == "rule_zero":
            x_train_num = x_train_num.fillna(0.0)
            x_test_num = x_test_num.fillna(0.0)
        elif strategy == "median":
            imputer = SimpleImputer(strategy="median")
            x_train_num = pd.DataFrame(
                imputer.fit_transform(x_train_num),
                columns=COMPONENT_COLS,
                index=train.index,
            )
            x_test_num = pd.DataFrame(
                imputer.transform(x_test_num),
                columns=COMPONENT_COLS,
                index=test.index,
            )
        elif strategy == "knn":
            n_neighbors = min(5, max(1, len(train)))
            imputer = KNNImputer(n_neighbors=n_neighbors)
            x_train_num = pd.DataFrame(
                imputer.fit_transform(x_train_num),
                columns=COMPONENT_COLS,
                index=train.index,
            )
            x_test_num = pd.DataFrame(
                imputer.transform(x_test_num),
                columns=COMPONENT_COLS,
                index=test.index,
            )
        elif strategy == "matimpute_lite":
            valid_cols = [
                col for col in COMPONENT_COLS
                if train[col].notna().sum() >= 3
            ]
            x_train_num = x_train_num[valid_cols]
            x_test_num = x_test_num[valid_cols]
            imputer = smp.KNNExtraTreesImputer(
                feature_cols=valid_cols,
                n_neighbors=5,
                max_iter=6,
                tol=1e-4,
                n_estimators=30,
                random_state=RANDOM_STATE + fold_idx,
            )
            x_train_num = imputer.fit_transform(x_train_num)
            x_test_num = imputer.transform(x_test_num)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if proxy_col:
            x_train_num = x_train_num.copy()
            x_test_num = x_test_num.copy()
            x_train_num[f"{proxy_col}_cep"] = train[proxy_col].astype(float).to_numpy()
            x_test_num[f"{proxy_col}_cep"] = test[proxy_col].astype(float).to_numpy()

        encoder = _make_onehot_encoder()
        x_train_cat = encoder.fit_transform(train[CATEGORICAL_COLS].astype(str))
        x_test_cat = encoder.transform(test[CATEGORICAL_COLS].astype(str))
        x_train = np.hstack([x_train_num.astype(float).to_numpy(), x_train_cat])
        x_test = np.hstack([x_test_num.astype(float).to_numpy(), x_test_cat])

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=3,
            random_state=RANDOM_STATE + fold_idx,
        )
        model.fit(x_train, y[train_idx])
        pred = model.predict(x_test)
        y_test = y[test_idx]
        fold_metrics.append({
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

    return {
        metric: float(np.mean([row[metric] for row in fold_metrics]))
        for metric in ["MAE", "RMSE", "R2"]
    } | {
        f"{metric}_std": float(np.std([row[metric] for row in fold_metrics]))
        for metric in ["MAE", "RMSE", "R2"]
    } | {
        "n_samples": int(len(data)),
    }


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def print_results(title: str, rows: list[tuple[str, dict]]) -> None:
    baseline_rmse = rows[0][1]["RMSE"]
    print(title)
    print("Strategy                 n     RMSE       MAE        R2     RMSE gain")
    print("-----------------------------------------------------------------------")
    for name, metrics in rows:
        gain = (baseline_rmse - metrics["RMSE"]) / baseline_rmse * 100.0
        print(
            f"{name:<22} "
            f"{metrics['n_samples']:>3d} "
            f"{metrics['RMSE']:>8.2f} "
            f"{metrics['MAE']:>9.2f} "
            f"{metrics['R2']:>8.3f} "
            f"{gain:>9.1f}%"
        )
    print()


def main() -> None:
    df = load_real_electrolyte_data()
    target_ready = df[TARGET_COLS].dropna()
    print("== Real electrolyte data demo ==")
    print(f"Source rows: {len(df)} | Rows with P15/P17: {len(target_ready)}")
    print(
        "A-X blank rate: "
        f"{df[COMPONENT_COLS].isna().sum().sum() / df[COMPONENT_COLS].size:.1%}"
    )
    print(f"P15/P17 Pearson correlation: {target_ready.corr().iloc[0, 1]:.3f}")
    print()

    for target_col, proxy_col in [("P17", "P15"), ("P15", "P17")]:
        rows = [
            ("rule zero", evaluate_target(df, target_col, "rule_zero")),
            ("median as missing", evaluate_target(df, target_col, "median")),
            ("KNN as missing", evaluate_target(df, target_col, "knn")),
            ("MatImpute-lite", evaluate_target(df, target_col, "matimpute_lite")),
            (
                f"rule zero + {proxy_col}",
                evaluate_target(df, target_col, "rule_zero", proxy_col=proxy_col),
            ),
        ]
        print_results(f"Target: {target_col}", rows)


if __name__ == "__main__":
    main()
