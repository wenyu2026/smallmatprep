"""Benchmark KNN, iterative, and material-aware imputation on electrolyte data.

The script uses the Zenodo lithium-ion electrolyte conductivity dataset
(DOI: 10.5281/zenodo.7244939). It can download the CSV into a local cache, or
run from a manually provided cached CSV when network access is unavailable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smallmatprep.impute import impute_group_median, impute_knn_extratrees  # noqa: E402


DATA_URL = (
    "https://zenodo.org/record/7244939/files/Conductivtiy_experiment.csv"
    "?download=1"
)
DATA_DOI = "10.5281/zenodo.7244939"
DEFAULT_DATA_PATH = ROOT / "data" / "external" / "zenodo_7244939" / "Conductivtiy_experiment.csv"
DEFAULT_REPORT_CSV = ROOT / "reports" / "electrolyte_impute_benchmark.csv"
DEFAULT_REPORT_MD = ROOT / "reports" / "electrolyte_impute_benchmark.md"


@dataclass(frozen=True)
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    group: pd.Series
    target_col: str
    group_col: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sklearn KNN imputation with SmallMatPrep iterative and "
            "material-aware imputation on a real electrolyte conductivity dataset."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--download", action="store_true", help="Download the Zenodo CSV if missing.")
    parser.add_argument("--sample-size", type=int, action="append", dest="sample_sizes")
    parser.add_argument("--missing-rate", type=float, action="append", dest="missing_rates")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sizes = args.sample_sizes or [100, 200, 300]
    missing_rates = args.missing_rates or [0.1, 0.2, 0.3]

    csv_path = ensure_dataset(args.data_path, args.download)
    raw = read_electrolyte_csv(csv_path)
    prepared = prepare_electrolyte_table(raw)

    rows: list[dict[str, object]] = []
    for sample_size in sample_sizes:
        for missing_rate in missing_rates:
            for seed in args.seeds:
                rows.extend(
                    run_one_setting(
                        prepared,
                        sample_size=sample_size,
                        missing_rate=missing_rate,
                        seed=seed,
                        n_neighbors=args.n_neighbors,
                        cv_folds=args.cv_folds,
                    )
                )

    results = pd.DataFrame(rows)
    summary = summarize_results(results)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    args.out_md.write_text(build_markdown_report(results, summary, prepared), encoding="utf-8")

    print(f"Rows evaluated: {len(results)}")
    print(f"Raw results: {args.out_csv}")
    print(f"Summary report: {args.out_md}")
    print(summary.to_string(index=False))


def ensure_dataset(path: Path, download: bool) -> Path:
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Download it with --download, or manually place the Zenodo CSV there.\n"
            f"Source DOI: {DATA_DOI}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} -> {path}")
    urllib.request.urlretrieve(DATA_URL, path)
    return path


def read_electrolyte_csv(path: Path) -> pd.DataFrame:
    """Read the Zenodo CSV, which is semicolon-delimited despite its suffix."""
    try:
        return pd.read_csv(path, sep=";")
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=None, engine="python")


def prepare_electrolyte_table(raw: pd.DataFrame) -> PreparedData:
    df = raw.copy()
    target_col = find_column(
        df,
        required_tokens=[["conductivity"]],
        excluded_tokens=["standard", "deviation", "std", "error"],
    )
    if target_col is None:
        raise ValueError(f"Could not identify conductivity target column. Columns: {list(df.columns)}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Target column '{target_col}' contains no usable numeric values.")

    group_col = choose_group_column(df, target_col)
    group = build_group_series(df, group_col)

    feature_source = df.drop(columns=[target_col])
    features = build_safe_numeric_features(feature_source)
    features = features.loc[:, features.notna().mean(axis=0) >= 0.6]
    features = features.loc[:, features.nunique(dropna=True) > 1]
    features = features.fillna(features.median(numeric_only=True))

    if features.empty:
        raise ValueError("No usable feature columns found after cleaning.")

    features = features.astype(float)
    target = df[target_col].astype(float)
    valid = np.isfinite(target.to_numpy()) & np.isfinite(features.to_numpy()).all(axis=1)
    features = features.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)
    group = group.loc[valid].reset_index(drop=True)

    if len(features) < 50:
        raise ValueError(f"Only {len(features)} clean rows remain; benchmark needs at least 50.")

    return PreparedData(features=features, target=target, group=group, target_col=target_col, group_col=group_col)


def build_safe_numeric_features(feature_source: pd.DataFrame) -> pd.DataFrame:
    """Keep formulation and condition features, excluding EIS-derived outputs."""
    safe_exact = {
        "temperature",
        "pc",
        "ec",
        "emc",
        "lipf_6",
        "lipf6",
        "liclo4",
        "litfsi",
        "lifsi",
        "cell_constant,_standard_deviation",
        "cell_constant",
    }
    safe_tokens = ("temperature", "temp", "solvent", "salt", "concentration")
    excluded_tokens = (
        "conductivity",
        "impedance",
        "resistance",
        "arrhenius",
        "fitted",
        "residual",
        "rmse",
        "phase",
        "frequency",
        "circuit",
        "metadata",
        "label",
    )

    features = pd.DataFrame(index=feature_source.index)
    for col in feature_source.columns:
        lowered = str(col).lower()
        if any(token in lowered for token in excluded_tokens):
            continue
        if lowered not in safe_exact and not any(token in lowered for token in safe_tokens):
            continue
        converted = pd.to_numeric(feature_source[col], errors="coerce")
        if converted.notna().mean() >= 0.6:
            features[col] = converted
    return features.replace([np.inf, -np.inf], np.nan)


def find_column(
    df: pd.DataFrame,
    required_tokens: list[list[str]],
    excluded_tokens: Iterable[str] = (),
) -> str | None:
    excluded = tuple(token.lower() for token in excluded_tokens)
    for col in df.columns:
        lowered = str(col).lower()
        if any(token in lowered for token in excluded):
            continue
        if all(any(token in lowered for token in alternatives) for alternatives in required_tokens):
            return col
    return None


def choose_group_column(df: pd.DataFrame, target_col: str) -> str:
    preferred_tokens = ("salt", "solvent", "electrolyte", "formula", "composition")
    candidates: list[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        nunique = series.nunique(dropna=True)
        if not (1 < nunique <= max(50, int(0.25 * len(df)))):
            continue
        lowered = str(col).lower()
        if any(token in lowered for token in preferred_tokens):
            candidates.insert(0, col)
        elif series.dtype == "object":
            candidates.append(col)
    return candidates[0] if candidates else "__temperature_bin__"


def build_group_series(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col != "__temperature_bin__":
        return df[group_col].astype("string").fillna("unknown")

    temp_col = find_column(
        df,
        required_tokens=[["temperature", "temp"]],
        excluded_tokens=["conductivity"],
    )
    if temp_col is not None:
        temp = pd.to_numeric(df[temp_col], errors="coerce")
        if temp.notna().nunique() >= 3:
            return pd.qcut(temp.rank(method="first"), q=5, labels=False, duplicates="drop").astype("string")

    return pd.Series(np.arange(len(df)) % 5, index=df.index, dtype="int64").astype("string")


def run_one_setting(
    prepared: PreparedData,
    sample_size: int,
    missing_rate: float,
    seed: int,
    n_neighbors: int,
    cv_folds: int,
) -> list[dict[str, object]]:
    if sample_size > len(prepared.features):
        raise ValueError(f"sample_size={sample_size} exceeds available rows ({len(prepared.features)}).")

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(prepared.features), size=sample_size, replace=False)
    x_full = prepared.features.iloc[sample_idx].reset_index(drop=True)
    y = prepared.target.iloc[sample_idx].reset_index(drop=True)
    group = prepared.group.iloc[sample_idx].reset_index(drop=True)
    x_missing = inject_mcar_missing(x_full, missing_rate=missing_rate, seed=seed)

    rows: list[dict[str, object]] = []
    for method_name, x_imputed in impute_all_methods(x_missing, group, n_neighbors=n_neighbors):
        if not np.isfinite(x_imputed.to_numpy()).all():
            raise ValueError(f"{method_name} produced NaN or Inf values.")
        rows.extend(
            evaluate_models(
                x_imputed,
                y,
                method=method_name,
                sample_size=sample_size,
                missing_rate=missing_rate,
                seed=seed,
                baseline=False,
                cv_folds=cv_folds,
            )
        )

    rows.extend(
        evaluate_models(
            x_full,
            y,
            method="complete_data_baseline",
            sample_size=sample_size,
            missing_rate=missing_rate,
            seed=seed,
            baseline=True,
            cv_folds=cv_folds,
        )
    )
    return rows


def inject_mcar_missing(df: pd.DataFrame, missing_rate: float, seed: int) -> pd.DataFrame:
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1).")
    rng = np.random.default_rng(seed + 10_000)
    out = df.copy()
    mask = rng.random(out.shape) < missing_rate
    # Keep at least one observed value in every feature for all imputation methods.
    for col_idx in range(mask.shape[1]):
        if mask[:, col_idx].all():
            mask[rng.integers(0, mask.shape[0]), col_idx] = False
    out = out.mask(mask)
    return out


def impute_all_methods(
    x_missing: pd.DataFrame,
    group: pd.Series,
    n_neighbors: int,
) -> Iterable[tuple[str, pd.DataFrame]]:
    feature_cols = list(x_missing.columns)
    safe_neighbors = max(1, min(n_neighbors, len(x_missing)))

    knn = x_missing.copy()
    knn[feature_cols] = KNNImputer(n_neighbors=safe_neighbors).fit_transform(knn[feature_cols])
    yield "knn_sklearn", knn

    iterative = impute_knn_extratrees(
        x_missing,
        feature_cols=feature_cols,
        n_neighbors=safe_neighbors,
        max_iter=5,
        n_estimators=100,
        random_state=42,
    )
    yield "iterative_knn_extratrees", iterative[feature_cols]

    grouped = x_missing.copy()
    grouped["__material_group__"] = group.astype("string").fillna("unknown").to_numpy()
    material = impute_group_median(grouped, feature_cols=feature_cols, group_col="__material_group__")
    yield "material_group_median", material[feature_cols]


def evaluate_models(
    x: pd.DataFrame,
    y: pd.Series,
    method: str,
    sample_size: int,
    missing_rate: float,
    seed: int,
    baseline: bool,
    cv_folds: int,
) -> list[dict[str, object]]:
    n_splits = max(2, min(cv_folds, len(x)))
    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "Ridge": Ridge(alpha=1.0),
    }
    rows: list[dict[str, object]] = []
    for model_name, model in models.items():
        x_model = x
        if model_name == "Ridge":
            x_model = pd.DataFrame(
                StandardScaler().fit_transform(x),
                columns=x.columns,
                index=x.index,
            )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = cross_val_predict(model, x_model, y, cv=cv)
        rows.append(
            {
                "sample_size": sample_size,
                "missing_rate": missing_rate,
                "seed": seed,
                "method": method,
                "model": model_name,
                "is_complete_data_baseline": baseline,
                "MAE": mean_absolute_error(y, pred),
                "RMSE": math.sqrt(mean_squared_error(y, pred)),
                "R2": r2_score(y, pred),
            }
        )
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sample_size", "missing_rate", "method", "model"]
    summary = (
        results.groupby(group_cols, as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .sort_values(["sample_size", "missing_rate", "model", "RMSE_mean"])
    )

    baseline = summary[summary["method"] == "complete_data_baseline"][
        ["sample_size", "missing_rate", "model", "RMSE_mean"]
    ].rename(columns={"RMSE_mean": "complete_RMSE"})
    knn = summary[summary["method"] == "knn_sklearn"][
        ["sample_size", "missing_rate", "model", "RMSE_mean"]
    ].rename(columns={"RMSE_mean": "knn_RMSE"})

    summary = summary.merge(baseline, on=["sample_size", "missing_rate", "model"], how="left")
    summary = summary.merge(knn, on=["sample_size", "missing_rate", "model"], how="left")
    summary["RMSE_loss_vs_complete_%"] = (
        (summary["RMSE_mean"] - summary["complete_RMSE"]) / summary["complete_RMSE"] * 100
    )
    summary["RMSE_improvement_vs_knn_%"] = (
        (summary["knn_RMSE"] - summary["RMSE_mean"]) / summary["knn_RMSE"] * 100
    )
    return summary


def build_markdown_report(results: pd.DataFrame, summary: pd.DataFrame, prepared: PreparedData) -> str:
    main = summary[summary["method"].isin(["knn_sklearn", "iterative_knn_extratrees", "material_group_median"])]
    ranked = main.sort_values(["sample_size", "missing_rate", "model", "RMSE_mean"])
    best_counts = (
        ranked.groupby(["sample_size", "missing_rate", "model"], as_index=False)
        .first()["method"]
        .value_counts()
        .rename_axis("method")
        .reset_index(name="best_count")
    )
    summary_table = dataframe_to_markdown(
        main[
            [
                "sample_size",
                "missing_rate",
                "method",
                "model",
                "RMSE_mean",
                "RMSE_std",
                "R2_mean",
                "RMSE_loss_vs_complete_%",
                "RMSE_improvement_vs_knn_%",
            ]
        ],
        floatfmt=".4f",
    )

    lines = [
        "# Electrolyte Imputation Benchmark",
        "",
        f"- Data source DOI: `{DATA_DOI}`",
        f"- Target column: `{prepared.target_col}`",
        f"- Material group source: `{prepared.group_col}`",
        f"- Clean rows available: `{len(prepared.features)}`",
        f"- Feature count: `{prepared.features.shape[1]}`",
        "",
        "## Methods",
        "",
        "- `knn_sklearn`: standard `sklearn.KNNImputer` baseline.",
        "- `iterative_knn_extratrees`: SmallMatPrep iterative KNN + ExtraTrees imputer.",
        "- `material_group_median`: SmallMatPrep material-aware group median imputer.",
        "",
        "## Best Method Counts",
        "",
        dataframe_to_markdown(best_counts),
        "",
        "## Summary",
        "",
        summary_table,
        "",
        "## Raw Result Shape",
        "",
        f"- Rows: `{len(results)}`",
        f"- Columns: `{len(results.columns)}`",
        "",
    ]
    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = "") -> str:
    """Render a small DataFrame as Markdown without requiring tabulate."""
    if df.empty:
        return "_No rows._"

    rendered = df.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(
                lambda value: format(float(value), floatfmt) if pd.notna(value) and floatfmt else value
            )
    text = rendered.astype(str)
    widths = {
        col: max(len(str(col)), *(len(value) for value in text[col]))
        for col in text.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in text.columns) + " |"
    divider = "| " + " | ".join("-" * widths[col] for col in text.columns) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in text.columns) + " |"
        for _, row in text.iterrows()
    ]
    return "\n".join([header, divider, *body])


if __name__ == "__main__":
    main()
