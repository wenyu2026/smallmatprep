"""Minimal demo for SmallMatPrep."""
from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import smallmatprep as smp

EXAMPLE_DIR = ROOT / "examples"


def main():
    cfg = smp.load_config(EXAMPLE_DIR / "config_template.json")
    df = smp.load_csv(EXAMPLE_DIR / "sample_electrolyte.csv", id_column=cfg.get("id_column"))

    print("== Missing report ==")
    miss = smp.missing_report(df)
    print(miss)

    feature_cols = [
        c for c in df.columns
        if c != cfg["target"]
    ]

    df_imp = smp.impute_median(df, feature_cols)
    print("\n== After median imputation ==")
    print(df_imp.head())

    train = df_imp.dropna(subset=[cfg["target"]])
    X = train[feature_cols]
    y = train[cfg["target"]]

    rec = smp.recommend_model(n_samples=len(X), n_features=X.shape[1])
    print("\n== Model recommendation ==")
    for r in rec:
        print(r)

    model = RandomForestRegressor(**rec[0]["params"])
    model.fit(X, y)
    pred = model.predict(X)
    metrics = smp.evaluate_model(y, pred)
    print("\n== Evaluation ==")
    print(metrics)

    summary = smp.build_summary(miss, metrics, rec[0]["model"])
    print("\n", summary)


if __name__ == "__main__":
    main()
