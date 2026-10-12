"""Train + compare regressors (FR-04, FR-06). Run: python -m src.train"""
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from .config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .data_loader import current_year, load_clean
from .preprocessing import build_model_pipeline

CANDIDATES = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=120, max_depth=20, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

PREDICT_COLUMNS = ["car_age", "km_driven", "brand", "fuel",
                   "seller_type", "transmission", "owner"]


def metrics(y_true, y_pred) -> dict:
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> dict:
    ref_year = current_year()
    df = load_clean(DATA_PATH, ref_year=ref_year)
    X = df[PREDICT_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Baseline: median of training target (PRD success criteria).
    baseline_pred = pd.Series(y_train.median()).repeat(len(y_test))
    results = {"baseline_median": metrics(y_test, baseline_pred)}
    print(f"Baseline (median={y_train.median():,.0f}): {results['baseline_median']}")

    fitted = {}
    for name, reg in CANDIDATES.items():
        pipe = build_model_pipeline(reg)
        pipe.fit(X_train, y_train)
        m = metrics(y_test, pipe.predict(X_test))
        results[name] = m
        fitted[name] = pipe
        print(f"{name}: MAE={m['mae']:,.0f} RMSE={m['rmse']:,.0f} R2={m['r2']:.4f}")

    # Select best by R2.
    best_name = max(
        (n for n in fitted), key=lambda n: results[n]["r2"]
    )
    print(f"Best model: {best_name}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": fitted[best_name], "model_name": best_name,
         "ref_year": ref_year, "feature_columns": PREDICT_COLUMNS},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(
        {"model": best_name, "ref_year": ref_year,
         "n_train": len(X_train), "n_test": len(X_test), "results": results},
        indent=2))
    print(f"Saved pipeline -> {MODEL_PATH}")
    print(f"Saved metrics  -> {METRICS_PATH}")
    return results


if __name__ == "__main__":
    main()
