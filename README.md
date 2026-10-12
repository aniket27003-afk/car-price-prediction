# AutoValue AI — Smart Car Price Predictor

ML-powered Streamlit app that estimates used-car selling prices (₹) from CarDekho listings.

## How this project works (step by step)

1. **Dataset** — `data/cardekho_used_cars.csv` holds 4,340 historical listings
   (name, year, selling_price, km_driven, fuel, seller_type, transmission, owner).
   See `data/DATA_DICTIONARY.md` for source and column meanings.
2. **Loading & cleaning** (`src/data_loader.py`) — normalises columns, drops rows
   with missing/invalid values (bad year, km, or price), keeps years 1992–2026,
   removes ~763 exact duplicates (~3,577 usable rows), and derives two features:
   `brand` (first word of `name`, shrinking 1,491 variants to ~30 brands) and
   `car_age` (reference year minus manufacturing year).
3. **Preprocessing** (`src/preprocessing.py`) — a `ColumnTransformer` scales the
   numeric features (`car_age`, `km_driven`) with `StandardScaler` and one-hot
   encodes the categoricals (`brand`, `fuel`, `seller_type`, `transmission`,
   `owner`) with `handle_unknown="ignore"`, so unseen brands never crash
   predictions. Preprocessing lives *inside* each model pipeline, so it is fit
   on training data only (no leakage) and reused identically at prediction time.
4. **Training & comparison** (`src/train.py`, run via `python -m src.train`) —
   splits data 80/20 (`random_state=42`), trains four regressors (Linear
   Regression, Decision Tree, Random Forest, Gradient Boosting), and scores each
   with MAE, RMSE, and R² on the held-out test set, alongside a median-price
   baseline. The best model by R² is saved.
5. **Saving** — the winning pipeline (preprocessing + model), its name,
   reference year, and feature list are stored with `joblib` at
   `models/car_price_pipeline.joblib`; all metrics go to `models/metrics.json`.
6. **Prediction** (`src/predict.py`) — validates inputs (year range, km range,
   allowed categories), rebuilds the `car_age`/brand row, runs it through the
   saved pipeline, and returns a non-negative price. Tree models additionally
   expose `feature_importances_` for explanation.
7. **Web app** (`app.py`, run via `streamlit run app.py`) — loads the model once,
   then offers three pages: Home (what the tool does + disclaimer), Predict
   Price (validated form → prominent ₹ estimate + top associated factors), and
   Model Insights (dataset overview, price-distribution / fuel / transmission /
   year charts, correlation table, model-comparison metrics, feature importance).
8. **Verification** — `tests/test_prediction.py` (`pytest -q`, 9 tests) covers
   valid/invalid inputs, unsupported categories, out-of-range years, extreme km
   readings, model-file loading, prediction consistency, and unknown brands;
   `notebooks/01_eda.ipynb` reproduces the exploratory charts.

Current result: **Random Forest** (120 trees, max_depth=20) — MAE 156,616,
RMSE 360,048, R² 0.60 on 716 test rows, clearly beating the median baseline
(MAE 283,305, R² −0.05).

## Skills used in this project

| Skill | Where it is applied |
|---|---|
| Python (3.14) | Entire codebase: data, training, app, tests |
| pandas / NumPy | CSV loading, cleaning, duplicates, aggregates, EDA |
| Data cleaning | Missing values, invalid records, outlier inspection, dedup |
| Feature engineering | `brand` extraction, `car_age`, categorical handling |
| scikit-learn | `ColumnTransformer`, `Pipeline`, 4 regressors, train/test split |
| Model evaluation | MAE, RMSE, R², median-baseline comparison, best-model selection |
| Leakage prevention | Preprocessing inside pipeline, fit-on-train-only, no target-derived features |
| Model persistence (joblib) | Saving/loading preprocessing + model as one artefact |
| Streamlit | 3-page UI, form validation, ₹ display, cached model/data, charts |
| Matplotlib / Seaborn | Price distribution, year/fuel/transmission trends, correlations |
| pytest | 9 input-validation, loading, and consistency tests |
| Git-ready packaging | `requirements.txt`, `.gitignore`, README, data dictionary, modular `src/` |

## Setup

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python -m src.train
```

## Run app

```powershell
streamlit run app.py
```

## Test

```powershell
pytest -q
```

## Project layout

- `data/cardekho_used_cars.csv` + `DATA_DICTIONARY.md`
- `src/` — `config`, `data_loader`, `preprocessing`, `train`, `predict`
- `app.py` — Home / Predict / Insights
- `notebooks/01_eda.ipynb` — exploratory analysis
- `models/` — saved pipeline + metrics
- `tests/` — validation and consistency tests

> Estimates are indicative only; actual resale depends on vehicle condition, location, and market demand.
