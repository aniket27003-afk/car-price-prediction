"""Central configuration for AutoValue AI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cardekho_used_cars.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

TARGET = "selling_price"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Reference year for car_age = current_year - year.
# Computed at runtime so future predictions age correctly.
# Dataset spans 1992-2020.
MIN_YEAR = 1992
MAX_YEAR = 2026

NUMERIC_FEATURES = ["car_age", "km_driven"]
CATEGORICAL_FEATURES = ["brand", "fuel", "seller_type", "transmission", "owner"]

# Allowed categories (from training data inspection). Unknown values are
# handled gracefully by OneHotEncoder(handle_unknown="ignore") and by
# input validation in the UI.
ALLOWED_FUELS = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
ALLOWED_SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
ALLOWED_TRANSMISSIONS = ["Manual", "Automatic"]
ALLOWED_OWNERS = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]
