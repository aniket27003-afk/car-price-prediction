"""Inference helper used by the Streamlit app and tests (FR-05)."""
import joblib
import pandas as pd

from .config import (
    ALLOWED_FUELS,
    ALLOWED_OWNERS,
    ALLOWED_SELLER_TYPES,
    ALLOWED_TRANSMISSIONS,
    MAX_YEAR,
    MIN_YEAR,
    MODEL_PATH,
)
from .data_loader import current_year, extract_brand

_ARTIFACT = None


def load_artifact(path=MODEL_PATH):
    global _ARTIFACT
    if _ARTIFACT is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found at {path}. Run: python -m src.train"
            )
        try:
            _ARTIFACT = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Model file is corrupted: {exc}") from exc
    return _ARTIFACT


def validate_inputs(year: int, km_driven: float, fuel: str,
                    seller_type: str, transmission: str, owner: str) -> list:
    errors = []
    if not (MIN_YEAR <= int(year) <= MAX_YEAR):
        errors.append(f"Year must be between {MIN_YEAR} and {MAX_YEAR}.")
    if km_driven is None or float(km_driven) < 0 or float(km_driven) > 2_000_000:
        errors.append("Kilometers driven must be between 0 and 2,000,000.")
    if fuel not in ALLOWED_FUELS:
        errors.append(f"Unsupported fuel type: {fuel}.")
    if seller_type not in ALLOWED_SELLER_TYPES:
        errors.append(f"Unsupported seller type: {seller_type}.")
    if transmission not in ALLOWED_TRANSMISSIONS:
        errors.append(f"Unsupported transmission: {transmission}.")
    if owner not in ALLOWED_OWNERS:
        errors.append(f"Unsupported owner category: {owner}.")
    return errors


def predict_price(car_name: str, year: int, km_driven: float, fuel: str,
                  seller_type: str, transmission: str, owner: str,
                  artifact=None) -> float:
    artifact = artifact or load_artifact()
    pipe = artifact["pipeline"]
    ref_year = artifact.get("ref_year", current_year())
    brand = extract_brand(car_name)
    row = pd.DataFrame([{
        "car_age": max(ref_year - int(year), 0),
        "km_driven": float(km_driven),
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }])
    price = float(pipe.predict(row)[0])
    return max(price, 0.0)


def feature_importance(artifact=None):
    """Return (names, importances) for tree models, else (None, None)."""
    from .preprocessing import get_feature_names
    artifact = artifact or load_artifact()
    pipe = artifact["pipeline"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None, None
    names = get_feature_names(pipe.named_steps["preprocess"])
    return names, model.feature_importances_
