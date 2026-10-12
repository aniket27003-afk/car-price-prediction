"""Dataset loading and cleaning (FR-02)."""
from datetime import datetime

import pandas as pd

from .config import MAX_YEAR, MIN_YEAR, TARGET


def current_year() -> int:
    return datetime.now().year


def extract_brand(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    return name.strip().split()[0]


def load_raw(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names to lower-case for robustness across mirrors.
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "km driven": "km_driven",
        "kms_driven": "km_driven",
        "selling price": "selling_price",
        "seller type": "seller_type",
    }
    df = df.rename(columns=rename)
    return df


def clean(df: pd.DataFrame, ref_year: int | None = None) -> pd.DataFrame:
    """Drop invalid records, duplicates; derive brand + car_age."""
    df = df.copy()
    ref_year = ref_year or current_year()

    required = ["name", "year", "selling_price", "km_driven",
                "fuel", "seller_type", "transmission", "owner"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Drop rows with missing values in required columns.
    df = df.dropna(subset=required)

    # Coerce numerics; invalid parsing -> NaN -> dropped.
    for col in ["year", "selling_price", "km_driven"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["year", "selling_price", "km_driven"])

    # Handle invalid records.
    df = df[(df["selling_price"] > 0) & (df["km_driven"] >= 0)]
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    # Exact duplicates (this mirror contains ~763) are removed.
    df = df.drop_duplicates().reset_index(drop=True)

    # Feature engineering: brand from free-text name, car age.
    df["brand"] = df["name"].apply(extract_brand)
    df["car_age"] = ref_year - df["year"]
    df.loc[df["car_age"] < 0, "car_age"] = 0

    # Strip whitespace on categoricals.
    for col in ["brand", "fuel", "seller_type", "transmission", "owner"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def load_clean(csv_path, ref_year: int | None = None) -> pd.DataFrame:
    return clean(load_raw(csv_path), ref_year=ref_year)
