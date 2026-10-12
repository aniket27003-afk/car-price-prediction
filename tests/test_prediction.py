"""Basic application tests (PRD section 12). Run: pytest -q"""
import joblib
import pandas as pd
import pytest

from src.config import MODEL_PATH
from src.data_loader import extract_brand
from src.predict import predict_price, validate_inputs


def test_valid_inputs_pass():
    assert validate_inputs(2015, 50_000, "Petrol", "Individual", "Manual",
                           "First Owner") == []


def test_missing_or_invalid_numeric():
    errs = validate_inputs(2015, -5, "Petrol", "Individual", "Manual", "First Owner")
    assert any("Kilometers" in e for e in errs)


def test_unsupported_category():
    errs = validate_inputs(2015, 10_000, "Hydrogen", "Individual", "Manual",
                           "First Owner")
    assert any("fuel" in e.lower() for e in errs)


def test_year_out_of_range():
    assert validate_inputs(1800, 10_000, "Petrol", "Individual",
                           "Manual", "First Owner")
    assert validate_inputs(2035, 10_000, "Petrol", "Individual",
                           "Manual", "First Owner")


def test_extreme_kilometres_rejected():
    assert validate_inputs(2015, 5_000_000, "Petrol", "Individual",
                           "Manual", "First Owner")


def test_brand_extraction():
    assert extract_brand("Hyundai Creta 1.6 SX") == "Hyundai"
    assert extract_brand("") == "Unknown"


def test_model_file_loads():
    assert MODEL_PATH.exists(), "Run python -m src.train first"
    art = joblib.load(MODEL_PATH)
    assert "pipeline" in art


def test_prediction_consistency():
    art = joblib.load(MODEL_PATH)
    p1 = predict_price("Maruti Swift VDI", 2015, 60_000, "Diesel",
                       "Individual", "Manual", "First Owner", art)
    p2 = predict_price("Maruti Swift VDI", 2015, 60_000, "Diesel",
                       "Individual", "Manual", "First Owner", art)
    assert p1 == pytest.approx(p2)
    assert p1 > 0


def test_unknown_brand_does_not_crash():
    art = joblib.load(MODEL_PATH)
    price = predict_price("Unicorn X999", 2018, 10_000, "Petrol",
                          "Dealer", "Manual", "First Owner", art)
    assert price > 0
