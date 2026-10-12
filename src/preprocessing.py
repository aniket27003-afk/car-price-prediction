"""Preprocessing pipeline (FR-02). Fit only on training data."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = StandardScaler()
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline(regressor) -> Pipeline:
    """Combine preprocessing + regressor so training and prediction
    always use identical transformations (no leakage, consistent UI)."""
    return Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", regressor)]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return output feature names after fitting (for importance plots)."""
    names = list(NUMERIC_FEATURES)
    cat = preprocessor.named_transformers_["cat"]
    try:
        names += list(cat.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        pass
    return names
