import joblib
import pandas as pd
from pathlib import Path

from app.config import MODEL_PATH, ALL_FEATURES

_model = None
_metadata = None


def load_model() -> bool:
    global _model, _metadata
    path = Path(MODEL_PATH)
    if not path.exists():
        return False
    artifact = joblib.load(path)
    _model = artifact["pipeline"]
    _metadata = artifact["metadata"]
    return True


def is_loaded() -> bool:
    return _model is not None


def get_metadata() -> dict:
    return _metadata or {}


def predict(features: dict) -> tuple[int, float]:
    df = pd.DataFrame([features])[ALL_FEATURES]
    prediction = int(_model.predict(df)[0])
    probability = float(_model.predict_proba(df)[0][1])
    return prediction, probability
