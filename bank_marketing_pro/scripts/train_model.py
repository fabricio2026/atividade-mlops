import io
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from app.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERICAL_FEATURES,
)

_UCI_ZIP = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
_ZIP_ENTRY = "bank-additional/bank-additional-full.csv"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace(".", "_") for c in df.columns]
    return df


def _load_via_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=222)
    X = _normalize_cols(dataset.data.features.copy())
    y = (dataset.data.targets.squeeze() == "yes").astype(int)
    missing = [f for f in ALL_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return X[ALL_FEATURES], y


def _load_via_direct_download():
    print(f"  Fetching {_UCI_ZIP} ...")
    with urllib.request.urlopen(_UCI_ZIP) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            with zf.open(_ZIP_ENTRY) as f:
                df = pd.read_csv(f, sep=";")
    df = _normalize_cols(df)
    X = df[ALL_FEATURES]
    y = (df["y"] == "yes").astype(int)
    return X, y


def _load_data():
    print("Downloading Bank Marketing dataset (UCI id=222)...")
    try:
        X, y = _load_via_ucimlrepo()
        print(f"  Loaded via ucimlrepo. Shape: {X.shape}")
        return X, y
    except Exception as e:
        print(f"  ucimlrepo returned incomplete data ({e}). Falling back to direct download...")
        X, y = _load_via_direct_download()
        print(f"  Loaded via direct download. Shape: {X.shape}")
        return X, y


def main():
    X, y = _load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERICAL_FEATURES),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    print("Training RandomForestClassifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"Metrics: {metrics}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "metadata": {"features": ALL_FEATURES, "metrics": metrics}},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
