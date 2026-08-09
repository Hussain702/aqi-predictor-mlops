"""
Shared model-loading + prediction helpers.

Used by BOTH dashboard/streamlit_app.py and analytics/forecast_log.py, so
they always load the same registered model and predict the same way --
if this logic lived in two places, the dashboard and the ground-truth
accuracy log could silently drift apart (e.g. one picking up a new model
version before the other).
"""

import os

import joblib
import numpy as np
import pandas as pd

from utils.hopsworks_client import get_project
from training.registry import MODEL_NAME
from training.train import FEATURE_COLUMNS, CATEGORICAL_COLUMN


def load_latest_model(project=None):
    """
    Loads the best registered model (by rmse_24h, falling back to the
    highest version number if get_best_model isn't available).

    Returns: (model, feature_names, model_name, framework, model_version)
    model.predict(X) works the same regardless of framework -- sklearn
    models are used directly; a Keras model gets wrapped so its .predict
    has the same signature.
    """
    project = project or get_project()

    mr = project.get_model_registry()
    try:
        model_meta = mr.get_best_model(MODEL_NAME, "rmse_24h", "min")
    except Exception:
        models = mr.get_models(MODEL_NAME)
        if not models:
            raise RuntimeError(
                f"No registered model named '{MODEL_NAME}' found. "
                "Run `python -m training.registry` first."
            )
        model_meta = max(models, key=lambda m: m.version)

    model_dir = model_meta.download()

    framework_path = os.path.join(model_dir, "framework.txt")
    framework = open(framework_path).read().strip() if os.path.exists(framework_path) else "sklearn"

    model_name_path = os.path.join(model_dir, "model_name.txt")
    model_name = (
        open(model_name_path).read().strip() if os.path.exists(model_name_path) else "unknown"
    )

    if framework == "tensorflow":
        import tensorflow as tf

        keras_model = tf.keras.models.load_model(os.path.join(model_dir, "keras_model.keras"))

        class _LoadedKerasModel:
            """Matches the .predict(X) interface sklearn models expose."""
            def predict(self, X):
                X_arr = X.to_numpy(dtype="float32") if hasattr(X, "to_numpy") else X
                return keras_model.predict(X_arr, verbose=0)

        model = _LoadedKerasModel()
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))

    with open(os.path.join(model_dir, "feature_names.txt")) as f:
        feature_names = f.read().splitlines()

    return model, feature_names, model_name, framework, model_meta.version


def build_feature_row(city_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """Build a single-row DataFrame matching the model's expected column order."""
    row = {col: city_row[col] for col in FEATURE_COLUMNS}
    for name in feature_names:
        if name.startswith("city_"):
            row[name] = 1 if name == f"city_{city_row[CATEGORICAL_COLUMN]}" else 0
    return pd.DataFrame([row])[feature_names]


def predict_for_city(city_row: pd.Series, model, feature_names: list):
    """Returns (values=[today, +24h, +48h, +72h], as_of timestamp, X_row used for the prediction)."""
    X = build_feature_row(city_row, feature_names)
    predictions = np.asarray(model.predict(X)).flatten()
    values = [float(city_row["aqi"])] + list(predictions)
    return values, city_row["timestamp"], X