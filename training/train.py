"""
Step 5 - Train ML models (multi-horizon forecasting)
--------------------------------------------------------
Fetches ALL historical rows from the 'aqi_features' Hopsworks feature group
and trains models to forecast AQI 24, 48, and 72 hours ahead -- all from
ONE call using only what's knowable TODAY (no future weather needed).

How the 3-day forecast actually works (the design gap flagged earlier,
now resolved):
    We use "direct multi-horizon forecasting": for each historical hourly
    row (at time t), we build THREE targets by looking at the SAME city's
    future rows already sitting in Hopsworks:
        aqi_target_24h = that city's aqi value 24 hours after t
        aqi_target_48h = ... 48 hours after t
        aqi_target_72h = ... 72 hours after t
    The model learns: "given today's temperature/humidity/pm2.5/pm10/current
    aqi/rolling average, what will aqi be in 1/2/3 days?" No future weather
    data is needed at prediction time -- only today's readings, which the
    live hourly pipeline already provides. Both Random Forest and Ridge
    support multi-output regression natively (no extra wrapper needed).

Models used for now: Random Forest, Ridge Regression (both fast, no extra
heavy installs). A TensorFlow/Keras model can be added as a follow-up.

Run standalone (just trains + prints, no evaluation/registration):
    python -m training.train
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from utils.hopsworks_client import get_feature_store

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pm25", "pm10",
    "hour", "day", "month", "is_weekend",
    "aqi_yesterday", "aqi_change_rate", "rolling_average_aqi",
]
CATEGORICAL_COLUMN = "city"

TARGET_HORIZONS = ["24h", "48h", "72h"]
TARGET_COLUMNS = [f"aqi_target_{h}" for h in TARGET_HORIZONS]
HORIZON_HOURS = {"24h": 24, "48h": 48, "72h": 72}


def _to_float_array(data) -> np.ndarray:
    """
    Force a clean float32 numpy array, regardless of input dtype quirks.

    Hopsworks reads data back via its Arrow-based Feature Query Service,
    which often returns columns as pandas' NULLABLE extension dtypes
    (capital-letter "Int64", "boolean") instead of plain numpy dtypes.
    A DataFrame mixing extension dtypes with regular numpy columns turns
    into dtype=object on a plain `.values`/`.to_numpy()` call -- sklearn
    tolerates that (it coerces internally), but TensorFlow does not and
    raises "ValueError: Invalid dtype: object". .to_numpy(dtype="float32")
    forces the real numeric cast instead of leaving it to guess.
    """
    if hasattr(data, "to_numpy"):
        return data.to_numpy(dtype="float32")
    return np.asarray(data, dtype="float32")


class KerasMultiOutputRegressor:
    """
    Thin wrapper around a small Keras Sequential model so it exposes the
    same .fit(X, y) / .predict(X) interface as our sklearn models. That's
    what lets it sit in get_models() / train_all() / evaluate.py /
    registry.py without any of those needing special-case code for "the
    neural network one" -- they just call .fit and .predict like normal.

    The `framework` class attribute is how registry.py knows to save this
    with model.save() (TensorFlow's format) instead of joblib.dump()
    (which doesn't work for Keras models).
    """
    framework = "tensorflow"

    def __init__(self, input_dim: int, output_dim: int, epochs=30, batch_size=64, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(output_dim),  # linear output: one value per horizon
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def fit(self, X, y):
        X_arr = _to_float_array(X)
        y_arr = _to_float_array(y)
        self.model.fit(
            X_arr, y_arr,
            epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
        )
        return self

    def predict(self, X):
        X_arr = _to_float_array(X)
        return self.model.predict(X_arr, verbose=0)


def load_training_data() -> pd.DataFrame:
    """Pull the full history from Hopsworks as one dataframe."""
    fs = get_feature_store()
    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
    )
    return feature_group.read()


def build_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aqi_target_24h/48h/72h columns by looking ahead WITHIN each city's
    own timeline (grouping by city keeps Lahore's future from leaking into
    Karachi's rows, etc). Assumes roughly-hourly rows (true for both the
    backfill and the live hourly pipeline), so shifting by N rows ~= N hours.
    """
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    grouped_aqi = df.groupby("city")["aqi"]

    for horizon, hours in HORIZON_HOURS.items():
        df[f"aqi_target_{horizon}"] = grouped_aqi.shift(-hours)

    return df


def prepare_features(df: pd.DataFrame):
    """
    Build horizon targets, one-hot encode city, drop rows where we don't
    have all 3 future targets yet (e.g. the last 3 days of data, or gaps),
    and split into X (today's features) / Y (3-column future AQI targets).
    """
    df = build_horizon_targets(df)

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    df = df.dropna(subset=required_columns).copy()

    city_dummies = pd.get_dummies(df[CATEGORICAL_COLUMN], prefix="city")
    X = pd.concat([df[FEATURE_COLUMNS], city_dummies], axis=1)
    Y = df[TARGET_COLUMNS]

    return X, Y


def get_models(input_dim: int = None, output_dim: int = None) -> dict:
    """
    Candidate models spanning statistical -> classical ML -> deep learning:
      - ridge_regression: linear/statistical baseline
      - random_forest: classical ML ensemble (constrained -- see note below)
      - neural_network: small deep learning model (only added if TensorFlow
        is installed AND input_dim/output_dim are known, since the network's
        shape depends on the data)

    RandomForestRegressor is constrained (max_depth, min_samples_leaf,
    n_estimators) -- with no limits, 200 fully-grown trees produced a
    ~600MB pickle that failed to upload. 100 trees got it to ~34MB, still
    too slow/fragile over a poor connection. Halved again to 50 trees.
    """
    models = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if TENSORFLOW_AVAILABLE and input_dim and output_dim:
        models["neural_network"] = KerasMultiOutputRegressor(input_dim, output_dim)
    elif not TENSORFLOW_AVAILABLE:
        print(
            "TensorFlow not installed -- skipping the neural_network candidate. "
            "Run `uv add tensorflow` to include it."
        )

    return models


def train_all(X_train, Y_train) -> dict:
    """Fit every candidate model on all 3 horizons at once. Returns {model_name: fitted_model}."""
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1] if hasattr(Y_train, "shape") else len(TARGET_COLUMNS)

    fitted = {}
    for name, model in get_models(input_dim, output_dim).items():
        print(f"Training {name}...")
        model.fit(X_train, Y_train)
        fitted[name] = model
    return fitted


if __name__ == "__main__":
    df = load_training_data()
    print(f"Loaded {len(df)} historical rows from Hopsworks.")

    X, Y = prepare_features(df)
    print(f"After building horizon targets and dropping incomplete rows: {len(X)} rows.")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    fitted_models = train_all(X_train, Y_train)
    print(f"\nTrained models: {list(fitted_models.keys())}")
    print("Run 'python -m training.evaluate' next to score them.")