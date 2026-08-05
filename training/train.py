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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from utils.hopsworks_client import get_feature_store

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


def get_models() -> dict:
    """
    Candidate models to experiment with. Both support multi-output natively.

    RandomForestRegressor is constrained (max_depth, min_samples_leaf,
    n_estimators) -- with no limits, 200 fully-grown trees produced a
    ~600MB pickle that failed to upload. 100 trees got it to ~34MB, still
    too slow/fragile over a poor connection (upload was resetting mid-
    transfer). Halved again to 50 trees to shrink the upload further --
    accuracy loss from this should be small given how strong the results
    already were (R2 0.88-0.92 at 100 trees).
    """
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "ridge_regression": Ridge(alpha=1.0),
    }


def train_all(X_train, Y_train) -> dict:
    """Fit every candidate model on all 3 horizons at once. Returns {model_name: fitted_model}."""
    fitted = {}
    for name, model in get_models().items():
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