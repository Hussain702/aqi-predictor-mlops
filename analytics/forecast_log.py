"""
Ground-Truth Forecast Validation
-------------------------------------
Your training metrics (RMSE/MAE/R2) are computed on a held-out TEST SPLIT
of historical data -- a real, valid measure of accuracy, but not the same
as watching the live model predict the actual future and checking later
whether it was right.

This module closes that loop:
    1. log_todays_forecasts() -- run daily. Predicts +24h/+48h/+72h for
       all 4 cities using whichever model is CURRENTLY registered as best,
       and logs each prediction (with which model made it) to a separate
       Hopsworks feature group: forecast_log.
    2. compute_accuracy() -- once real time has passed and those target
       dates have actually happened, joins forecast_log against the real
       aqi_features history and computes the actual error.

Because model_name is logged with every prediction, if a future daily
retrain ever picks Ridge or the neural network instead of Random Forest,
that model's real-world accuracy shows up here too -- not just Random
Forest's.

Run:
    python -m analytics.forecast_log                # log today's forecasts
    python -m analytics.forecast_log --accuracy      # show accuracy so far
"""

import sys
from datetime import timedelta

import pandas as pd

from utils.hopsworks_client import get_feature_store, get_project
from training.train import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, HORIZON_HOURS
from serving.predict import load_latest_model, predict_for_city

FORECAST_LOG_NAME = "forecast_log"
FORECAST_LOG_VERSION = 1
CITIES = ["lahore", "islamabad", "karachi", "peshawar"]


def get_or_create_forecast_log(fs):
    return fs.get_or_create_feature_group(
        name=FORECAST_LOG_NAME,
        version=FORECAST_LOG_VERSION,
        description="Logged +24h/+48h/+72h AQI forecasts, for comparison against actual outcomes",
        primary_key=["city", "horizon"],
        event_time="forecast_made_at",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def log_todays_forecasts():
    """
    Predicts +24h/+48h/+72h for all 4 cities using the CURRENT best
    registered model, and logs each prediction (3 horizons x 4 cities =
    up to 12 rows) to forecast_log, tagged with which model made it.
    """
    project = get_project()
    fs = project.get_feature_store()

    model, feature_names, model_name, framework, model_version = load_latest_model(project)

    feature_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = feature_fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest_per_city = df.sort_values("timestamp").groupby("city").tail(1)

    rows = []
    for city in CITIES:
        city_rows = latest_per_city[latest_per_city["city"] == city]
        if city_rows.empty:
            print(f"Skipping {city}: no data yet.")
            continue

        values, as_of, _ = predict_for_city(city_rows.iloc[0], model, feature_names)
        # values = [today_actual, pred_24h, pred_48h, pred_72h] -- only log
        # the actual FUTURE predictions, not today's (which isn't a forecast)
        for horizon, predicted_aqi in zip(["24h", "48h", "72h"], values[1:]):
            rows.append({
                "city": city,
                "forecast_made_at": as_of,
                "target_date": as_of + timedelta(hours=HORIZON_HOURS[horizon]),
                "horizon": horizon,
                "predicted_aqi": float(predicted_aqi),
                "model_name": model_name,
                "model_version": int(model_version),
            })

    if not rows:
        print("Nothing to log -- no city has data yet.")
        return

    log_df = pd.DataFrame(rows)
    log_df["forecast_made_at"] = pd.to_datetime(log_df["forecast_made_at"])
    log_df["target_date"] = pd.to_datetime(log_df["target_date"])

    forecast_log = get_or_create_forecast_log(fs)
    forecast_log.insert(log_df)
    print(f"Logged {len(log_df)} forecasts ({len(CITIES)} cities x 3 horizons) "
          f"from model '{model_name}' v{model_version}.")


def compute_accuracy(tolerance_hours: int = 2) -> pd.DataFrame:
    """
    For every logged prediction whose target_date has now actually
    happened, finds the closest real reading (within `tolerance_hours`)
    and computes the error. Returns one row per matched prediction --
    empty if nothing has come due yet (check back in a day or two).
    """
    fs = get_feature_store()

    forecast_fg = fs.get_feature_group(name=FORECAST_LOG_NAME, version=FORECAST_LOG_VERSION)
    forecast_df = forecast_fg.read()
    forecast_df["target_date"] = pd.to_datetime(forecast_df["target_date"])

    actual_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    actual_df = actual_fg.read()[["city", "timestamp", "aqi"]]
    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"])

    matched = []
    for city in forecast_df["city"].unique():
        city_forecasts = forecast_df[forecast_df["city"] == city].sort_values("target_date")
        city_actuals = actual_df[actual_df["city"] == city].sort_values("timestamp")
        if city_actuals.empty:
            continue

        merged = pd.merge_asof(
            city_forecasts, city_actuals,
            left_on="target_date", right_on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        )
        matched.append(merged.dropna(subset=["aqi"]))

    if not matched:
        return pd.DataFrame()

    result = pd.concat(matched, ignore_index=True)
    result["error"] = result["predicted_aqi"] - result["aqi"]
    result["abs_error"] = result["error"].abs()
    return result


if __name__ == "__main__":
    if "--accuracy" in sys.argv:
        accuracy_df = compute_accuracy()
        if accuracy_df.empty:
            print("No matched predictions yet -- check back in a day or two "
                  "once some logged forecasts' target dates have actually passed.")
        else:
            summary = (
                accuracy_df.groupby(["horizon", "model_name"])["abs_error"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "MAE", "count": "n"})
            )
            print(summary)
    else:
        log_todays_forecasts()