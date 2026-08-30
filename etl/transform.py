

import pandas as pd
from datetime import datetime


def add_time_features(record: dict) -> dict:
    """Add hour, day, month, is_weekend from the record's timestamp."""
    ts = pd.to_datetime(record["timestamp"])

    record["hour"] = ts.hour
    record["day"] = ts.day
    record["month"] = ts.month
    record["is_weekend"] = ts.dayofweek >= 5  # Saturday=5, Sunday=6

    return record


def add_history_features(record: dict, history_df: pd.DataFrame = None) -> dict:
    """
    Add aqi_yesterday, aqi_change_rate, rolling_average_aqi using past records.

    history_df: a DataFrame of previous records (must have 'timestamp' and 'aqi'
                columns). Pass None or an empty DataFrame if you don't have
                history yet (e.g. first run).
    """
    if history_df is None or history_df.empty:
        record["aqi_yesterday"] = float("nan")
        record["aqi_change_rate"] = float("nan")
        record["rolling_average_aqi"] = float("nan")
        return record

    history_df = history_df.copy()
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.sort_values("timestamp")

    current_time = pd.to_datetime(record["timestamp"])

    # AQI ~24 hours ago (closest record to exactly 1 day back)
    one_day_ago = current_time - pd.Timedelta(hours=24)
    past_day = history_df[history_df["timestamp"] <= one_day_ago]
    aqi_yesterday = past_day.iloc[-1]["aqi"] if not past_day.empty else float("nan")

    # Rolling average of last 24 records (roughly last 24 hours if hourly)
    last_24 = history_df.tail(24)
    rolling_avg = last_24["aqi"].mean() if not last_24.empty else float("nan")

    # Change rate vs the most recent previous record
    last_record = history_df.iloc[-1]
    prev_aqi = last_record["aqi"]
    change_rate = (record["aqi"] - prev_aqi) / prev_aqi if prev_aqi else float("nan")

    record["aqi_yesterday"] = aqi_yesterday
    record["aqi_change_rate"] = change_rate
    record["rolling_average_aqi"] = rolling_avg

    return record


def handle_missing_values(record: dict) -> dict:
    """
    Simple missing-value handling for beginner stage.
    pm10/pm25 sometimes come back as None from AQICN -- fill with 0 for now.
    (We can swap this for a smarter strategy, e.g. rolling mean, later.)
    """
    for key in ["pm25", "pm10"]:
        if record.get(key) is None:
            record[key] = 0
    return record


def transform(record: dict, history_df: pd.DataFrame = None) -> dict:
    """Run the full transform: missing values -> time features -> history features."""
    record = handle_missing_values(record)
    record = add_time_features(record)
    record = add_history_features(record, history_df)
    return record



