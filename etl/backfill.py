

import time
import numpy as np
import pandas as pd
import requests

from utils.hopsworks_client import get_feature_store

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

START_DATE = "2025-12-31"
END_DATE = "2026-07-31"

# lat, lon for each city
CITIES = {
    "lahore": (31.5204, 74.3587),
    "islamabad": (33.6844, 73.0479),
    "karachi": (24.8607, 67.0011),
    "peshawar": (34.0151, 71.5249),
}

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    """Real historical temperature/humidity/wind for the date range (ERA5)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    return pd.DataFrame({
        "timestamp": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
    })


def fetch_air_quality(lat: float, lon: float) -> pd.DataFrame:
    """Real historical PM2.5, PM10, US AQI for the date range (CAMS)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "pm10,pm2_5,us_aqi",
        "timezone": "Asia/Karachi",
    }
    response = requests.get(AIR_QUALITY_URL, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    return pd.DataFrame({
        "timestamp": hourly["time"],
        "pm10": hourly["pm10"],
        "pm25": hourly["pm2_5"],
        "aqi": hourly["us_aqi"],
    })


def build_city_dataframe(city: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch + merge + engineer all 15 feature columns for one city, one year."""
    print(f"Fetching {city}...")
    weather_df = fetch_weather(lat, lon)
    air_df = fetch_air_quality(lat, lon)

    df = pd.merge(weather_df, air_df, on="timestamp", how="inner")
    df["city"] = city
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Missing values (same rule as etl/transform.py's handle_missing_values)
    df["pm25"] = df["pm25"].fillna(0)
    df["pm10"] = df["pm10"].fillna(0)
    df["aqi"] = df["aqi"].fillna(df["aqi"].mean())

    # Time features (same as etl/transform.py's add_time_features)
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    # History features -- computed for real here, since we have the full year
    df["aqi_yesterday"] = df["aqi"].shift(24)
    df["aqi_change_rate"] = (df["aqi"] - df["aqi"].shift(1)) / df["aqi"].shift(1)
    df["aqi_change_rate"] = df["aqi_change_rate"].replace([np.inf, -np.inf], np.nan)
    df["rolling_average_aqi"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    # Force float64 so an all-NaN stretch (e.g. first day, no "yesterday" yet)
    # doesn't get inferred as an unsupported 'null' type by Hopsworks
    for col in ["aqi_yesterday", "aqi_change_rate", "rolling_average_aqi"]:
        df[col] = df[col].astype("float64")

    # Match the feature group's EXISTING schema (locked in by the first
    # etl/load.py test insert, where pm25/pm10/hour/day/month came through
    # as whole numbers -> Hopsworks stored them as bigint/int64).
    # Open-Meteo returns pm25/pm10 as decimals, so we round before casting
    # (small precision tradeoff, worth knowing about -- it's not lossless).
    df["pm25"] = df["pm25"].round().astype("int64")
    df["pm10"] = df["pm10"].round().astype("int64")
    df["hour"] = df["hour"].astype("int64")
    df["day"] = df["day"].astype("int64")
    df["month"] = df["month"].astype("int64")

    return df


def load_to_hopsworks(df: pd.DataFrame):
    """Insert the full backfilled dataframe into the same feature group used live."""
    fs = get_feature_store()
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Lahore, Islamabad, Karachi, Peshawar",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )
    feature_group.insert(df)
    print(f"Inserted {len(df)} rows into '{FEATURE_GROUP_NAME}'.")


def main():
    all_city_dfs = []

    for city, (lat, lon) in CITIES.items():
        city_df = build_city_dataframe(city, lat, lon)
        all_city_dfs.append(city_df)
        time.sleep(1)  # be polite to the free API between cities

    full_df = pd.concat(all_city_dfs, ignore_index=True)
    print(f"Total rows to backfill: {len(full_df)} "
          f"({len(CITIES)} cities x ~{len(full_df) // len(CITIES)} hours)")

    load_to_hopsworks(full_df)


if __name__ == "__main__":
    main()