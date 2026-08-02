"""
Step 3 - LOAD
--------------
Takes a transformed record (output of transform.py) and saves it into the
Hopsworks Feature Store.

"""

import pandas as pd

from etl.extract import extract
from etl.transform import transform
from utils.hopsworks_client import get_feature_store

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def get_or_create_feature_group(fs):

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Lahore",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=False,  # keep it simple for now; can enable later for the Flask API
        time_travel_format="HUDI", 
    )
    return feature_group


def load(record: dict):
    """Insert a single transformed record into the Hopsworks Feature Store."""
    fs = get_feature_store()
    feature_group = get_or_create_feature_group(fs)

    # Hopsworks expects a DataFrame, even for a single row
    df = pd.DataFrame([record])
    df["timestamp"] = pd.to_datetime(df["timestamp"])


    for col in ["aqi_yesterday", "aqi_change_rate", "rolling_average_aqi"]:
        df[col] = df[col].astype("float64")

    feature_group.insert(df)
    print(f"Inserted 1 record into '{FEATURE_GROUP_NAME}' feature group.")


if __name__ == "__main__":
    raw_record = extract()

    if raw_record is None:
        print("Stopping: extract() failed, nothing to load.")
    else:
      
        transformed_record = transform(raw_record, history_df=None)
        load(transformed_record)
