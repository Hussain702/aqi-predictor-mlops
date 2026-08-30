

import pandas as pd

from etl.extract import extract
from etl.transform import transform
from utils.hopsworks_client import get_feature_store

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def get_or_create_feature_group(fs):
    """
    Get the 'aqi_features' feature group if it exists, otherwise create it.

    primary_key=['city']   -> identifies which entity each row is about
    event_time='timestamp' -> tells Hopsworks this is time-series data
    """
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Lahore",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=False,  # keep it simple for now; can enable later for the Flask API
        time_travel_format="HUDI",  # standard format for plain-Python (non-Spark) clients;
                                     # avoids needing the extra 'delta' library
    )
    return feature_group


def load(record: dict):
    """Insert a single transformed record into the Hopsworks Feature Store."""
    fs = get_feature_store()
    feature_group = get_or_create_feature_group(fs)

    # Hopsworks expects a DataFrame, even for a single row
    df = pd.DataFrame([record])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Safety net: force these to float64 explicitly. If they're all NaN
    # (e.g. very first run, no history yet), pandas/pyarrow can otherwise
    # infer an ambiguous 'null' type that Hopsworks rejects.
    for col in ["aqi_yesterday", "aqi_change_rate", "rolling_average_aqi"]:
        df[col] = df[col].astype("float64")

    feature_group.insert(df)
    print(f"Inserted 1 record into '{FEATURE_GROUP_NAME}' feature group.")


if __name__ == "__main__":
    raw_records = extract()

    if not raw_records:
        print("Stopping: extract() returned no records for any city.")
    else:
        # No history yet in this simple standalone run -> history features = NaN
        # (the Airflow DAG pulls real history from Hopsworks per city instead)
        for raw_record in raw_records:
            transformed_record = transform(raw_record, history_df=None)
            load(transformed_record)