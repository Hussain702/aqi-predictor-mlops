"""
Step 8 - Feature Pipeline DAG (TaskFlow API)
------------------------------------------------
Runs extract -> transform -> load every hour, forever.

Improvement over the earlier version: the transform task now pulls the
last 48 hours of real history from Hopsworks before calling transform(),
so aqi_yesterday / aqi_change_rate / rolling_average_aqi get REAL values
on every hourly run -- not NaN like when transform() was called with
history_df=None.

Why 48 hours, not fewer: rolling_average_aqi is a 24-hour rolling window,
and aqi_yesterday looks exactly 24 hours back -- 48 hours of buffer
comfortably covers both even if an hourly run is occasionally late/missed.

catchup=False on purpose: historical data is handled separately by
backfill.py (the 2025 4-city dataset). This DAG only adds NEW data
going forward.
"""

from airflow.decorators import dag, task
from datetime import datetime

from etl.extract import extract
from etl.transform import transform
from etl.load import load
from utils.hopsworks_client import get_feature_store


@dag(
    dag_id="feature_pipeline",
    start_date=datetime(2026, 8, 2),
    schedule="@hourly",
    catchup=False,
    is_paused_upon_creation=False,
    tags=["aqi-predictor", "feature-pipeline"],
)
def feature_pipeline_dag():

    @task
    def extraction():
        return extract()

    @task
    def transformation(record):
        fs = get_feature_store()

        fg = fs.get_feature_group(
            name="aqi_features",
            version=1,
        )

        # Filter to THIS record's city before reading, so rolling/lag
        # features (aqi_yesterday, aqi_change_rate, rolling_average_aqi)
        # never mix another city's history in -- important now that this
        # matters for correct per-city predictions, and essential if this
        # DAG is ever extended to run hourly for more than one city.
        query = fg.select_all().filter(fg.city == record["city"])

        history_df = (
            query.read()
            .sort_values("timestamp")
            .tail(48)  # last 48 hourly records (~2 days) -- covers both the
                       # 24h lookback and the 24h rolling window with buffer
        )

        return transform(record, history_df)

    @task
    def loading(record):
        load(record)

    raw_record = extraction()
    transformed_record = transformation(raw_record)
    loading(transformed_record)


feature_pipeline_dag()