"""
Step 8 - Feature Pipeline DAG (TaskFlow API, dynamic task mapping)
------------------------------------------------------------------------
Runs extract -> transform -> load every hour, forever, for ALL 4 cities.

extract() now returns a LIST of records (one per city). transformation
and loading use Airflow's dynamic task mapping (.expand()) to run one
task instance PER CITY automatically -- no hardcoded per-city task chain,
and if a 5th city is ever added to etl/extract.py's CITIES list, this DAG
handles it with zero changes.

The transform task pulls the last 48 hours of real history from Hopsworks,
filtered to that record's own city (so Karachi's history never leaks into
Lahore's rolling/lag features, and vice versa), before calling transform().

Why 48 hours, not fewer: rolling_average_aqi is a 24-hour rolling window,
and aqi_yesterday looks exactly 24 hours back -- 48 hours of buffer
comfortably covers both even if an hourly run is occasionally late/missed.

catchup=False on purpose: historical data is handled separately by
backfill.py (the 2025 4-city dataset). This DAG only adds NEW data
going forward.

max_active_tis_per_dag=1 on transformation/loading (below): by default,
dynamic task mapping runs all 4 cities' instances of a task CONCURRENTLY.
Each one independently opens its own Hopsworks connection (pyarrow,
grpcio, boto3 and friends) -- 4 of those at once was enough to exceed
available container memory and get one process OOM-killed mid-run. That
showed up as a confusing Airflow-internal error ("executor reported
failed, but state attribute is queued") rather than a normal Python
exception, because the killed process never got the chance to report its
own failure. Capping this to 1 makes the 4 cities run one at a time --
slightly slower, but each city's ETL takes seconds, so for an hourly
schedule this costs nothing that matters and removes the resource race
entirely.
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
        return extract()  # list of records, one per city

    @task(max_active_tis_per_dag=1)
    def transformation(record):
        fs = get_feature_store()

        fg = fs.get_feature_group(
            name="aqi_features",
            version=1,
        )

        # Filter to THIS record's city before reading, so rolling/lag
        # features (aqi_yesterday, aqi_change_rate, rolling_average_aqi)
        # never mix another city's history in.
        query = fg.select_all().filter(fg.city == record["city"])

        history_df = (
            query.read()
            .sort_values("timestamp")
            .tail(48)  # last 48 hourly records (~2 days) -- covers both the
                       # 24h lookback and the 24h rolling window with buffer
        )

        return transform(record, history_df)

    @task(max_active_tis_per_dag=1)
    def loading(record):
        load(record)

    raw_records = extraction()
    transformed_records = transformation.expand(record=raw_records)
    loading.expand(record=transformed_records)


feature_pipeline_dag()