from airflow.decorators import dag, task
from datetime import datetime

from etl.extract import extract
from etl.transform import transform
from etl.load import load
from utils.hopsworks_client import get_feature_store


@dag(
    start_date=datetime(2026, 8, 2),
    schedule="@hourly",
    catchup=False,
    is_paused_upon_creation=False,
)
def first_orchestrator():

    @task
    def extraction():
        return extract()

    @task
    def transformation(record):

        fs = get_feature_store()

        fg = fs.get_feature_group(
            name="aqi_features",
            version=1
        )

        query = fg.select_all()

        history_df = (
           query.read()
           .sort_values("timestamp")
           .tail(48)      # last 48 hourly records (~2 days)
        )

        return transform(record, history_df)

    @task
    def loading(record):
        load(record)

    raw_record = extraction()
    transformed_record = transformation(raw_record)
    loading(transformed_record)


first_orchestrator()